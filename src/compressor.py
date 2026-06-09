"""
上下文压缩与记忆流转
所有压缩方法均为规则驱动，不调用大模型 API
"""
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from .config import MAX_EVIDENCE_CHARS, PROMPTS_DIR

# 正则：高信息量句子（含数字/金额/比例/日期/条款号）
RE_HIGH_VALUE = re.compile(
    r"(?:第[一二三四五六七八九十百零\d]+[条章节条款])"  # 条款号
    r"|\d[\d,]*\.?\d*\s*(?:万|亿|百万|千万)?\s*(?:元|美元|人民币)"  # 金额
    r"|\d+\.?\d*\s*%"  # 比例
    r"|\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"  # 日期
    r"|\d+\s*(?:天|日|个月|月|年)"  # 期限
)
# 表格行：以 | 开头
RE_TABLE_ROW = re.compile(r"^\|.+\|$")


# ====================================================================
#  ChunkCompressor：规则驱动的 chunk 压缩
# ====================================================================

class ChunkCompressor:
    """
    基于规则压缩单个 chunk，不调用模型。
    """

    def __init__(self, relevance_threshold: float = 0.0,
                 keep_high_value: bool = True):
        """
        Args:
            relevance_threshold: 关键词匹配度低于此值的句子被删除（0=保留全部）
            keep_high_value: 是否强制保留含数字/条款号的高信息量句子
        """
        self.relevance_threshold = relevance_threshold
        self.keep_high_value = keep_high_value

    def compress(self, chunk: dict, query: str = "") -> dict:
        """
        压缩单个 chunk，返回新 chunk（不修改原始）。

        Args:
            chunk: 原始 chunk dict
            query: 用户问题（用于计算相关性）

        Returns:
            压缩后的 chunk dict，增加 compressed=True 标记
        """
        text = chunk.get("text", "")
        is_table = chunk.get("is_table", False)

        if is_table:
            compressed_text = self._compress_table(text)
        else:
            compressed_text = self._compress_text(text, query)

        return {
            **chunk,
            "text": compressed_text,
            "original_len": len(text),
            "compressed_len": len(compressed_text),
            "compressed": True,
        }

    def compress_batch(self, chunks: list[dict], query: str = "") -> list[dict]:
        """批量压缩"""
        return [self.compress(c, query) for c in chunks]

    # ---------- 文本压缩 ----------

    def _compress_text(self, text: str, query: str) -> str:
        sentences = self._split_sentences(text)
        if not sentences:
            return text

        query_keywords = self._extract_keywords(query) if query else set()
        kept = []

        for sent in sentences:
            sent_stripped = sent.strip()
            if not sent_stripped:
                continue

            # 高信息量句子：强制保留
            if self.keep_high_value and RE_HIGH_VALUE.search(sent_stripped):
                kept.append(sent_stripped)
                continue

            # 关键词匹配度
            if query_keywords:
                matched = sum(1 for kw in query_keywords if kw in sent_stripped)
                ratio = matched / len(query_keywords) if query_keywords else 0
                if ratio >= self.relevance_threshold:
                    kept.append(sent_stripped)
            else:
                # 无 query 时保留全部
                kept.append(sent_stripped)

        return "\n".join(kept) if kept else text[:200]

    # ---------- 表格压缩 ----------

    def _compress_table(self, table_md: str) -> str:
        """
        长表格转关键行摘要：
        - 保留表头（前2行）
        - 保留含数字/金额的行
        - 超过 10 行时只保留前 5 行 + 尾行
        """
        lines = table_md.split("\n")
        if len(lines) <= 6:
            return table_md

        header = lines[:2]  # 表头 + 分隔行
        data_lines = lines[2:]

        # 保留含高信息量的行
        high_value = [l for l in data_lines if RE_HIGH_VALUE.search(l)]

        if len(high_value) >= 3:
            # 高信息量行足够多，只保留这些
            return "\n".join(header + high_value)

        # 否则保留前5行 + 尾行
        if len(data_lines) > 8:
            summary = data_lines[:5] + ["| ... |（共{}行）...|".format(len(data_lines))] + [data_lines[-1]]
            return "\n".join(header + summary)

        return table_md

    # ---------- 工具 ----------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """按中文句号、分号、换行分句"""
        return re.split(r"[。；\n]", text)

    @staticmethod
    def _extract_keywords(text: str) -> set[str]:
        """简单关键词提取（2字以上连续中文）"""
        return {m for m in re.findall(r"[一-鿿]{2,}", text)}


# ====================================================================
#  EvidenceAggregator：多轮证据合并去重
# ====================================================================

class EvidenceAggregator:
    """
    合并多轮检索的证据，去重，按 doc_id 分组保留最相关的。
    """

    def __init__(self, max_per_doc: int = 5, max_total: int = 15):
        """
        Args:
            max_per_doc: 每个 doc_id 最多保留的 chunk 数
            max_total: 全局最多保留的 chunk 总数
        """
        self.max_per_doc = max_per_doc
        self.max_total = max_total

    def aggregate(self, evidence_chunks: list[dict]) -> list[dict]:
        """
        合并去重，按 doc_id 分组取 top，返回精简后的证据列表。
        """
        # 去重：(doc_id, clause_number, text前50字) 作为 key
        seen = set()
        unique = []
        for c in evidence_chunks:
            key = (c.get("doc_id", ""), c.get("clause_number", ""),
                   c.get("text", "")[:50])
            if key not in seen:
                seen.add(key)
                unique.append(c)

        # 按 doc_id 分组
        by_doc: dict[str, list[dict]] = {}
        for c in unique:
            doc_id = c.get("doc_id", "unknown")
            by_doc.setdefault(doc_id, []).append(c)

        # 每组按 score 降序取 top
        result = []
        for doc_id, chunks in by_doc.items():
            chunks.sort(key=lambda x: x.get("score", 0), reverse=True)
            result.extend(chunks[:self.max_per_doc])

        # 全局截断
        result.sort(key=lambda x: x.get("score", 0), reverse=True)
        return result[:self.max_total]

    def build_evidence_list(self, chunks: list[dict]) -> list[dict]:
        """
        生成 evidence_list 格式：[{doc_id, clause, relevance_score, text_preview}]
        """
        evidence_list = []
        for c in chunks:
            evidence_list.append({
                "doc_id": c.get("doc_id", ""),
                "clause": c.get("clause_number", ""),
                "section": c.get("section", ""),
                "relevance_score": round(c.get("score", 0), 3),
                "text_preview": c.get("text", "")[:150],
            })
        return evidence_list


# ====================================================================
#  ContextBuilder：构建送入模型的 prompt 上下文
# ====================================================================

class ContextBuilder:
    """
    构建 prompt：system（领域专用） + evidence（压缩后） + question + options
    控制单轮 prompt 总长度不超过 max_chars（估算中文字符 ≈ token）
    """

    # 领域 -> prompt 文件映射
    DOMAIN_PROMPT_MAP = {
        "insurance": "insurance.txt",
        "regulatory": "regulatory.txt",
        "financial_contracts": "financial_contracts.txt",
        "financial_reports": "financial_reports.txt",
        "research": "research.txt",
    }

    def __init__(self, max_chars: int = 8000):
        """
        Args:
            max_chars: 单轮 prompt 最大字符数（中文字符 ≈ 1 token）
        """
        self.max_chars = max_chars
        self._prompt_cache: dict[str, str] = {}

    def build_system_prompt(self, domain: str) -> str:
        """加载领域专用 system prompt"""
        if domain in self._prompt_cache:
            return self._prompt_cache[domain]

        filename = self.DOMAIN_PROMPT_MAP.get(domain)
        if filename:
            path = PROMPTS_DIR / filename
            if path.exists():
                prompt = path.read_text(encoding="utf-8").strip()
                self._prompt_cache[domain] = prompt
                return prompt

        # fallback
        fallback = "你是一个金融文档问答助手，请严格根据提供的文档证据回答问题。"
        self._prompt_cache[domain] = fallback
        return fallback

    def build_user_prompt(
        self,
        question: str,
        options: Optional[dict[str, str]],
        evidence_chunks: list[dict],
        answer_format: str = "mcq",
    ) -> str:
        """
        构建 user prompt：证据 + 题目 + 选项 + 格式要求。

        Args:
            question: 题目文本
            options: {"A": "...", "B": "...", ...} 或 None（判断题可能无选项）
            evidence_chunks: 压缩后的证据 chunk 列表
            answer_format: "mcq" / "multi" / "tf"
        """
        parts = []

        # 1. 证据
        evidence_text = self._format_evidence(evidence_chunks)
        parts.append(f"## 文档证据\n{evidence_text}")

        # 2. 题目
        parts.append(f"## 题目\n{question}")

        # 3. 选项
        if options:
            opts = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
            parts.append(f"## 选项\n{opts}")

        # 4. 格式要求
        format_instruction = self._get_format_instruction(answer_format)
        parts.append(f"## 答案格式要求\n{format_instruction}")

        user_prompt = "\n\n".join(parts)

        # 5. 截断控制
        if len(user_prompt) > self.max_chars:
            # 超长时截断证据部分
            overflow = len(user_prompt) - self.max_chars
            evidence_text = evidence_text[:len(evidence_text) - overflow - 100] + "\n...（证据过长，已截断）"
            parts[0] = f"## 文档证据\n{evidence_text}"
            user_prompt = "\n\n".join(parts)

        return user_prompt

    def build_messages(
        self,
        question: str,
        options: Optional[dict[str, str]],
        evidence_chunks: list[dict],
        domain: str,
        answer_format: str = "mcq",
    ) -> list[dict]:
        """构建完整的 messages 列表（system + user）"""
        return [
            {"role": "system", "content": self.build_system_prompt(domain)},
            {"role": "user", "content": self.build_user_prompt(
                question, options, evidence_chunks, answer_format,
            )},
        ]

    def estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数（中文字符 ≈ 1 token，英文单词 ≈ 1 token）"""
        cn_chars = len(re.findall(r"[一-鿿]", text))
        en_words = len(re.findall(r"[a-zA-Z]+", text))
        other = len(text) - cn_chars - en_words
        return cn_chars + en_words + other // 4

    # ---------- 内部 ----------

    def _format_evidence(self, chunks: list[dict]) -> str:
        """格式化证据块"""
        if not chunks:
            return "（无相关证据）"

        parts = []
        for i, c in enumerate(chunks, 1):
            doc_id = c.get("doc_id", "未知")
            page = c.get("page", "?")
            clause = c.get("clause_number", "")
            section = c.get("section", "")
            text = c.get("text", "")
            score = c.get("score", 0)

            header = f"[证据{i}] 来源: {doc_id} (第{page}页)"
            if section:
                header += f" | 章节: {section}"
            if clause:
                header += f" | 条款: {clause}"
            if score:
                header += f" | 相关度: {score:.2f}"

            parts.append(f"{header}\n{text}")

        return "\n\n".join(parts)

    @staticmethod
    def _get_format_instruction(answer_format: str) -> str:
        if answer_format == "mcq":
            return (
                "本题为单选题。请先逐项分析每个选项的判断依据（引用具体条款/数据），\n"
                "然后给出最终答案。\n"
                "最终答案必须严格为以下 JSON 格式：\n"
                '{"reasoning": "分析过程", "answer": "A"}\n'
                "其中 answer 只能是 A/B/C/D 中的一个字母。"
            )
        elif answer_format == "multi":
            return (
                "本题为多选题。请先逐项分析每个选项的判断依据（引用具体条款/数据），\n"
                "然后给出所有正确选项。\n"
                "最终答案必须严格为以下 JSON 格式：\n"
                '{"reasoning": "分析过程", "answer": ["A", "C"]}\n'
                "其中 answer 为正确选项字母列表，可包含多个。"
            )
        elif answer_format == "tf":
            return (
                "本题为判断题。请先列出判断依据（引用具体条款/数据），\n"
                "然后给出最终判断。\n"
                "最终答案必须严格为以下 JSON 格式：\n"
                '{"reasoning": "分析过程", "answer": "A"}\n'
                "其中 A 表示正确，B 表示错误。"
            )
        else:
            return '请输出 JSON 格式：{"reasoning": "分析过程", "answer": "..."}'


# ====================================================================
#  MemoryManager：跨题目记忆与 Token 预算管理
# ====================================================================

class MemoryManager:
    """
    - 跨题目保留高频法条/条款缓存
    - LRU 缓存已查询的文档摘要
    - 记录 Token 消耗，接近预算时切换压缩策略
    """

    def __init__(
        self,
        token_budget: int = 5_000_000,
        lru_size: int = 256,
        warn_ratio: float = 0.8,
        critical_ratio: float = 0.95,
    ):
        self.token_budget = token_budget
        self.tokens_used = 0
        self.warn_ratio = warn_ratio
        self.critical_ratio = critical_ratio

        # LRU 缓存：query_hash -> compressed_evidence
        self._cache: OrderedDict[str, list[dict]] = OrderedDict()
        self._lru_size = lru_size

        # 高频法条缓存：clause_number -> {text, hit_count}
        self.clause_cache: dict[str, dict] = {}

    # ---------- Token 管理 ----------

    def record_usage(self, prompt_tokens: int, completion_tokens: int):
        """记录一次 API 调用的 token 消耗"""
        self.tokens_used += prompt_tokens + completion_tokens

    def remaining_budget(self) -> int:
        return max(0, self.token_budget - self.tokens_used)

    def budget_status(self) -> str:
        """返回 'normal' / 'warn' / 'critical' / 'exhausted'"""
        if self.tokens_used >= self.token_budget:
            return "exhausted"
        ratio = self.tokens_used / self.token_budget
        if ratio >= self.critical_ratio:
            return "critical"
        if ratio >= self.warn_ratio:
            return "warn"
        return "normal"

    def compression_level(self) -> str:
        """
        根据预算状态返回建议的压缩级别：
        - normal:  标准压缩
        - warn:    激进压缩（缩小证据长度）
        - critical: 最大压缩（只保留条款号 + 数字）
        """
        status = self.budget_status()
        if status in ("critical", "exhausted"):
            return "aggressive"
        if status == "warn":
            return "moderate"
        return "normal"

    # ---------- LRU 缓存 ----------

    def get_cached(self, query_key: str) -> Optional[list[dict]]:
        if query_key in self._cache:
            self._cache.move_to_end(query_key)
            return self._cache[query_key]
        return None

    def put_cache(self, query_key: str, evidence: list[dict]):
        self._cache[query_key] = evidence
        self._cache.move_to_end(query_key)
        if len(self._cache) > self._lru_size:
            self._cache.popitem(last=False)

    # ---------- 高频法条缓存 ----------

    def record_clause(self, clause_number: str, text: str):
        """记录被引用的法条"""
        if clause_number in self.clause_cache:
            self.clause_cache[clause_number]["hit_count"] += 1
        else:
            self.clause_cache[clause_number] = {"text": text, "hit_count": 1}

    def get_frequent_clauses(self, min_hits: int = 2) -> dict[str, str]:
        """返回被引用 min_hits 次以上的法条"""
        return {
            k: v["text"]
            for k, v in self.clause_cache.items()
            if v["hit_count"] >= min_hits
        }
