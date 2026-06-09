"""
无 Embedding 的混合索引：KeywordIndex + RuleIndex + SectionIndex
使用 jieba 中文分词 + BM25/TF-IDF 打分
严禁使用任何向量模型或 Embedding 模型
"""
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import jieba

from .config import INDEX_DIR, BM25_K1, BM25_B, DOMAINS

# ==================== 金融自定义词典 ====================
# 高频金融术语（4~8 字），jieba 默认词典切不准的
_FINANCE_TERMS = [
    # 保险
    "身故保险金", "现金价值", "犹豫期", "等待期", "免赔额", "保险金额",
    "保险责任", "责任免除", "受益人", "投保人", "被保险人", "保险合同",
    "保险期间", "续保", "退保", "减额交清", "保单贷款", "万能账户",
    "年金", "生存保险金", "重大疾病", "轻症", "中症", "豁免保费",
    # 监管
    "股东大会", "董事会", "独立董事", "监事会", "特别决议", "普通决议",
    "关联交易", "信息披露", "合规管理", "内部控制", "风险管理",
    "资本充足率", "流动性覆盖率", "净稳定资金比例",
    # 财务报告
    "营业收入", "营业成本", "净利润", "毛利率", "净利率",
    "经营活动", "投资活动", "筹资活动", "现金流量",
    "资产负债率", "流动比率", "速动比率", "净资产收益率",
    "每股收益", "市盈率", "市净率", "研发投入", "研发费用",
    # 金融合同
    "募集说明书", "评级报告", "信用评级", "违约责任",
    "偿付能力", "担保条款", "提前赎回", "回售条款",
    "票面利率", "发行价格", "到期收益率",
    # 研究报告
    "行业趋势", "市场份额", "同比增长", "环比增长",
    "渗透率", "集中度", "竞争格局", "商业模式",
]

# 条款号正则
RE_CLAUSE_NUM = re.compile(r"第[一二三四五六七八九十百零\d]+条")
# 章节号正则
RE_CHAPTER_NUM = re.compile(r"第[一二三四五六七八九十百零\d]+[章节]")
# 金额正则
RE_AMOUNT = re.compile(r"\d[\d,]*\.?\d*\s*(?:万|亿|百万|千万)?\s*(?:元|美元|欧元|港币|人民币)")
# 比例正则
RE_RATIO = re.compile(r"\d+\.?\d*\s*%|百分之[一二三四五六七八九十\d]+")

# 停用词（高频无意义虚词）
_STOPWORDS = frozenset([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那",
    "被", "从", "对", "把", "与", "以", "及", "等", "但", "或", "而",
    "如果", "因为", "所以", "虽然", "但是", "可以", "这个", "那个",
    "什么", "怎么", "哪个", "哪些", "多少", "是否", "能否", "应该",
    "其", "中", "为", "并", "于", "如", "所", "之", "则", "故",
])

# 为 jieba 添加金融词典
for _term in _FINANCE_TERMS:
    jieba.add_word(_term, freq=99999, tag="nz")


# ====================================================================
#  KeywordIndex：倒排索引 + BM25 打分
# ====================================================================

class KeywordIndex:
    """
    基于 jieba 分词的倒排索引 + BM25 打分。
    倒排表结构: term -> List[(chunk_key, position, tf)]
    """

    def __init__(self):
        # chunk_key = (doc_id, chunk_idx) 唯一标识一个 chunk
        self.chunks: list[dict] = []                          # 全部 chunk（按插入顺序）
        self.chunk_keys: list[tuple[str, int]] = []           # 对齐的 (doc_id, chunk_idx)
        self.inverted: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
        self.doc_freq: dict[str, int] = {}                    # term -> 含该词的 chunk 数
        self.doc_lens: list[int] = []                         # 每个 chunk 的 token 数
        self.avg_doc_len: float = 0.0
        self.is_built = False

    # ---------- 构建 ----------

    def add_chunks(self, chunks: list[dict]):
        """批量添加 chunk"""
        for i, c in enumerate(chunks):
            doc_id = c.get("doc_id", "")
            chunk_idx = c.get("chunk_idx", i)
            self.chunks.append(c)
            self.chunk_keys.append((doc_id, chunk_idx))
        self.is_built = False

    def build(self):
        """构建倒排索引"""
        self.inverted.clear()
        self.doc_freq.clear()
        self.doc_lens = []

        for idx, chunk in enumerate(self.chunks):
            tokens = self._tokenize(chunk["text"])
            self.doc_lens.append(len(tokens))

            # 记录每个 term 在该 chunk 中的出现次数
            tf_map = Counter(tokens)
            for term, tf in tf_map.items():
                self.inverted[term].append((self.chunk_keys[idx][0],
                                            self.chunk_keys[idx][1], tf))
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        n = len(self.chunks)
        self.avg_doc_len = sum(self.doc_lens) / n if n else 0
        self.is_built = True

    # ---------- 检索 ----------

    def search(self, query: str, top_k: int = 10,
               clause_boost: float = 2.0) -> list[dict]:
        """
        BM25 检索。
        对条款号（如"第四十七条"）给予额外权重 boost。
        """
        if not self.is_built:
            self.build()

        tokens = self._tokenize(query)
        if not tokens:
            return []

        # 提取条款号用于 boost
        clause_hits = set(RE_CLAUSE_NUM.findall(query))

        n = len(self.chunks)
        scores = [0.0] * n

        for term in tokens:
            posting = self.inverted.get(term)
            if not posting:
                continue

            df = self.doc_freq.get(term, 0)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            for doc_id, chunk_idx, tf in posting:
                # 定位到全局 idx
                idx = self._key_to_idx(doc_id, chunk_idx)
                if idx is None:
                    continue
                dl = self.doc_lens[idx]
                # BM25 tf-normalization
                tf_norm = (tf * (BM25_K1 + 1)) / (
                    tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avg_doc_len)
                )
                scores[idx] += idf * tf_norm

        # 条款号 boost：如果 chunk 的 clause_number 命中 query 中的条款号
        for idx, chunk in enumerate(self.chunks):
            cn = chunk.get("clause_number", "")
            if cn and cn in clause_hits:
                scores[idx] *= clause_boost

        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked[:top_k]:
            if scores[idx] <= 0:
                break
            results.append({**self.chunks[idx], "score": scores[idx]})
        return results

    # ---------- 分词 ----------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """jieba 分词 + 去停用词 + 提取条款号/金额/比例"""
        text = text.lower()
        # 先用正则提取特殊实体，保证不被切碎
        specials = RE_CLAUSE_NUM.findall(text) + RE_AMOUNT.findall(text) + RE_RATIO.findall(text)
        # jieba 精确模式
        tokens = list(jieba.cut(text, cut_all=False))
        # 去停用词、去空白
        tokens = [t.strip() for t in tokens if t.strip() and t.strip() not in _STOPWORDS]
        # 合并 specials（小写化）
        tokens.extend(s.lower() for s in specials)
        return tokens

    def extract_query_keywords(self, query: str) -> dict:
        """
        从 query 中提取结构化关键词。
        返回 {terms: [...], clauses: [...], amounts: [...], ratios: [...]}
        """
        return {
            "terms": [t for t in self._tokenize(query) if len(t) >= 2],
            "clauses": RE_CLAUSE_NUM.findall(query),
            "amounts": RE_AMOUNT.findall(query),
            "ratios": RE_RATIO.findall(query),
        }

    # ---------- 内部工具 ----------

    def _key_to_idx(self, doc_id: str, chunk_idx: int) -> Optional[int]:
        """chunk_key -> chunks 列表下标（首次构建时缓存）"""
        if not hasattr(self, "_key_index"):
            self._key_index = {k: i for i, k in enumerate(self.chunk_keys)}
        return self._key_index.get((doc_id, chunk_idx))

    # ---------- 序列化 ----------

    def save(self, path: Optional[Path] = None):
        if path is None:
            path = INDEX_DIR / "keyword_index.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": self.chunks,
            "chunk_keys": self.chunk_keys,
            "doc_freq": self.doc_freq,
            "doc_lens": self.doc_lens,
            "avg_doc_len": self.avg_doc_len,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: Optional[Path] = None):
        if path is None:
            path = INDEX_DIR / "keyword_index.pkl"
        if not path.exists():
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.chunk_keys = data["chunk_keys"]
        self.doc_freq = data["doc_freq"]
        self.doc_lens = data["doc_lens"]
        self.avg_doc_len = data["avg_doc_len"]
        # 重建倒排
        self.inverted.clear()
        for idx, chunk in enumerate(self.chunks):
            tokens = self._tokenize(chunk["text"])
            tf_map = Counter(tokens)
            for term, tf in tf_map.items():
                self.inverted[term].append((
                    self.chunk_keys[idx][0], self.chunk_keys[idx][1], tf
                ))
        self.is_built = True


# ====================================================================
#  RuleIndex：领域预定义关键词精确匹配
# ====================================================================

# 领域 -> 预定义关键词列表
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "insurance": [
        "身故保险金", "现金价值", "退保", "犹豫期", "免赔额",
        "保险责任", "责任免除", "受益人", "保险金额", "等待期",
        "续保", "保单贷款", "万能账户", "年金", "豁免保费",
    ],
    "regulatory": [
        "股东大会", "董事会", "独立董事", "特别决议", "普通决议",
        "关联交易", "信息披露", "合规管理", "内部控制", "监事会",
        "资本充足率", "流动性覆盖率", "净稳定资金比例",
    ],
    "financial_contracts": [
        "债券", "募集说明书", "评级", "违约", "偿付",
        "信用评级", "票面利率", "担保条款", "提前赎回", "回售条款",
        "发行价格", "到期收益率", "违约责任",
    ],
    "financial_reports": [
        "营业收入", "净利润", "现金流", "研发投入", "资产负债率",
        "毛利率", "净利率", "经营活动", "投资活动", "筹资活动",
        "每股收益", "市盈率", "市净率", "净资产收益率",
    ],
    "research": [
        "行业趋势", "市场份额", "同比增长", "毛利率", "预测",
        "环比增长", "渗透率", "集中度", "竞争格局", "商业模式",
        "渗透率", "景气度", "龙头",
    ],
}


class RuleIndex:
    """
    按领域关键词精确匹配。
    结构: domain -> keyword -> set(chunk_idx)
    """

    def __init__(self):
        self.index: dict[str, dict[str, set[int]]] = {
            d: defaultdict(set) for d in DOMAIN_KEYWORDS
        }

    def build(self, chunks: list[dict]):
        """
        遍历 chunk，按其所属 domain 将 chunk_idx 注册到命中的关键词下。
        chunk 需包含 domain 字段（由 build_index 写入）。
        """
        for idx, chunk in enumerate(chunks):
            domain = chunk.get("domain", "")
            if domain not in self.index:
                continue
            text = chunk["text"]
            for kw in DOMAIN_KEYWORDS[domain]:
                if kw in text:
                    self.index[domain][kw].add(idx)

    def search(self, query: str, domain: str, chunks: list[dict]) -> list[dict]:
        """
        返回 query 中命中领域关键词的 chunk（按命中关键词数排序）。
        """
        if domain not in self.index:
            return []

        # 从 query 中提取出现在领域关键词表中的词
        hit_keywords = [kw for kw in DOMAIN_KEYWORDS[domain] if kw in query]
        if not hit_keywords:
            # query 没有直接命中领域关键词，尝试用 query 分词后匹配
            tokens = set(jieba.cut(query))
            for kw in DOMAIN_KEYWORDS[domain]:
                if kw in tokens or any(kw in t for t in tokens):
                    hit_keywords.append(kw)

        # 收集命中 chunk_idx 及命中次数
        hit_counter: Counter = Counter()
        for kw in hit_keywords:
            for idx in self.index[domain].get(kw, set()):
                hit_counter[idx] += 1

        # 按命中数降序
        ranked = hit_counter.most_common()
        return [{**chunks[idx], "rule_hits": count} for idx, count in ranked]


# ====================================================================
#  SectionIndex：章节标题 -> chunk_ids 快速定位
# ====================================================================

class SectionIndex:
    """
    章节/条款号 -> chunk_idx 列表
    """

    def __init__(self):
        self.section_map: dict[str, list[int]] = defaultdict(list)
        self.clause_map: dict[str, list[int]] = defaultdict(list)

    def build(self, chunks: list[dict]):
        for idx, chunk in enumerate(chunks):
            # 章节
            sec = chunk.get("section", "")
            if sec:
                self.section_map[sec].append(idx)
            # 条款号
            cn = chunk.get("clause_number", "")
            if cn:
                self.clause_map[cn].append(idx)

    def search_by_section(self, section: str, chunks: list[dict]) -> list[dict]:
        """按章节标题检索"""
        idxs = self.section_map.get(section, [])
        return [chunks[i] for i in idxs]

    def search_by_clause(self, clause: str, chunks: list[dict]) -> list[dict]:
        """按条款号精确检索（如"第四十七条"）"""
        idxs = self.clause_map.get(clause, [])
        return [chunks[i] for i in idxs]

    def search(self, query: str, chunks: list[dict]) -> list[dict]:
        """
        从 query 中提取条款号/章节号，返回精确匹配的 chunk。
        """
        results = []
        seen = set()

        # 条款号精确匹配（优先）
        for clause in RE_CLAUSE_NUM.findall(query):
            for idx in self.clause_map.get(clause, []):
                if idx not in seen:
                    seen.add(idx)
                    results.append({**chunks[idx], "match_type": "clause", "match_key": clause})

        # 章节号匹配
        for chap in RE_CHAPTER_NUM.findall(query):
            for idx in self.section_map.get(chap, []):
                if idx not in seen:
                    seen.add(idx)
                    results.append({**chunks[idx], "match_type": "chapter", "match_key": chap})

        return results


# ====================================================================
#  构建入口
# ====================================================================

def build_index(processed_dir: Optional[Path] = None) -> tuple[KeywordIndex, RuleIndex, SectionIndex]:
    """
    遍历所有 JSONL，构建三个索引并保存。

    Args:
        processed_dir: processed 目录路径；默认遍历全部领域

    Returns:
        (keyword_index, rule_index, section_index)
    """
    all_chunks = _load_all_chunks(processed_dir)

    # 为每个 chunk 写入 domain 字段（如果还没有）
    _tag_domain(all_chunks)

    ki = KeywordIndex()
    ki.add_chunks(all_chunks)
    ki.build()
    ki.save()

    ri = RuleIndex()
    ri.build(all_chunks)

    si = SectionIndex()
    si.build(all_chunks)

    return ki, ri, si


def search(
    query: str,
    domain: str,
    ki: KeywordIndex,
    ri: RuleIndex,
    si: SectionIndex,
    top_k: int = 10,
) -> list[dict]:
    """
    混合检索入口：
    1. SectionIndex 精确匹配条款号/章节号
    2. RuleIndex 领域关键词匹配
    3. KeywordIndex BM25 打分
    4. 合并去重，按综合分排序
    """
    all_chunks = ki.chunks
    results: dict[int, dict] = {}  # idx -> {chunk..., final_score}

    def _merge(idx: int, extra_score: float, source: str):
        if idx in results:
            results[idx]["final_score"] += extra_score
            results[idx]["sources"].add(source)
        else:
            results[idx] = {
                **all_chunks[idx],
                "final_score": extra_score,
                "sources": {source},
            }

    # 1) 章节/条款精确匹配（最高权重）
    sec_hits = si.search(query, all_chunks)
    for hit in sec_hits:
        # 在 all_chunks 中找到对应 idx
        for idx, c in enumerate(all_chunks):
            if c is hit:
                _merge(idx, 10.0, "section")
                break

    # 2) RuleIndex 领域关键词
    rule_hits = ri.search(query, domain, all_chunks)
    for hit in rule_hits:
        rule_score = hit.get("rule_hits", 1) * 3.0
        for idx, c in enumerate(all_chunks):
            if c is hit:
                _merge(idx, rule_score, "rule")
                break

    # 3) KeywordIndex BM25
    bm25_hits = ki.search(query, top_k=top_k * 2)
    for hit in bm25_hits:
        bm25_score = hit.get("score", 0)
        for idx, c in enumerate(all_chunks):
            if c.get("doc_id") == hit.get("doc_id") and c.get("text") == hit.get("text"):
                _merge(idx, bm25_score, "bm25")
                break

    # 排序
    ranked = sorted(results.values(), key=lambda x: x["final_score"], reverse=True)

    # 对 regulatory 领域：条款号精确匹配的结果强制置顶
    if domain == "regulatory":
        clause_hits = [r for r in ranked if "section" in r["sources"]]
        others = [r for r in ranked if "section" not in r["sources"]]
        ranked = clause_hits + others

    return ranked[:top_k]


# ====================================================================
#  内部工具
# ====================================================================

def _load_all_chunks(processed_dir: Optional[Path] = None) -> list[dict]:
    """加载所有 JSONL 文件"""
    chunks = []
    if processed_dir:
        dirs = [processed_dir]
    else:
        dirs = [DOMAINS[d]["processed_dir"] for d in DOMAINS]

    for d in dirs:
        if not d.exists():
            continue
        for jsonl_path in sorted(d.rglob("*.jsonl")):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        chunks.append(json.loads(line))
    return chunks


def _tag_domain(chunks: list[dict]):
    """
    根据 chunk 所属的 JSONL 路径推断 domain，写入 chunk["domain"]。
    """
    for chunk in chunks:
        if "domain" in chunk:
            continue
        doc_id = chunk.get("doc_id", "")
        # 根据 processed_dir 子目录推断
        # 简单逻辑：遍历 DOMAINS 看 processed_dir 下是否有该 doc_id
        chunk["domain"] = _guess_domain(doc_id)


def _guess_domain(doc_id: str) -> str:
    """根据 doc_id 猜测所属领域"""
    lower = doc_id.lower()
    if any(k in lower for k in ("insurance", "保险", "保单", "寿险")):
        return "insurance"
    if any(k in lower for k in ("regulatory", "监管", "办法", "规定", "令")):
        return "regulatory"
    if any(k in lower for k in ("contract", "合同", "债券", "募集")):
        return "financial_contracts"
    if any(k in lower for k in ("report", "报告", "年报", "财报")):
        return "financial_reports"
    if any(k in lower for k in ("research", "研报", "研究", "分析")):
        return "research"
    return ""
