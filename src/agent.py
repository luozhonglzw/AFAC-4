"""
金融长文本问答 Agent
仅使用 Qwen3.6-plus，所有推理必须基于文档证据
支持超时降级、断点续跑、额度预警
"""
import json
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI

from .config import (
    API_KEY, QWEN_MODEL, QWEN_BASE_URL,
    MAX_TOKENS, TEMPERATURE, TOP_P,
    MAX_RETRIES, RETRY_BACKOFF, REQUEST_TIMEOUT, MAX_CONCURRENCY,
    TOTAL_TOKEN_BUDGET, DOMAINS,
)
from .indexer import KeywordIndex, RuleIndex, SectionIndex, search
from .compressor import (
    ChunkCompressor, EvidenceAggregator, ContextBuilder, MemoryManager,
)
from .answer_formatter import normalize_answer, validate_answer, extract_json_answer

logger = logging.getLogger(__name__)

# 断点续跑：checkpoint 文件路径
CHECKPOINT_FILE = Path(__file__).parent.parent / "output" / "checkpoint.json"
FAILED_QUESTIONS_FILE = Path(__file__).parent.parent / "output" / "failed_questions.json"


class FinancialAgent:
    """
    金融文档问答 Agent 主类。

    流程：解析题目 -> 检索 -> 压缩 -> 推理 -> 后处理
    """

    def __init__(
        self,
        ki: KeywordIndex,
        ri: RuleIndex,
        si: SectionIndex,
        token_budget: int = TOTAL_TOKEN_BUDGET,
    ):
        self.client = OpenAI(
            api_key=API_KEY,
            base_url=QWEN_BASE_URL,
            timeout=REQUEST_TIMEOUT,
        )
        self.ki = ki
        self.ri = ri
        self.si = si

        # 压缩与记忆
        self.compressor = ChunkCompressor()
        self.aggregator = EvidenceAggregator()
        self.context_builder = ContextBuilder()
        self.memory = MemoryManager(token_budget=token_budget)

        # 统计
        self.token_usage = {"prompt": 0, "completion": 0, "total": 0}

    # ====================================================================
    #  主流程
    # ====================================================================

    def solve(self, question_item: dict) -> dict:
        """
        处理单道题目，返回结果 dict。
        支持超时降级：Top5 -> Top3 -> Top1 -> 空

        Args:
            question_item: {
                "qid": str,
                "domain": str,
                "question": str,
                "options": {"A": "...", ...} or None,
                "answer_format": "mcq" | "multi" | "tf",
                "doc_ids": ["text01", ...] or None,  # A榜提供
            }

        Returns:
            {qid, answer, prompt_tokens, completion_tokens, total_tokens, evidence}
        """
        qid = question_item.get("qid", "")
        domain = question_item.get("domain", "")
        question = question_item.get("question", "")
        options = question_item.get("options")
        answer_format = question_item.get("answer_format", "mcq")
        doc_ids = question_item.get("doc_ids")

        # 1. 检索
        raw_chunks = self._retrieve(question, domain, doc_ids)
        retrieval_count = len(raw_chunks)

        # 2. 压缩（设置领域）
        self.compressor.set_domain(domain)
        evidence_chunks = self._compress(raw_chunks, question)
        compressed_len = sum(len(c.get("text", "")) for c in evidence_chunks)

        # 3. 记录高频法条
        for c in evidence_chunks:
            cn = c.get("clause_number", "")
            if cn:
                self.memory.record_clause(cn, c.get("text", "")[:200])

        # 4. 推理（带超时降级）
        raw_answer, usage = self._reason_with_fallback(
            question, options, evidence_chunks, domain, answer_format,
        )

        # 5. 后处理
        final_answer = self._post_process(raw_answer, answer_format)

        # 6. 构建 evidence 列表
        evidence_list = self.aggregator.build_evidence_list(evidence_chunks)

        return {
            "qid": qid,
            "answer": final_answer,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "evidence": evidence_list,
            # 调试字段
            "retrieval_count": retrieval_count,
            "compressed_len": compressed_len,
            "raw_model_output": raw_answer,
        }

    # ====================================================================
    #  检索阶段
    # ====================================================================

    def _retrieve(
        self,
        question: str,
        domain: str,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        混合检索：KeywordIndex(BM25) + RuleIndex(领域关键词) + SectionIndex(条款号)
        A榜有 doc_ids 时限定范围，B榜全库检索。
        """
        # 先按 domain 过滤索引中的 chunk
        domain_chunks = [c for c in self.ki.chunks if c.get("domain") == domain]

        all_hits = search(question, domain, self.ki, self.ri, self.si, top_k=20)

        # doc_id 标准化："text01" -> "1"
        def _normalize_doc_id(did: str) -> str:
            if did.startswith("text"):
                return did[4:].lstrip("0") or "0"
            return did

        # 如果有 doc_ids，优先保留指定文档的结果
        if doc_ids:
            norm_ids = {_normalize_doc_id(d) for d in doc_ids}
            primary = [h for h in all_hits if _normalize_doc_id(h.get("doc_id", "")) in norm_ids]
            secondary = [h for h in all_hits if _normalize_doc_id(h.get("doc_id", "")) not in norm_ids]
            all_hits = primary + secondary

        # 确保结果属于当前领域（兜底）
        if not all_hits:
            all_hits = domain_chunks[:10]

        return all_hits

    # ====================================================================
    #  B 榜盲测模式
    # ====================================================================

    def solve_blind(self, question_item: dict) -> dict:
        """
        B 榜模式：无 doc_ids，盲测检索候选文档。
        流程：跨域粗排 -> doc_id 聚合 -> 领域推断 -> 精排 -> 推理
        """
        qid = question_item.get("qid", "")
        domain_hint = question_item.get("domain", "")
        question = question_item.get("question", "")
        options = question_item.get("options")
        answer_format = question_item.get("answer_format", "mcq")

        # 1. 跨域粗排
        raw_chunks, inferred_domain = self._retrieve_blind(question, domain_hint)

        # 2. 压缩
        self.compressor.set_domain(inferred_domain)
        evidence_chunks = self._compress(raw_chunks, question)
        compressed_len = sum(len(c.get("text", "")) for c in evidence_chunks)

        # 3. 记录高频法条
        for c in evidence_chunks:
            cn = c.get("clause_number", "")
            if cn:
                self.memory.record_clause(cn, c.get("text", "")[:200])

        # 4. 推理
        raw_answer, usage = self._reason_with_fallback(
            question, options, evidence_chunks, inferred_domain, answer_format,
        )

        # 5. 后处理
        final_answer = self._post_process(raw_answer, answer_format)

        # 6. 构建 evidence 列表
        evidence_list = self.aggregator.build_evidence_list(evidence_chunks)

        return {
            "qid": qid,
            "answer": final_answer,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "evidence": evidence_list,
            "retrieval_count": len(raw_chunks),
            "compressed_len": compressed_len,
            "raw_model_output": raw_answer,
            "inferred_domain": inferred_domain,
        }

    def _retrieve_blind(self, question: str, domain_hint: str = "") -> tuple[list[dict], str]:
        """
        盲测检索：跨所有领域粗排 -> 全局 BM25 -> doc_id 聚合 -> 领域推断 -> 精排

        Returns:
            (chunks, inferred_domain)
        """
        all_domains = list(DOMAINS.keys())

        # 步骤 1：跨域粗排，每个领域取 Top 5
        all_candidates = []
        for domain in all_domains:
            try:
                hits = search(question, domain, self.ki, self.ri, self.si, top_k=5)
                for h in hits:
                    h["_search_domain"] = domain
                all_candidates.extend(hits)
            except Exception:
                continue

        # 步骤 1b：全局 BM25 检索（不过滤领域）
        global_hits = self.ki.search(question, top_k=15)
        for h in global_hits:
            h["_search_domain"] = h.get("domain", "")
            all_candidates.append(h)

        # 步骤 1c：实体匹配（题目中提到的文档名 -> doc_id）
        doc_name_patterns = re.findall(r'(?:fc_text|fin_text|ins_text|reg_text|res_text)_(\d+)', question)
        for num in doc_name_patterns:
            doc_id = f"text{int(num):02d}"
            matched = [c for c in self.ki.chunks if c.get("doc_id") == doc_id]
            for h in matched[:10]:
                h["_search_domain"] = h.get("domain", "")
                h["score"] = 10.0
                all_candidates.append(h)

        if not all_candidates:
            fallback = [c for c in self.ki.chunks if c.get("domain") == domain_hint][:10]
            return fallback, domain_hint

        # 步骤 2：按 doc_id 聚合分数
        doc_scores: dict[str, dict] = {}
        for cand in all_candidates:
            doc_id = cand.get("doc_id", "unknown")
            domain = cand.get("_search_domain", "")
            score = cand.get("score", 0)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"score": 0, "domain": domain, "count": 0}
            doc_scores[doc_id]["score"] += score
            doc_scores[doc_id]["count"] += 1

        # 步骤 3：推断领域（按聚合分数投票）
        domain_votes: dict[str, float] = {}
        for doc_id, info in doc_scores.items():
            d = info["domain"]
            domain_votes[d] = domain_votes.get(d, 0) + info["score"]

        # 如果有 domain_hint，加权
        if domain_hint and domain_hint in domain_votes:
            domain_votes[domain_hint] *= 1.5

        inferred_domain = max(domain_votes, key=domain_votes.get) if domain_votes else domain_hint

        # 步骤 4：取 Top 10 doc_id（增加候选数）
        ranked_docs = sorted(
            doc_scores.items(),
            key=lambda x: (x[1]["score"] * 0.6 + x[1]["count"] * 0.4),
            reverse=True,
        )[:10]
        top_doc_ids = {doc_id for doc_id, _ in ranked_docs}

        # 步骤 5：在推断领域内精排 + 全局 BM25 补充
        refined_hits = search(question, inferred_domain, self.ki, self.ri, self.si, top_k=15)

        # 也搜索第二可能领域
        if len(domain_votes) > 1:
            second_domain = sorted(domain_votes.items(), key=lambda x: x[1], reverse=True)[1][0]
            second_hits = search(question, second_domain, self.ki, self.ri, self.si, top_k=8)
            refined_hits.extend(second_hits)

        # 合并全局 BM25 结果
        global_refined = self.ki.search(question, top_k=10)
        for h in global_refined:
            if h not in refined_hits:
                refined_hits.append(h)

        # 优先保留 top_doc_ids 的 chunks
        primary = [h for h in refined_hits if h.get("doc_id") in top_doc_ids]
        secondary = [h for h in refined_hits if h.get("doc_id") not in top_doc_ids]
        result = primary + secondary

        if not result:
            result = all_candidates[:10]

        logger.info(f"[盲测] 推断领域: {inferred_domain}, 候选文档: {list(top_doc_ids)[:5]}")
        return result, inferred_domain

    # ====================================================================
    #  压缩阶段
    # ====================================================================

    def _compress(self, chunks: list[dict], question: str) -> list[dict]:
        """
        根据 Token 预算状态选择压缩级别：
        - normal:   标准压缩
        - moderate: 缩短证据长度
        - aggressive: 只保留含条款号/数字的句子
        """
        level = self.memory.compression_level()

        if level == "aggressive":
            self.compressor.relevance_threshold = 0.3
            self.compressor.keep_high_value = True
        elif level == "moderate":
            self.compressor.relevance_threshold = 0.1
            self.compressor.keep_high_value = True
        else:
            self.compressor.relevance_threshold = 0.0
            self.compressor.keep_high_value = True

        # 压缩
        compressed = self.compressor.compress_batch(chunks, query=question)

        # 聚合去重，保留 top
        aggregated = self.aggregator.aggregate(compressed)

        return aggregated

    # ====================================================================
    #  推理阶段（带超时降级）
    # ====================================================================

    def _reason_with_fallback(
        self,
        question: str,
        options: Optional[dict[str, str]],
        evidence_chunks: list[dict],
        domain: str,
        answer_format: str,
    ) -> tuple[str, dict]:
        """
        超时降级策略：
        Level 0: 正常 context（所有 chunks）
        Level 1: Top 3 chunks，缩短 prompt
        Level 2: Top 1 chunk + 题目摘要
        Level 3: 全部失败，返回空
        """
        # Level 0: 正常调用
        raw_answer, usage = self._try_reason(
            question, options, evidence_chunks, domain, answer_format, level=0
        )
        if raw_answer:
            return raw_answer, usage

        # Level 1: Top 3 chunks
        if len(evidence_chunks) > 3:
            logger.info(f"降级 Level 1: Top 3 chunks (原 {len(evidence_chunks)} 个)")
            top3 = evidence_chunks[:3]
            raw_answer, usage = self._try_reason(
                question, options, top3, domain, answer_format, level=1
            )
            if raw_answer:
                return raw_answer, usage

        # Level 2: Top 1 chunk
        if len(evidence_chunks) > 1:
            logger.info(f"降级 Level 2: Top 1 chunk")
            top1 = evidence_chunks[:1]
            raw_answer, usage = self._try_reason(
                question, options, top1, domain, answer_format, level=2
            )
            if raw_answer:
                return raw_answer, usage

        # Level 3: 全部失败
        logger.error(f"所有降级级别失败，返回空答案")
        self._record_failed(question, domain, "all_levels_failed")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _try_reason(
        self,
        question: str,
        options: Optional[dict[str, str]],
        evidence_chunks: list[dict],
        domain: str,
        answer_format: str,
        level: int = 0,
    ) -> tuple[str, dict]:
        """
        尝试单次推理调用。
        level > 0 时增加 temperature 以提高灵活性。
        """
        messages = self.context_builder.build_messages(
            question, options, evidence_chunks, domain, answer_format,
        )

        # 降级时增加 temperature
        temp = TEMPERATURE + level * 0.05

        raw_answer, usage = self._call_model(messages, temperature=temp)
        return raw_answer, usage

    def _call_model(
        self,
        messages: list[dict],
        temperature: float = TEMPERATURE,
    ) -> tuple[str, dict]:
        """
        调用 Qwen3.6-plus，含重试、退避、Token 计数。
        使用 RETRY_BACKOFF 序列进行退避。

        Returns: (answer_text, {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int})
        """
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=QWEN_MODEL,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=temperature,
                    top_p=TOP_P,
                )
                answer = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }

                # 累加统计
                self.token_usage["prompt"] += usage["prompt_tokens"]
                self.token_usage["completion"] += usage["completion_tokens"]
                self.token_usage["total"] += usage["total_tokens"]
                self.memory.record_usage(usage["prompt_tokens"], usage["completion_tokens"])

                return answer, usage

            except Exception as e:
                last_error = e
                # 使用预定义的退避序列
                if attempt < len(RETRY_BACKOFF):
                    wait = RETRY_BACKOFF[attempt]
                else:
                    wait = RETRY_BACKOFF[-1] * (2 ** (attempt - len(RETRY_BACKOFF) + 1))
                logger.warning(f"模型调用失败 (attempt {attempt+1}/{MAX_RETRIES}): {e}, 等待 {wait}s")
                time.sleep(wait)

        # 全部重试失败
        logger.error(f"模型调用最终失败: {last_error}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def _record_failed(self, question: str, domain: str, reason: str):
        """记录失败题目到 failed_questions.json"""
        try:
            FAILED_QUESTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            failed = []
            if FAILED_QUESTIONS_FILE.exists():
                with open(FAILED_QUESTIONS_FILE, "r", encoding="utf-8") as f:
                    failed = json.load(f)
            failed.append({
                "question": question[:100],
                "domain": domain,
                "reason": reason,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            with open(FAILED_QUESTIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(failed, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"记录失败题目时出错: {e}")

    # ====================================================================
    #  后处理
    # ====================================================================

    def _post_process(self, raw_answer: str, answer_format: str) -> str:
        """
        1. 从模型输出中提取 JSON 中的 answer 字段
        2. 格式校验（单选/多选/判断）
        3. 标准化
        """
        # 提取原始 answer 字段
        extracted = extract_json_answer(raw_answer, answer_format)

        # 标准化
        answer = normalize_answer(extracted, answer_format)

        # 校验
        if not validate_answer(answer, answer_format):
            logger.warning(f"答案格式不合法: extracted={extracted!r}, 原始={raw_answer[:100]}")
            # 再尝试一次：直接从原始文本中提取
            answer = normalize_answer(raw_answer, answer_format)

        if not answer:
            logger.warning(f"无法提取有效答案，返回默认 A")

        return answer

    # ====================================================================
    #  批量处理
    # ====================================================================

    def batch_solve(
        self,
        questions: list[dict],
        max_workers: int = MAX_CONCURRENCY,
        blind: bool = False,
    ) -> list[dict]:
        """
        并发处理多道题目，控制总 Token 消耗。

        Args:
            questions: 题目列表
            max_workers: 最大并发数
            blind: 是否使用盲测模式

        Returns:
            结果列表（顺序与输入一致）
        """
        results = [None] * len(questions)
        solve_fn = self.solve_blind if blind else self.solve

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, q in enumerate(questions):
                # 检查预算
                if self.memory.budget_status() == "exhausted":
                    logger.warning(f"Token 预算耗尽，第 {idx} 题起跳过")
                    results[idx] = {
                        "qid": q.get("qid", ""),
                        "answer": "A",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "evidence": [],
                    }
                    continue

                future = executor.submit(solve_fn, q)
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"题目 {idx} 处理失败: {e}")
                    q = questions[idx]
                    results[idx] = {
                        "qid": q.get("qid", ""),
                        "answer": "A",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "evidence": [],
                    }

        return results

    # ====================================================================
    #  工具
    # ====================================================================

    def get_stats(self) -> dict:
        """返回统计信息"""
        return {
            "token_usage": self.token_usage.copy(),
            "budget_status": self.memory.budget_status(),
            "remaining_budget": self.memory.remaining_budget(),
            "frequent_clauses": self.memory.get_frequent_clauses(min_hits=2),
        }
