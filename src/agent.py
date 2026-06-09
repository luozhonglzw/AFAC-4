"""
金融长文本问答 Agent
仅使用 Qwen3.6-plus，所有推理必须基于文档证据
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
    MAX_RETRIES, RETRY_BASE_DELAY, REQUEST_TIMEOUT, MAX_CONCURRENCY,
    TOTAL_TOKEN_BUDGET,
)
from .indexer import KeywordIndex, RuleIndex, SectionIndex, search
from .compressor import (
    ChunkCompressor, EvidenceAggregator, ContextBuilder, MemoryManager,
)
from .answer_formatter import normalize_answer, validate_answer, extract_json_answer

logger = logging.getLogger(__name__)


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

        # 2. 压缩
        evidence_chunks = self._compress(raw_chunks, question)
        compressed_len = sum(len(c.get("text", "")) for c in evidence_chunks)

        # 3. 记录高频法条
        for c in evidence_chunks:
            cn = c.get("clause_number", "")
            if cn:
                self.memory.record_clause(cn, c.get("text", "")[:200])

        # 4. 推理
        raw_answer, usage = self._reason(
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
    #  推理阶段
    # ====================================================================

    def _reason(
        self,
        question: str,
        options: Optional[dict[str, str]],
        evidence_chunks: list[dict],
        domain: str,
        answer_format: str,
    ) -> tuple[str, dict]:
        """
        调用 Qwen3.6-plus 进行推理。
        Returns: (raw_answer, usage_dict)
        """
        messages = self.context_builder.build_messages(
            question, options, evidence_chunks, domain, answer_format,
        )

        raw_answer, usage = self._call_model(messages)
        return raw_answer, usage

    def _call_model(
        self,
        messages: list[dict],
        temperature: float = TEMPERATURE,
    ) -> tuple[str, dict]:
        """
        调用 Qwen3.6-plus，含重试、退避、Token 计数。

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
                wait = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"模型调用失败 (attempt {attempt+1}/{MAX_RETRIES}): {e}, 等待 {wait}s")
                time.sleep(wait)

        # 全部重试失败
        logger.error(f"模型调用最终失败: {last_error}")
        return "", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

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
    ) -> list[dict]:
        """
        并发处理多道题目，控制总 Token 消耗。

        Args:
            questions: 题目列表
            max_workers: 最大并发数

        Returns:
            结果列表（顺序与输入一致）
        """
        results = [None] * len(questions)

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

                future = executor.submit(self.solve, q)
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
