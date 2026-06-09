"""
多轮证据检索、跨文档定位
使用 BM25 + 规则索引，无 Embedding
"""
from typing import Optional

from .indexer import BM25Index, RuleBasedIndex
from .config import TOP_K_RETRIEVAL, MAX_EVIDENCE_CHARS


class MultiRoundRetriever:
    """
    多轮检索器
    支持：关键词检索、章节定位、条款匹配、跨文档检索
    """

    def __init__(self, bm25_index: BM25Index, rule_index: RuleBasedIndex):
        self.bm25 = bm25_index
        self.rule_index = rule_index
        self.retrieval_history: list[dict] = []  # 检索历史

    def retrieve(self, query: str, context: Optional[str] = None,
                 strategy: str = "hybrid") -> list[dict]:
        """
        执行检索

        Args:
            query: 检索查询
            context: 上下文信息（用于多轮检索）
            strategy: 检索策略 (bm25/rule/hybrid)

        Returns:
            检索结果列表
        """
        results = []

        if strategy == "bm25":
            results = self._bm25_retrieve(query)
        elif strategy == "rule":
            results = self._rule_retrieve(query)
        elif strategy == "hybrid":
            # 混合检索：先规则精确匹配，再 BM25 模糊匹配
            rule_results = self._rule_retrieve(query)
            bm25_results = self._bm25_retrieve(query)

            # 合并去重
            seen_ids = set()
            for doc in rule_results + bm25_results:
                doc_id = doc.get("chunk_id") or doc.get("doc_id")
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    results.append(doc)

        # 记录检索历史
        self.retrieval_history.append({
            "query": query,
            "strategy": strategy,
            "results_count": len(results)
        })

        return results[:TOP_K_RETRIEVAL]

    def retrieve_with_expansion(self, query: str, expanded_terms: list[str]) -> list[dict]:
        """
        使用扩展词进行检索（用于多轮检索中的查询扩展）

        Args:
            query: 原始查询
            expanded_terms: 扩展的检索词

        Returns:
            检索结果
        """
        all_results = []

        # 原始查询检索
        all_results.extend(self._bm25_retrieve(query))

        # 扩展词检索
        for term in expanded_terms:
            all_results.extend(self._bm25_retrieve(term))

        # 去重排序
        seen_ids = set()
        unique_results = []
        for doc in all_results:
            doc_id = doc.get("chunk_id") or doc.get("doc_id")
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(doc)

        return unique_results[:TOP_K_RETRIEVAL]

    def retrieve_by_document(self, doc_name: str, query: str) -> list[dict]:
        """
        在指定文档中检索

        Args:
            doc_name: 文档名称
            query: 检索查询

        Returns:
            该文档中的相关结果
        """
        all_results = self._bm25_retrieve(query)
        return [r for r in all_results if r.get("file_name") == doc_name]

    def _bm25_retrieve(self, query: str) -> list[dict]:
        """BM25 关键词检索"""
        return self.bm25.search(query, top_k=TOP_K_RETRIEVAL)

    def _rule_retrieve(self, query: str) -> list[dict]:
        """规则检索：章节、条款、关键词"""
        results = []

        # 章节匹配
        import re
        sections = re.findall(r'第[一二三四五六七八九十百千]+[章节条款]', query)
        for section in sections:
            results.extend(self.rule_index.search_by_section(section))

        # 条款匹配
        clauses = re.findall(r'第?\d+[条款]', query)
        for clause in clauses:
            results.extend(self.rule_index.search_by_clause(clause))

        # 关键词匹配
        keywords = self.rule_index._extract_keywords(query)
        for kw in keywords:
            results.extend(self.rule_index.search_by_keyword(kw))

        return results

    def format_evidence(self, results: list[dict]) -> str:
        """
        格式化检索结果为证据文本

        Args:
            results: 检索结果

        Returns:
            格式化的证据字符串
        """
        if not results:
            return "未找到相关证据"

        evidence_parts = []
        total_chars = 0

        for i, result in enumerate(results, 1):
            source = result.get("file_name", "未知来源")
            page = result.get("page_num", "?")
            text = result.get("text", "")
            score = result.get("score", 0)

            # 控制总长度
            if total_chars + len(text) > MAX_EVIDENCE_CHARS:
                break

            evidence_parts.append(
                f"[证据{i}] 来源: {source} (第{page}页)\n"
                f"相关度: {score:.2f}\n"
                f"内容: {text}\n"
            )
            total_chars += len(text)

        return "\n".join(evidence_parts)

    def get_retrieval_stats(self) -> dict:
        """获取检索统计信息"""
        return {
            "total_queries": len(self.retrieval_history),
            "queries": self.retrieval_history
        }
