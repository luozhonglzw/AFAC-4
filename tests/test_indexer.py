"""
索引模块测试
"""
import pytest
from src.indexer import BM25Index, RuleBasedIndex


class TestBM25Index:
    """BM25 索引测试"""

    def test_add_documents(self):
        """测试添加文档"""
        index = BM25Index()
        chunks = [
            {"chunk_id": 0, "file_name": "test.pdf", "page_num": 1,
             "text": "保险合同中的犹豫期为10天"},
            {"chunk_id": 1, "file_name": "test.pdf", "page_num": 2,
             "text": "保费缴纳方式为年缴"}
        ]
        index.add_documents(chunks)
        assert len(index.documents) == 2

    def test_build_and_search(self):
        """测试构建索引和检索"""
        index = BM25Index()
        chunks = [
            {"chunk_id": 0, "file_name": "test.pdf", "page_num": 1,
             "text": "保险合同中的犹豫期为10天"},
            {"chunk_id": 1, "file_name": "test.pdf", "page_num": 2,
             "text": "保费缴纳方式为年缴"},
            {"chunk_id": 2, "file_name": "test.pdf", "page_num": 3,
             "text": "理赔需要提供医院诊断证明"}
        ]
        index.add_documents(chunks)
        index.build()

        # 检索"犹豫期"
        results = index.search("犹豫期")
        assert len(results) > 0
        assert results[0]["chunk_id"] == 0

        # 检索"理赔"
        results = index.search("理赔")
        assert len(results) > 0
        assert results[0]["chunk_id"] == 2

    def test_save_and_load(self, tmp_path):
        """测试保存和加载索引"""
        index = BM25Index()
        chunks = [
            {"chunk_id": 0, "file_name": "test.pdf", "page_num": 1,
             "text": "测试文档内容"}
        ]
        index.add_documents(chunks)
        index.build()

        # 保存
        save_path = tmp_path / "test_index.json"
        index.save(save_path)

        # 加载
        new_index = BM25Index()
        new_index.load(save_path)
        assert len(new_index.documents) == 1


class TestRuleBasedIndex:
    """规则索引测试"""

    def test_search_by_keyword(self):
        """测试关键词检索"""
        index = RuleBasedIndex()
        chunk = {"chunk_id": 0, "file_name": "test.pdf", "page_num": 1,
                 "text": "本保险合同的犹豫期为10天"}
        index.add_document({"file_name": "test.pdf"}, [chunk])

        results = index.search_by_keyword("保险")
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__])
