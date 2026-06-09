"""
B 榜盲测召回率测试
用 A 榜数据模拟 B 榜，测试盲测检索能否找到正确文档
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.indexer import KeywordIndex, RuleIndex, SectionIndex, search
from src.config import DOMAINS


def main():
    # 加载索引
    print("加载索引...")
    ki = KeywordIndex()
    ki.load(Path("data/index/keyword_index.pkl"))
    ri = RuleIndex()
    ri.build(ki.chunks)
    si = SectionIndex()
    si.build(ki.chunks)
    print(f"索引加载完成: {len(ki.chunks)} chunks")

    # 加载 A 榜题目（含 doc_ids）
    qdir = Path("public_dataset_a/public_dataset_upload/questions/group_a")
    all_questions = []
    for qf in sorted(qdir.glob("*_questions.json")):
        domain = qf.stem.replace("_questions", "")
        with open(qf, "r", encoding="utf-8") as f:
            questions = json.load(f)
        for q in questions:
            q["domain"] = domain
            all_questions.append(q)

    # 只测试有 doc_ids 的题目
    test_questions = [q for q in all_questions if q.get("doc_ids")]
    print(f"测试题目: {len(test_questions)} 题（有 doc_ids）")

    # doc_id 标准化
    def normalize_doc_id(did: str) -> str:
        if did.startswith("text"):
            return did[4:].lstrip("0") or "0"
        return did

    # 测试盲测召回率
    correct_top1 = 0
    correct_top3 = 0
    correct_top5 = 0
    total = 0
    domain_stats = {}

    for q in test_questions:
        true_doc_ids = set(normalize_doc_id(d) for d in q.get("doc_ids", []))
        if not true_doc_ids:
            continue

        domain = q["domain"]
        question = q["question"]

        # 跨域粗排
        all_candidates = []
        for d in DOMAINS:
            try:
                hits = search(question, d, ki, ri, si, top_k=5)
                for h in hits:
                    h["_search_domain"] = d
                all_candidates.extend(hits)
            except Exception:
                continue

        # 全局 BM25 补充
        global_hits = ki.search(question, top_k=15)
        for h in global_hits:
            h["_search_domain"] = h.get("domain", "")
            all_candidates.append(h)

        # 实体匹配：题目中提到的文档名 -> doc_id
        import re
        doc_name_patterns = re.findall(r'(?:fc_text|fin_text|ins_text|reg_text|res_text)_(\d+)', question)
        for num in doc_name_patterns:
            doc_id = f"text{int(num):02d}"
            # 直接添加该 doc_id 的所有 chunks
            matched = [c for c in ki.chunks if c.get("doc_id") == doc_id]
            for h in matched[:10]:
                h["_search_domain"] = h.get("domain", "")
                h["score"] = 10.0  # 高分 boost
                all_candidates.append(h)

        # 按 doc_id 聚合
        doc_scores = {}
        for cand in all_candidates:
            doc_id = cand.get("doc_id", "")
            score = cand.get("score", 0)
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {"score": 0, "domain": cand.get("_search_domain", ""), "count": 0}
            doc_scores[doc_id]["score"] += score
            doc_scores[doc_id]["count"] += 1

        # 排序取 Top 10 doc_id
        ranked = sorted(
            doc_scores.items(),
            key=lambda x: (x[1]["score"] * 0.6 + x[1]["count"] * 0.4),
            reverse=True,
        )
        top8_ids = [normalize_doc_id(doc_id) for doc_id, _ in ranked[:8]]
        top5_ids = top8_ids[:5]
        top3_ids = top5_ids[:3]
        top1_ids = top5_ids[:1]

        # 计算召回
        hit_top1 = any(did in true_doc_ids for did in top1_ids)
        hit_top3 = any(did in true_doc_ids for did in top3_ids)
        hit_top5 = any(did in true_doc_ids for did in top5_ids)

        if hit_top1:
            correct_top1 += 1
        if hit_top3:
            correct_top3 += 1
        if hit_top5:
            correct_top5 += 1
        total += 1

        # 按领域统计
        if domain not in domain_stats:
            domain_stats[domain] = {"total": 0, "top3_hit": 0}
        domain_stats[domain]["total"] += 1
        if hit_top3:
            domain_stats[domain]["top3_hit"] += 1

        status = "OK" if hit_top3 else "MISS"
        print(f"  [{status}] {q['qid']} ({domain}): Top3={top3_ids} True={list(true_doc_ids)[:3]}")

    # 汇总
    print(f"\n{'='*60}")
    print(f"盲测召回率 (样本={total}):")
    print(f"  Top1 召回: {correct_top1}/{total} = {correct_top1/total*100:.1f}%")
    print(f"  Top3 召回: {correct_top3}/{total} = {correct_top3/total*100:.1f}%")
    print(f"  Top5 召回: {correct_top5}/{total} = {correct_top5/total*100:.1f}%")

    print(f"\n按领域统计:")
    for domain, stats in sorted(domain_stats.items()):
        rate = stats["top3_hit"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {domain}: {stats['top3_hit']}/{stats['total']} = {rate:.1f}%")

    top3_rate = correct_top3 / total if total > 0 else 0
    top5_rate = correct_top5 / total if total > 0 else 0

    print(f"\n{'='*60}")
    if top3_rate >= 0.75:
        print(f"[OK] Top3 召回率 {top3_rate:.1%} >= 75%，B 榜代码可用")
    else:
        print(f"[WARN] Top3 召回率 {top3_rate:.1%} < 75%，需要优化检索策略")
    if top5_rate >= 0.85:
        print(f"[OK] Top5 召回率 {top5_rate:.1%} >= 85%")
    else:
        print(f"[WARN] Top5 召回率 {top5_rate:.1%} < 85%")

    return top3_rate


if __name__ == "__main__":
    rate = main()
    sys.exit(0 if rate >= 0.75 else 1)
