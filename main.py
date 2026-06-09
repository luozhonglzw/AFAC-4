"""
AFAC2026-4 金融文档智能问答系统 - 主入口

用法:
    python main.py                          # 处理全部 A 榜题目
    python main.py --split A                # 指定 A/B 榜
    python main.py --split B --workers 10   # B 榜，10 并发
    python main.py --budget-limit 4000000   # 自定义 Token 预算
    python main.py --skip-parse             # 跳过文档解析（已有 processed 数据）
    python main.py --skip-index             # 跳过索引构建（已有 index 数据）
    python main.py --domain insurance       # 只处理单个领域
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# 项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import (
    PROJECT_ROOT, DOMAINS, TOTAL_TOKEN_BUDGET, MAX_CONCURRENCY,
    PROCESSED_DIR, INDEX_DIR,
)
from src.document_parser import process_domain, process_all_domains
from src.indexer import build_index, KeywordIndex, RuleIndex, SectionIndex
from src.compressor import ChunkCompressor, EvidenceAggregator, ContextBuilder, MemoryManager
from src.agent import FinancialAgent
from src.submit import generate_submission, load_questions, compute_token_score, compute_final_score

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")


# ==================== 命令行参数 ====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="AFAC2026-4 金融文档智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split", choices=["A", "B", "a", "b"], default="A",
        help="数据集分割：A 榜或 B 榜（默认 A）",
    )
    parser.add_argument(
        "--workers", type=int, default=MAX_CONCURRENCY,
        help=f"并发线程数（默认 {MAX_CONCURRENCY}）",
    )
    parser.add_argument(
        "--budget-limit", type=int, default=TOTAL_TOKEN_BUDGET,
        help=f"Token 预算上限（默认 {TOTAL_TOKEN_BUDGET:,}）",
    )
    parser.add_argument(
        "--skip-parse", action="store_true",
        help="跳过文档解析（使用已有的 processed 数据）",
    )
    parser.add_argument(
        "--skip-index", action="store_true",
        help="跳过索引构建（使用已有的 index 数据）",
    )
    parser.add_argument(
        "--domain",
        choices=list(DOMAINS.keys()),
        help="只处理指定领域（默认全部）",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="输出目录（默认项目根目录）",
    )
    return parser.parse_args()


# ==================== 步骤 1：文档预处理 ====================

def step_parse_documents(domain_filter: str = None):
    """离线文档解析，不计 Token"""
    logger.info("=" * 50)
    logger.info("步骤 1：文档预处理（离线，不计 Token）")
    logger.info("=" * 50)

    if domain_filter:
        domains = [domain_filter]
    else:
        domains = list(DOMAINS.keys())

    total_chunks = 0
    for domain in domains:
        cfg = DOMAINS[domain]
        out_dir = cfg["processed_dir"]

        # 检查是否已有处理结果
        if out_dir.exists() and any(out_dir.glob("*.jsonl")):
            existing = list(out_dir.glob("*.jsonl"))
            logger.info(f"  [{domain}] 已有 {len(existing)} 个 JSONL 文件，跳过")
            total_chunks += _count_chunks_in_dir(out_dir)
            continue

        logger.info(f"  [{domain}] 开始解析...")
        t0 = time.time()
        chunks = process_domain(domain)
        elapsed = time.time() - t0
        logger.info(f"  [{domain}] 完成: {len(chunks)} chunks, 耗时 {elapsed:.1f}s")
        total_chunks += len(chunks)

    logger.info(f"文档预处理完成，共 {total_chunks} chunks")
    return total_chunks


# ==================== 步骤 2：构建索引 ====================

def step_build_indexes(domain_filter: str = None):
    """构建检索索引，不计 Token"""
    logger.info("=" * 50)
    logger.info("步骤 2：构建索引（离线，不计 Token）")
    logger.info("=" * 50)

    if domain_filter:
        domains = [domain_filter]
    else:
        domains = list(DOMAINS.keys())

    # 检查是否已有索引
    index_file = INDEX_DIR / "keyword_index.pkl"
    if index_file.exists():
        logger.info("  已有 keyword_index.pkl，尝试加载...")
        ki = KeywordIndex()
        try:
            ki.load(index_file)
            logger.info(f"  加载成功: {len(ki.chunks)} chunks indexed")
            ri = RuleIndex()
            ri.build(ki.chunks)
            si = SectionIndex()
            si.build(ki.chunks)
            return ki, ri, si
        except Exception as e:
            logger.warning(f"  加载失败 ({e})，重新构建...")

    # 加载所有 chunk
    from src.indexer import _load_all_chunks, _tag_domain
    all_chunks = _load_all_chunks()
    if domain_filter:
        all_chunks = [c for c in all_chunks if c.get("domain") == domain_filter]
    _tag_domain(all_chunks)

    logger.info(f"  加载了 {len(all_chunks)} chunks")

    # 构建索引
    t0 = time.time()
    ki = KeywordIndex()
    ki.add_chunks(all_chunks)
    ki.build()
    ki.save()
    elapsed = time.time() - t0
    logger.info(f"  KeywordIndex 构建完成: {len(ki.chunks)} chunks, {len(ki.inverted)} terms, 耗时 {elapsed:.1f}s")

    ri = RuleIndex()
    ri.build(all_chunks)

    si = SectionIndex()
    si.build(all_chunks)

    logger.info(f"  RuleIndex + SectionIndex 构建完成")
    return ki, ri, si


# ==================== 步骤 3：加载题目 ====================

def step_load_questions(split: str, domain_filter: str = None) -> list[dict]:
    """加载题目文件"""
    logger.info("=" * 50)
    logger.info(f"步骤 3：加载 {split} 榜题目")
    logger.info("=" * 50)

    questions_dir = PROJECT_ROOT / "public_dataset_a" / "public_dataset_upload" / "questions" / f"group_{split.lower()}"

    if not questions_dir.exists():
        logger.error(f"题目目录不存在: {questions_dir}")
        sys.exit(1)

    all_questions = []
    for qfile in sorted(questions_dir.glob("*_questions.json")):
        # 从文件名推断领域
        fname = qfile.stem  # e.g. "insurance_questions"
        domain = fname.replace("_questions", "")

        if domain_filter and domain != domain_filter:
            continue

        questions = load_questions(qfile)
        # 确保每道题有 domain 字段
        for q in questions:
            q.setdefault("domain", domain)
        all_questions.extend(questions)
        logger.info(f"  [{domain}] {len(questions)} 题")

    logger.info(f"共加载 {len(all_questions)} 题")
    return all_questions


# ==================== 步骤 4：批量解题 ====================

def step_solve(
    questions: list[dict],
    ki: KeywordIndex,
    ri: RuleIndex,
    si: SectionIndex,
    max_workers: int,
    budget_limit: int,
) -> list[dict]:
    """批量解题"""
    logger.info("=" * 50)
    logger.info(f"步骤 4：批量解题（{max_workers} 并发，预算 {budget_limit:,}）")
    logger.info("=" * 50)

    agent = FinancialAgent(ki, ri, si, token_budget=budget_limit)

    t0 = time.time()
    results = agent.batch_solve(questions, max_workers=max_workers)
    elapsed = time.time() - t0

    stats = agent.get_stats()
    logger.info(f"解题完成: {len(results)} 题, 耗时 {elapsed:.1f}s")
    logger.info(f"Token 消耗: {stats['token_usage']}")
    logger.info(f"预算状态: {stats['budget_status']}")

    return results


# ==================== 主函数 ====================

def main():
    args = parse_args()
    split = args.split.upper()
    domain_filter = args.domain
    max_workers = args.workers
    budget_limit = args.budget_limit
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT

    logger.info("AFAC2026-4 金融文档智能问答系统 启动")
    logger.info(f"  Split:    {split}")
    logger.info(f"  Domain:   {domain_filter or '全部'}")
    logger.info(f"  Workers:  {max_workers}")
    logger.info(f"  Budget:   {budget_limit:,}")
    logger.info(f"  Output:   {output_dir}")

    t_start = time.time()

    # 步骤 1：文档预处理
    if not args.skip_parse:
        step_parse_documents(domain_filter)
    else:
        logger.info("跳过文档预处理（--skip-parse）")

    # 步骤 2：构建索引
    ki, ri, si = step_build_indexes(domain_filter)

    # 步骤 3：加载题目
    questions = step_load_questions(split, domain_filter)
    if not questions:
        logger.error("没有找到题目，退出")
        sys.exit(1)

    # 步骤 4：批量解题
    results = step_solve(questions, ki, ri, si, max_workers, budget_limit)

    # 步骤 5：生成提交文件
    logger.info("=" * 50)
    logger.info("步骤 5：生成提交文件")
    logger.info("=" * 50)
    csv_path, json_path = generate_submission(results, output_dir)

    # 完成
    t_total = time.time() - t_start
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    logger.info(f"全部完成！总耗时 {t_total:.1f}s, 总 Token {total_tokens:,}")
    logger.info(f"  answer.csv:    {csv_path}")
    logger.info(f"  evidence.json: {json_path}")


if __name__ == "__main__":
    main()


# ==================== 辅助函数 ====================

def _count_chunks_in_dir(d: Path) -> int:
    """统计目录下 JSONL 文件的总行数"""
    count = 0
    for f in d.glob("*.jsonl"):
        count += sum(1 for _ in open(f, encoding="utf-8"))
    return count
