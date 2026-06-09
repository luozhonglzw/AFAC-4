"""
生成 answer.csv + evidence.json，统计 Token 消耗与预估得分
"""
import csv
import json
import logging
from pathlib import Path
from typing import Optional

from .config import CSV_ENCODING, PROJECT_ROOT, TOTAL_TOKEN_BUDGET, OUTPUT_FIELDS

logger = logging.getLogger(__name__)


# ====================================================================
#  生成提交文件
# ====================================================================

def generate_submission(
    results: list[dict],
    output_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    """
    根据 Agent 批量运行结果生成 answer.csv 和 evidence.json。

    Args:
        results: [{qid, answer, prompt_tokens, completion_tokens, total_tokens, evidence}, ...]
        output_dir: 输出目录，默认 PROJECT_ROOT

    Returns:
        (answer.csv 路径, evidence.json 路径)
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "answer.csv"
    json_path = output_dir / "evidence.json"

    _write_csv(results, csv_path)
    _write_evidence(results, json_path)

    _print_stats(results)

    return csv_path, json_path


# ====================================================================
#  CSV 生成
# ====================================================================

def _write_csv(results: list[dict], path: Path):
    """
    生成 answer.csv：
    第一行：summary,,{total_prompt},{total_completion},{total}
    后续每行：qid,answer,prompt_tokens,completion_tokens,total_tokens
    """
    total_prompt = sum(r.get("prompt_tokens", 0) for r in results)
    total_completion = sum(r.get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)

    with open(path, "w", newline="", encoding=CSV_ENCODING) as f:
        writer = csv.writer(f)
        # summary 行
        writer.writerow(["summary", "", total_prompt, total_completion, total_tokens])
        # 数据行
        for r in results:
            writer.writerow([
                r.get("qid", ""),
                r.get("answer", ""),
                r.get("prompt_tokens", 0),
                r.get("completion_tokens", 0),
                r.get("total_tokens", 0),
            ])

    logger.info(f"answer.csv 已生成: {path} ({len(results)} 题)")


# ====================================================================
#  Evidence JSON 生成
# ====================================================================

def _write_evidence(results: list[dict], path: Path):
    """
    生成 evidence.json：按 qid 组织证据链。
    """
    evidence_map = {}
    for r in results:
        qid = r.get("qid", "")
        evidence_map[qid] = r.get("evidence", [])

    with open(path, "w", encoding="utf-8") as f:
        json.dump(evidence_map, f, ensure_ascii=False, indent=2)

    logger.info(f"evidence.json 已生成: {path}")


# ====================================================================
#  统计与报告
# ====================================================================

def _print_stats(results: list[dict]):
    """打印统计信息"""
    total_prompt = sum(r.get("prompt_tokens", 0) for r in results)
    total_completion = sum(r.get("completion_tokens", 0) for r in results)
    total_tokens = sum(r.get("total_tokens", 0) for r in results)
    n = len(results)

    print("\n" + "=" * 60)
    print("提交统计")
    print("=" * 60)
    print(f"  题目总数:          {n}")
    print(f"  Prompt Tokens:     {total_prompt:,}")
    print(f"  Completion Tokens: {total_completion:,}")
    print(f"  Total Tokens:      {total_tokens:,}")
    print(f"  Token 预算:        {TOTAL_TOKEN_BUDGET:,}")
    print(f"  预算使用率:        {total_tokens / TOTAL_TOKEN_BUDGET * 100:.1f}%")

    if total_tokens > TOTAL_TOKEN_BUDGET:
        print(f"  ⚠️  警告：Token 消耗已超过预算上限！")

    # 预估 TokenScore
    # TokenScore = max(0, 1 - (total_tokens - 3_600_000) / 1_400_000) * 100
    # 简化：3.6M 以内满分，5M 为 0 分
    if total_tokens <= 3_600_000:
        token_score = 100.0
    elif total_tokens >= 5_000_000:
        token_score = 0.0
    else:
        token_score = max(0, (5_000_000 - total_tokens) / 1_400_000 * 100)

    print(f"  预估 TokenScore:   {token_score:.1f}/100")
    print("=" * 60)


def compute_token_score(total_tokens: int) -> float:
    """
    计算 TokenScore（0-100）。
    3.6M 以内满分，5M 为 0 分，线性插值。
    """
    if total_tokens <= 3_600_000:
        return 100.0
    if total_tokens >= 5_000_000:
        return 0.0
    return max(0.0, (5_000_000 - total_tokens) / 1_400_000 * 100)


def compute_final_score(accuracy: float, token_score: float) -> float:
    """
    计算 FinalScore = 0.7 × Accuracy + 0.3 × TokenScore
    """
    return 0.7 * accuracy + 0.3 * token_score


# ====================================================================
#  加载题目
# ====================================================================

def load_questions(path: Path) -> list[dict]:
    """
    加载题目文件（JSON 数组 或 JSONL）。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    # 尝试 JSON 数组
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 尝试 JSONL
    questions = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            questions.append(json.loads(line))
    return questions
