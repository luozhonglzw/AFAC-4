"""
验证 4 项紧急修复的测试脚本
"""
import sys
sys.path.insert(0, ".")

from src.config import (
    MAX_CONCURRENCY, MAX_RETRIES, REQUEST_TIMEOUT, RETRY_BACKOFF,
    QUESTION_TYPES,
)
from src.answer_formatter import normalize_answer, validate_answer, extract_json_answer
from src.compressor import ChunkCompressor

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


# ==================== Fix 1: 超时与降级策略 ====================
print("\n=== Fix 1: 超时与降级策略 ===")

check("REQUEST_TIMEOUT = 180", REQUEST_TIMEOUT == 180, f"got {REQUEST_TIMEOUT}")
check("MAX_RETRIES = 5", MAX_RETRIES == 5, f"got {MAX_RETRIES}")
check("MAX_CONCURRENCY = 3", MAX_CONCURRENCY == 3, f"got {MAX_CONCURRENCY}")
check("RETRY_BACKOFF 长度 = 5", len(RETRY_BACKOFF) == 5, f"got {RETRY_BACKOFF}")
check("RETRY_BACKOFF[0] = 5", RETRY_BACKOFF[0] == 5, f"got {RETRY_BACKOFF[0]}")
check("RETRY_BACKOFF[4] = 80", RETRY_BACKOFF[4] == 80, f"got {RETRY_BACKOFF[4]}")


# ==================== Fix 2: 答案越界与格式校验 ====================
print("\n=== Fix 2: 答案越界与格式校验 ===")

# multi 题型 valid_answers 只有 A-D
check("multi valid_answers = {A,B,C,D}", QUESTION_TYPES["multi"]["valid_answers"] == {"A","B","C","D"})

# normalize_answer: "ABCDE" -> "ABCD"（截断 E）
result = normalize_answer("ABCDE", "multi")
check("normalize 'ABCDE' multi -> 'ABCD'", result == "ABCD", f"got {result!r}")

# normalize_answer: "E" -> ""（E 不合法）
result = normalize_answer("E", "multi")
check("normalize 'E' multi -> ''", result == "", f"got {result!r}")

# normalize_answer: "AC" -> "AC"
result = normalize_answer("AC", "multi")
check("normalize 'AC' multi -> 'AC'", result == "AC", f"got {result!r}")

# normalize_answer: "CAB" -> "ABC"（排序）
result = normalize_answer("CAB", "multi")
check("normalize 'CAB' multi -> 'ABC'", result == "ABC", f"got {result!r}")

# validate_answer: "ABCDE" multi -> False
check("validate 'ABCDE' multi -> False", not validate_answer("ABCDE", "multi"))

# validate_answer: "ABCD" multi -> True
check("validate 'ABCD' multi -> True", validate_answer("ABCD", "multi"))

# validate_answer: "A" mcq -> True
check("validate 'A' mcq -> True", validate_answer("A", "mcq"))

# validate_answer: "AB" mcq -> False
check("validate 'AB' mcq -> False", not validate_answer("AB", "mcq"))

# validate_answer: "A" tf -> True
check("validate 'A' tf -> True", validate_answer("A", "tf"))

# validate_answer: "C" tf -> False
check("validate 'C' tf -> False", not validate_answer("C", "tf"))

# extract_json_answer: 从 JSON 中提取
raw = '{"reasoning": "test", "answer": "AC"}'
result = extract_json_answer(raw, "multi")
check("extract_json_answer JSON multi -> 'AC'", result == "AC", f"got {result!r}")

# extract_json_answer: 从代码块中提取
raw = '```json\n{"reasoning": "test", "answer": "B"}\n```'
result = extract_json_answer(raw, "mcq")
check("extract_json_answer code block -> 'B'", result == "B", f"got {result!r}")

# extract_json_answer: 正则 fallback
raw = '分析过程... 最终答案是 AC'
result = extract_json_answer(raw, "multi")
check("extract_json_answer regex fallback -> 'AC'", result == "AC", f"got {result!r}")


# ==================== Fix 3: Research 领域压缩 ====================
print("\n=== Fix 3: Research 领域压缩 ===")

# 测试研报噪声过滤
compressor = ChunkCompressor()
compressor.set_domain("research")

research_text = """摘要：本报告分析了新能源汽车行业2024年发展趋势。
市场规模达到5000亿元，同比增长35%。
数据来源：Wind数据库，截至2024年12月31日。
免责声明：本报告仅供参考，不构成投资建议。
风险提示：行业政策变化可能导致预期偏差。
公司核心竞争力在于技术创新和成本控制。"""

result = compressor._compress_research(research_text, "新能源汽车市场规模")
check("研报压缩: 删除数据来源", "数据来源" not in result, f"got: {result}")
check("研报压缩: 删除免责声明", "免责声明" not in result, f"got: {result}")
check("研报压缩: 删除风险提示", "风险提示" not in result, f"got: {result}")
check("研报压缩: 保留摘要", "摘要" in result, f"got: {result}")
check("研报压缩: 保留市场规模数据", "5000亿" in result, f"got: {result}")

# 测试财报噪声过滤
compressor.set_domain("financial_reports")

report_text = """公司治理：本公司设有董事会、监事会等治理机构。
营业收入为100亿元，同比增长15%。
股东情况：前十大股东持股比例合计65%。
净利润为20亿元，毛利率35%。"""

result = compressor._compress_financial_report(report_text, "营业收入")
check("财报压缩: 删除公司治理", "公司治理" not in result, f"got: {result}")
check("财报压缩: 删除股东情况", "股东情况" not in result, f"got: {result}")
check("财报压缩: 保留营业收入", "100亿" in result, f"got: {result}")
check("财报压缩: 保留净利润", "20亿" in result, f"got: {result}")


# ==================== Fix 4: 额度管理 ====================
print("\n=== Fix 4: 额度管理 ===")

from src.compressor import MemoryManager

mm = MemoryManager(token_budget=5_000_000)
check("初始预算状态 = normal", mm.budget_status() == "normal")
check("初始剩余预算 = 5,000,000", mm.remaining_budget() == 5_000_000)
check("初始压缩级别 = normal", mm.compression_level() == "normal")

# 模拟消耗到 80%
mm.record_usage(4_000_000, 0)
check("80% 消耗后状态 = warn", mm.budget_status() == "warn")
check("80% 消耗后压缩级别 = moderate", mm.compression_level() == "moderate")

# 模拟消耗到 95%
mm.record_usage(750_000, 0)
check("95% 消耗后状态 = critical", mm.budget_status() == "critical")
check("95% 消耗后压缩级别 = aggressive", mm.compression_level() == "aggressive")

# 模拟消耗完
mm.record_usage(500_000, 0)
check("100% 消耗后状态 = exhausted", mm.budget_status() == "exhausted")


# ==================== 汇总 ====================
print(f"\n{'='*50}")
print(f"测试结果: {PASS} 通过, {FAIL} 失败, 共 {PASS+FAIL} 项")
print(f"{'='*50}")

if FAIL > 0:
    sys.exit(1)
else:
    print("所有测试通过！")
    sys.exit(0)
