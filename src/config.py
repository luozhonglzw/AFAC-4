"""
全局配置：API Key、模型参数、路径设置、题型约束、Token 预算
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 自动加载项目根目录的 .env 文件
load_dotenv(Path(__file__).parent.parent / ".env")

# ==================== 路径配置 ====================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# ==================== API Key 配置 ====================
# 优先百炼，其次魔搭
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY", "")
API_KEY = DASHSCOPE_API_KEY or MODELSCOPE_API_KEY

if not API_KEY:
    import warnings
    warnings.warn(
        "未检测到 API Key，请设置环境变量 DASHSCOPE_API_KEY 或 MODELSCOPE_API_KEY",
        stacklevel=2,
    )

# ==================== 模型配置 ====================
QWEN_MODEL = "qwen3.6-plus"

# 百炼（优先）/ 魔搭 双 endpoint
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODELSCOPE_BASE_URL = "https://api-inference.modelscope.cn/v1"

# 按优先级选择 endpoint
if DASHSCOPE_API_KEY:
    QWEN_BASE_URL = DASHSCOPE_BASE_URL
elif MODELSCOPE_API_KEY:
    QWEN_BASE_URL = MODELSCOPE_BASE_URL
else:
    QWEN_BASE_URL = DASHSCOPE_BASE_URL  # 默认百炼

# 模型生成参数
MAX_TOKENS = 4096
TEMPERATURE = 0.1   # 低温度保证答案稳定性
TOP_P = 0.9

# ==================== Token 预算 ====================
TOTAL_TOKEN_BUDGET = 5_000_000   # 总 Token 预算
prompt_tokens_used = 0           # 已用 prompt tokens（运行时累加）
completion_tokens_used = 0       # 已用 completion tokens（运行时累加）

# ==================== 并发与重试 ====================
MAX_CONCURRENCY = 3              # 最大并发请求数（降低并发减少限流）
MAX_RETRIES = 5                  # 单次请求最大重试次数
RETRY_BASE_DELAY = 5.0           # 重试基础间隔（秒），指数退避
REQUEST_TIMEOUT = 180            # 单次请求超时（秒），大context需要3分钟
RETRY_BACKOFF = [5, 10, 20, 40, 80]  # 指数退避序列（秒）

# ==================== 检索配置 ====================
BM25_K1 = 1.5
BM25_B = 0.75
TOP_K_RETRIEVAL = 10
MAX_EVIDENCE_CHARS = 8000

# ==================== 文档解析配置 ====================
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TABLE_EXTRACT_ENABLED = True

# ==================== 领域映射 ====================
# 每个领域的文档路径、索引路径、专用 prompt
# raw_dir 指向 public_dataset_a 实际数据，processed/index 指向 data/ 输出
_DATASET_RAW = PROJECT_ROOT / "public_dataset_a" / "public_dataset_upload" / "raw"

DOMAINS = {
    "insurance": {
        "label": "保险",
        "raw_dir": _DATASET_RAW / "insurance",
        "processed_dir": PROCESSED_DIR / "insurance",
        "index_dir": INDEX_DIR / "insurance",
        "prompt_file": PROMPTS_DIR / "insurance.txt",
    },
    "regulatory": {
        "label": "监管法规",
        "raw_dir": _DATASET_RAW / "regulatory",
        "processed_dir": PROCESSED_DIR / "regulatory",
        "index_dir": INDEX_DIR / "regulatory",
        "prompt_file": PROMPTS_DIR / "regulatory.txt",
    },
    "financial_contracts": {
        "label": "金融合同",
        "raw_dir": _DATASET_RAW / "financial_contracts",
        "processed_dir": PROCESSED_DIR / "financial_contracts",
        "index_dir": INDEX_DIR / "financial_contracts",
        "prompt_file": PROMPTS_DIR / "financial_contracts.txt",
    },
    "financial_reports": {
        "label": "财务报告",
        "raw_dir": _DATASET_RAW / "financial_reports",
        "processed_dir": PROCESSED_DIR / "financial_reports",
        "index_dir": INDEX_DIR / "financial_reports",
        "prompt_file": PROMPTS_DIR / "financial_reports.txt",
    },
    "research": {
        "label": "研究报告",
        "raw_dir": _DATASET_RAW / "research",
        "processed_dir": PROCESSED_DIR / "research",
        "index_dir": INDEX_DIR / "research",
        "prompt_file": PROMPTS_DIR / "research.txt",
    },
}

# ==================== 题型映射 ====================
QUESTION_TYPES = {
    "mcq": {
        "label": "单选题",
        "valid_answers": {"A", "B", "C", "D"},
        "format_rule": "仅输出一个大写字母，如 A",
    },
    "multi": {
        "label": "多选题",
        "valid_answers": {"A", "B", "C", "D"},  # 严格限制 A-D，不允许 E
        "format_rule": "按字母升序拼接，无分隔符，如 ABC（最多4个字母）",
    },
    "tf": {
        "label": "判断题",
        "valid_answers": {"A", "B"},          # A=正确, B=错误
        "format_rule": "仅输出 A 或 B",
    },
}

# ==================== 答案约束 ====================
class AnswerConstraint:
    """答案格式校验规则"""

    @staticmethod
    def validate(answer: str, qtype: str) -> tuple[bool, str]:
        """
        校验答案格式是否合法

        Args:
            answer: 待校验答案
            qtype: 题型 (mcq / multi / tf)

        Returns:
            (是否合法, 错误信息)
        """
        answer = answer.strip().upper()
        spec = QUESTION_TYPES.get(qtype)
        if spec is None:
            return False, f"未知题型: {qtype}"

        valid = spec["valid_answers"]

        if qtype in ("mcq", "tf"):
            if len(answer) != 1 or answer not in valid:
                return False, f"{spec['label']}答案必须是 {'/'.join(sorted(valid))} 中的一个字母"
        elif qtype == "multi":
            if not answer:
                return False, "多选答案不能为空"
            for ch in answer:
                if ch not in valid:
                    return False, f"非法选项 {ch}，合法范围: {'/'.join(sorted(valid))}"
            if answer != "".join(sorted(answer)):
                return False, "多选答案必须按字母升序排列"

        return True, ""

    @staticmethod
    def normalize(answer: str, qtype: str) -> str:
        """标准化答案格式"""
        answer = answer.strip().upper()
        if qtype == "multi":
            answer = "".join(sorted(set(answer)))
        return answer


# ==================== 输出字段 ====================
OUTPUT_FIELDS = [
    "qid",              # 题目 ID
    "answer",           # 答案
    "prompt_tokens",    # 本轮 prompt 消耗
    "completion_tokens",# 本轮 completion 消耗
    "total_tokens",     # 本轮总消耗
]

# ==================== 答案格式配置 ====================
ANSWER_MAX_LENGTH = 500
CSV_ENCODING = "utf-8-sig"
