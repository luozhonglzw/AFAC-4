"""
答案格式化、标准化、合法性校验
"""
import logging
import re

from .config import QUESTION_TYPES

logger = logging.getLogger(__name__)

# 合法字母集合
_MCQ_VALID = QUESTION_TYPES["mcq"]["valid_answers"]    # {A,B,C,D}
_MULTI_VALID = QUESTION_TYPES["multi"]["valid_answers"] # {A,B,C,D,E}
_TF_VALID = QUESTION_TYPES["tf"]["valid_answers"]       # {A,B}


def normalize_answer(answer: str, format_type: str) -> str:
    """
    标准化答案格式。

    mcq:  取第一个有效字母 A/B/C/D
    tf:   取第一个有效字母 A/B
    multi: 提取所有有效字母，去重，按字母序排序拼接（如 "BD"）

    Args:
        answer: 原始答案字符串（可能含 JSON、推理文本等）
        format_type: "mcq" / "multi" / "tf"

    Returns:
        标准化后的答案字符串；非法时返回 ""
    """
    if not answer:
        return ""

    answer = answer.strip().upper()

    if format_type in ("mcq", "tf"):
        valid = _MCQ_VALID if format_type == "mcq" else _TF_VALID
        # 优先：正则找独立字母（避免 JSON 字符串中 "answer" 的 a 干扰）
        matches = re.findall(r"\b([A-D])\b", answer)
        for m in matches:
            if m in valid:
                return m
        # 兜底：单字符答案
        if len(answer) == 1 and answer in valid:
            return answer
        logger.warning(f"无法从答案中提取有效字母: answer={answer!r}, format={format_type}")
        return ""

    elif format_type == "multi":
        # 提取所有有效字母，去重排序
        chars = sorted(set(ch for ch in answer if ch in _MULTI_VALID))
        if not chars:
            logger.warning(f"多选答案中无有效字母: answer={answer!r}")
            return ""
        return "".join(chars)

    else:
        logger.warning(f"未知题型: {format_type}")
        return ""


def validate_answer(answer: str, format_type: str) -> bool:
    """
    校验答案是否合法。

    Args:
        answer: 已标准化的答案
        format_type: "mcq" / "multi" / "tf"

    Returns:
        True 合法 / False 非法
    """
    if not answer:
        return False

    answer = answer.strip().upper()

    if format_type == "mcq":
        return len(answer) == 1 and answer in _MCQ_VALID

    elif format_type == "tf":
        return len(answer) == 1 and answer in _TF_VALID

    elif format_type == "multi":
        if not answer:
            return False
        for ch in answer:
            if ch not in _MULTI_VALID:
                return False
        # 必须按字母序
        return answer == "".join(sorted(answer))

    return False


def extract_json_answer(raw_text: str, format_type: str) -> str:
    """
    从模型原始输出中提取 answer 字段。
    支持：纯 JSON、JSON 包裹在 ``` 代码块中、含推理文本的混合输出。

    Returns:
        提取到的原始 answer 字符串（未标准化）
    """
    if not raw_text:
        return ""

    # 1. 尝试直接 JSON 解析
    answer = _try_parse_json(raw_text, format_type)
    if answer:
        return answer

    # 2. 尝试从 ```json ... ``` 代码块中提取
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_text, re.DOTALL)
    if code_block:
        answer = _try_parse_json(code_block.group(1), format_type)
        if answer:
            return answer

    # 3. 尝试找 { ... } 块
    json_match = re.search(r"\{[^{}]*\"answer\"[^{}]*\}", raw_text, re.DOTALL)
    if json_match:
        answer = _try_parse_json(json_match.group(), format_type)
        if answer:
            return answer

    # 4. 兜底：正则直接提取字母
    return raw_text


def _try_parse_json(text: str, format_type: str) -> str:
    """尝试 JSON 解析提取 answer"""
    import json
    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return ""

    ans = data.get("answer", "")
    if ans is None:
        return ""

    if isinstance(ans, list):
        # multi 题型：列表 -> 排序拼接
        return "".join(sorted(set(str(a).upper().strip() for a in ans)))

    return str(ans).strip()
