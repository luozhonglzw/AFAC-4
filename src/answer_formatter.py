"""
答案格式化、标准化、合法性校验
严格校验：mcq/tf 只允许 A-D，multi 最多 4 个字母，杜绝 ABCDE
"""
import json
import logging
import re

from .config import QUESTION_TYPES

logger = logging.getLogger(__name__)

# 合法字母集合（严格限制 A-D）
_MCQ_VALID = QUESTION_TYPES["mcq"]["valid_answers"]    # {A,B,C,D}
_MULTI_VALID = QUESTION_TYPES["multi"]["valid_answers"] # {A,B,C,D} 严格4个
_TF_VALID = QUESTION_TYPES["tf"]["valid_answers"]       # {A,B}


def normalize_answer(answer: str, format_type: str) -> str:
    """
    标准化答案格式，严格校验。

    mcq:  取第一个有效字母 A/B/C/D
    tf:   取第一个有效字母 A/B
    multi: 提取所有有效字母（A-D），去重，按字母序排序，最多4个

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
        # 最终兜底：从任意位置找第一个有效字母
        for ch in answer:
            if ch in valid:
                logger.warning(f"Fallback: 从非标准答案中提取字母 {ch!r}, 原始={answer[:50]}")
                return ch
        logger.warning(f"无法从答案中提取有效字母: answer={answer!r}, format={format_type}")
        return ""

    elif format_type == "multi":
        # 严格提取 A-D 的字母，去重排序，最多4个
        chars = sorted(set(ch for ch in answer if ch in _MULTI_VALID))
        if not chars:
            logger.warning(f"多选答案中无有效字母: answer={answer!r}")
            return ""
        # 截断为最多4个字母
        if len(chars) > 4:
            logger.warning(f"多选答案超过4个选项，截断: {chars} -> {chars[:4]}")
            chars = chars[:4]
        result = "".join(chars)
        if result != answer.strip():
            logger.info(f"多选答案标准化: {answer!r} -> {result}")
        return result

    else:
        logger.warning(f"未知题型: {format_type}")
        return ""


def validate_answer(answer: str, format_type: str) -> bool:
    """
    校验答案是否合法（严格模式）。

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
        # 每个字符都必须在 A-D 范围内
        for ch in answer:
            if ch not in _MULTI_VALID:
                return False
        # 必须按字母序
        if answer != "".join(sorted(answer)):
            return False
        # 长度不超过4
        if len(answer) > 4:
            return False
        return True

    return False


def extract_json_answer(raw_text: str, format_type: str) -> str:
    """
    从模型原始输出中提取 answer 字段。
    多层 fallback：
    Layer 1: JSON 精确提取
    Layer 2: Markdown 代码块提取
    Layer 3: 纯文本正则提取最后一个字母组合
    Layer 4: 默认答案

    Returns:
        提取到的原始 answer 字符串（未标准化）
    """
    if not raw_text:
        return ""

    # Layer 1: 尝试直接 JSON 解析
    answer = _try_parse_json(raw_text, format_type)
    if answer:
        return answer

    # Layer 2: 尝试从 ```json ... ``` 代码块中提取
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_text, re.DOTALL)
    if code_block:
        answer = _try_parse_json(code_block.group(1), format_type)
        if answer:
            return answer

    # Layer 3: 尝试找 { ... } 块
    json_match = re.search(r"\{[^{}]*\"answer\"[^{}]*\}", raw_text, re.DOTALL)
    if json_match:
        answer = _try_parse_json(json_match.group(), format_type)
        if answer:
            return answer

    # Layer 4: 正则直接提取字母组合（找最后一个出现的，通常是最终答案）
    if format_type in ("mcq", "tf"):
        # 找最后一个独立的 A-D 字母
        matches = re.findall(r"\b([A-D])\b", raw_text)
        if matches:
            result = matches[-1]
            logger.info(f"Layer4 fallback (mcq/tf): 从文本提取字母 {result}")
            return result
    elif format_type == "multi":
        # 找最后一个 A-D 字母组合（如 "AC", "ABD"）
        matches = re.findall(r"\b([A-D]{2,4})\b", raw_text)
        if matches:
            result = matches[-1]
            logger.info(f"Layer4 fallback (multi): 从文本提取字母组合 {result}")
            return result
        # 也尝试找单个字母的组合
        single_matches = re.findall(r"\b([A-D])\b", raw_text)
        if single_matches:
            result = "".join(sorted(set(single_matches)))[:4]
            logger.info(f"Layer4 fallback (multi): 从散落字母组合 {result}")
            return result

    # 全部失败，返回原始文本（normalize_answer 会进一步处理）
    logger.warning(f"所有 fallback 层失败，返回原始文本: {raw_text[:100]}")
    return raw_text


def _try_parse_json(text: str, format_type: str) -> str:
    """尝试 JSON 解析提取 answer"""
    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return ""

    ans = data.get("answer", "")
    if ans is None:
        return ""

    if isinstance(ans, list):
        # multi 题型：列表 -> 排序拼接，严格限制 A-D
        valid_chars = sorted(set(str(a).upper().strip() for a in ans if str(a).upper().strip() in _MULTI_VALID))
        if len(valid_chars) > 4:
            valid_chars = valid_chars[:4]
        return "".join(valid_chars)

    return str(ans).strip()
