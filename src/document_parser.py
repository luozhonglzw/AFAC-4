"""
PDF/文本解析、表格提取、结构化分块
离线预处理阶段，不调用大模型 API，不计入 Token 消耗
"""
import json
import re
import hashlib
from pathlib import Path
from typing import Optional

import pdfplumber
import fitz  # PyMuPDF

from .config import (
    RAW_DIR, PROCESSED_DIR, INDEX_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, DOMAINS,
)

# ==================== 正则：条款/章节边界 ====================
# 第四十七条、第一百零二条、第3条
RE_CLAUSE = re.compile(r"第[一二三四五六七八九十百零\d]+条")
# 第三章、第十二章
RE_CHAPTER = re.compile(r"第[一二三四五六七八九十百零\d]+章")
# 第五节、第一节
RE_SECTION = re.compile(r"第[一二三四五六七八九十百零\d]+节")
# 一、二、三、……（常见总则/分则大标题）
RE_HAN_NUM_TITLE = re.compile(r"^[一二三四五六七八九十]+[、．.]")

# ==================== 正则：金融实体 ====================
# 金额：1,234.56元 / 100万元 / 5亿美元
RE_AMOUNT = re.compile(
    r"\d[\d,]*\.?\d*\s*(?:万|亿|百万|千万)?\s*(?:元|美元|欧元|港币|人民币)"
)
# 比例：3.5% / 百分之三
RE_RATIO = re.compile(r"\d+\.?\d*\s*%|百分之[一二三四五六七八九十\d]+")
# 期限：30天 / 6个月 / 2年
RE_TERM = re.compile(r"\d+\s*(?:天|日|个月|月|年)")
# 日期：2024年1月1日 / 2024-01-01 / 2024/01/01
RE_DATE = re.compile(
    r"\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|\d{4}[-/]\d{1,2}[-/]\d{1,2}"
)

# ==================== 表格转 Markdown ====================

def _table_to_markdown(table: list[list]) -> str:
    """将 pdfplumber 提取的二维列表转为 Markdown 表格"""
    if not table or not table[0]:
        return ""

    # 清洗 None
    cleaned = [[(cell if cell is not None else "") for cell in row] for row in table]
    n_cols = max(len(row) for row in cleaned)

    # 补齐列数
    for row in cleaned:
        while len(row) < n_cols:
            row.append("")

    lines = []
    # 表头
    lines.append("| " + " | ".join(str(c) for c in cleaned[0]) + " |")
    lines.append("| " + " | ".join("---" for _ in range(n_cols)) + " |")
    # 数据行
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")

    return "\n".join(lines)


# ==================== 核心解析 ====================

def parse_pdf(
    pdf_path: Path,
    domain: Optional[str] = None,
) -> list[dict]:
    """
    解析单个 PDF，提取文本 + 表格，按页输出 chunk 列表。

    Args:
        pdf_path: PDF 文件路径
        domain: 领域标识（对 regulatory 领域保留法条完整段落）

    Returns:
        List[Dict]，每个元素:
        {
            doc_id: str,        # 文件 stem
            page: int,          # 页码（1-based）
            section: str,       # 所属章节标题（如 "第三章"）
            clause_number: str, # 条款编号（如 "第四十七条"）
            text: str,          # 文本内容
            is_table: bool,     # 是否为表格
        }
    """
    doc_id = pdf_path.stem
    chunks: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_num = page_idx + 1

            # --- 文本 ---
            raw_text = page.extract_text() or ""
            if raw_text.strip():
                structured = structure_text(raw_text, domain=domain)
                for item in structured:
                    chunks.append({
                        "doc_id": doc_id,
                        "page": page_num,
                        "section": item.get("section", ""),
                        "clause_number": item.get("clause_number", ""),
                        "text": item["text"],
                        "is_table": False,
                    })

            # --- 表格 ---
            tables = page.extract_tables()
            for tbl in tables:
                md = _table_to_markdown(tbl)
                if md.strip():
                    chunks.append({
                        "doc_id": doc_id,
                        "page": page_num,
                        "section": "",
                        "clause_number": "",
                        "text": md,
                        "is_table": True,
                    })

    # pdfplumber 全部失败时回退 PyMuPDF
    if not chunks:
        chunks = _parse_pdf_fitz(pdf_path, domain=domain)

    return chunks


def _parse_pdf_fitz(
    pdf_path: Path,
    domain: Optional[str] = None,
) -> list[dict]:
    """PyMuPDF 回退方案（pdfplumber 无法提取文本时使用）"""
    doc_id = pdf_path.stem
    chunks: list[dict] = []

    doc = fitz.open(str(pdf_path))
    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        raw_text = doc[page_idx].get_text("text")
        if not raw_text.strip():
            continue
        structured = structure_text(raw_text, domain=domain)
        for item in structured:
            chunks.append({
                "doc_id": doc_id,
                "page": page_num,
                "section": item.get("section", ""),
                "clause_number": item.get("clause_number", ""),
                "text": item["text"],
                "is_table": False,
            })
    doc.close()
    return chunks


# ==================== 文本结构化 ====================

def structure_text(
    raw_text: str,
    domain: Optional[str] = None,
) -> list[dict]:
    """
    识别条款边界、提取关键金融实体。

    Args:
        raw_text: 一页原始文本
        domain: 领域标识；regulatory 领域会保留法条完整段落

    Returns:
        List[Dict]，每个元素:
        {
            section: str,         # 章节标题
            clause_number: str,   # 条款编号
            text: str,            # 段落文本
            entities: {           # 提取的金融实体
                amounts: [...],
                ratios: [...],
                terms: [...],
                dates: [...],
            }
        }
    """
    lines = raw_text.split("\n")
    results: list[dict] = []

    current_section = ""
    current_clause = ""
    buf: list[str] = []

    def _flush():
        """将缓冲区中的行合并为一个 chunk"""
        text = "\n".join(buf).strip()
        if not text:
            return
        entities = _extract_entities(text)
        results.append({
            "section": current_section,
            "clause_number": current_clause,
            "text": text,
            "entities": entities,
        })

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # ---- 章节标题 ----
        if RE_CHAPTER.match(stripped) or RE_SECTION.match(stripped):
            _flush()
            buf = []
            current_section = stripped
            current_clause = ""
            continue

        # ---- 条款边界 ----
        clause_match = RE_CLAUSE.match(stripped)
        if clause_match:
            current_clause = clause_match.group()
            if domain == "regulatory":
                # regulatory：不切分，同章节内多条法条聚合成一个完整段落
                pass
            else:
                _flush()
                buf = []
            buf.append(stripped)
            continue

        # ---- 中文大写序号标题（一、二、三、……） ----
        if RE_HAN_NUM_TITLE.match(stripped) and len(stripped) < 60:
            _flush()
            buf = []
            current_section = stripped
            continue

        # ---- 普通行 ----
        buf.append(stripped)

    _flush()
    return results


def _extract_entities(text: str) -> dict:
    """提取金融实体"""
    return {
        "amounts": RE_AMOUNT.findall(text),
        "ratios": RE_RATIO.findall(text),
        "terms": RE_TERM.findall(text),
        "dates": RE_DATE.findall(text),
    }


# ==================== 保存 ====================

def save_processed(data: list[dict], output_path: Path) -> Path:
    """
    按 doc_id 保存为 JSONL，每行一个 chunk。

    Args:
        data: parse_pdf 返回的 chunk 列表
        output_path: 输出文件路径（.jsonl）

    Returns:
        输出文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in data:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return output_path


# ==================== 批量处理入口 ====================

def parse_txt(txt_path: Path, domain: Optional[str] = None) -> list[dict]:
    """
    解析纯文本文件，返回与 parse_pdf 相同格式的 chunk 列表。
    """
    doc_id = txt_path.stem
    raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
    chunks: list[dict] = []

    structured = structure_text(raw_text, domain=domain)
    for item in structured:
        chunks.append({
            "doc_id": doc_id,
            "page": 1,
            "section": item.get("section", ""),
            "clause_number": item.get("clause_number", ""),
            "text": item["text"],
            "is_table": False,
        })
    return chunks


def process_domain(domain: str) -> list[dict]:
    """
    处理指定领域下所有文档（PDF + TXT），输出 JSONL 到 processed 目录。

    Args:
        domain: 领域 key（如 "insurance"）

    Returns:
        该领域全部 chunk
    """
    cfg = DOMAINS[domain]
    raw_dir: Path = cfg["raw_dir"]
    out_dir: Path = cfg["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []

    if not raw_dir.exists():
        return all_chunks

    # PDF（含子目录）
    for pdf_path in sorted(raw_dir.rglob("*.pdf")):
        chunks = parse_pdf(pdf_path, domain=domain)
        for c in chunks:
            c["domain"] = domain
        all_chunks.extend(chunks)
        out_file = out_dir / f"{pdf_path.stem}.jsonl"
        save_processed(chunks, out_file)

    # TXT（含子目录）
    for txt_path in sorted(raw_dir.rglob("*.txt")):
        chunks = parse_txt(txt_path, domain=domain)
        for c in chunks:
            c["domain"] = domain
        all_chunks.extend(chunks)
        out_file = out_dir / f"{txt_path.stem}.jsonl"
        save_processed(chunks, out_file)

    return all_chunks


def process_all_domains() -> dict[str, list[dict]]:
    """
    处理全部 5 个领域。

    Returns:
        {domain_key: [chunk, ...], ...}
    """
    result = {}
    for domain in DOMAINS:
        result[domain] = process_domain(domain)
    return result


# ==================== 工具函数 ====================

def _file_hash(file_path: Path) -> str:
    """计算文件 MD5"""
    return hashlib.md5(file_path.read_bytes()).hexdigest()


def load_processed_jsonl(jsonl_path: Path) -> list[dict]:
    """加载单个 JSONL 文件"""
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks
