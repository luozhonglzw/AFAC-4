# AFAC2026-4 金融文档智能问答系统

> AFAC2026 竞赛第四题 —— 基于 RAG 的金融长文本多领域问答

## 项目简介

本系统面向金融领域的文档理解与问答任务，覆盖 **保险、监管法规、金融合同、财务报告、研究报告** 五大领域，共 100 道 A 榜题目（单选/多选/判断）。

核心设计原则：**严禁使用 Embedding 模型**，仅依赖 BM25 关键词检索 + 规则匹配 + jieba 中文分词。

## 系统架构

```
PDF/TXT 文档
    │
    ▼
┌─────────────────────────────────────────────────┐
│  文档解析层（离线，不计 Token）                    │
│  pdfplumber + PyMuPDF → 结构化 JSONL             │
│  保留：章节、条款号、表格（Markdown）、页码        │
└───────────────────────┬─────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────┐
│  索引层（离线，不计 Token）                        │
│  KeywordIndex: jieba 分词 + 倒排索引 + BM25      │
│  RuleIndex:    领域关键词精确匹配                  │
│  SectionIndex: 章节/条款号 → chunk 定位            │
└───────────────────────┬─────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────┐
│  Agent 层（在线，计 Token）                        │
│  检索 → 压缩 → Prompt 构造 → Qwen3.6-plus 推理   │
│  → JSON 答案提取 → 格式校验 → Token 统计          │
└───────────────────────┬─────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────┐
│  提交层                                           │
│  answer.csv  +  evidence.json                     │
│  TokenScore = f(total_tokens)                     │
│  FinalScore = 0.7 × Accuracy + 0.3 × TokenScore  │
└─────────────────────────────────────────────────┘
```

## 项目结构

```
AFAC-4/
├── main.py                     # 主入口（命令行参数）
├── data/
│   ├── raw/                    # 原始 PDF/文本
│   ├── processed/              # 解析后的 JSONL
│   └── index/                  # BM25 索引缓存
├── src/
│   ├── config.py               # 全局配置、API Key、Token 预算
│   ├── document_parser.py      # PDF/文本解析、条款识别、表格提取
│   ├── indexer.py              # BM25 + RuleIndex + SectionIndex
│   ├── compressor.py           # 规则压缩、证据聚合、上下文构建
│   ├── agent.py                # FinancialAgent 核心推理
│   ├── answer_formatter.py     # 答案标准化、合法性校验
│   └── submit.py               # CSV/Evidence 生成、TokenScore 计算
├── prompts/                    # 5 个领域专用 System Prompt
├── public_dataset_a/           # 官方数据集（PDF + 题目）
├── requirements.txt
└── README.md
```

## 快速开始

### 1. 环境配置

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY="sk-xxxxxxxx"

# Linux/Mac
export DASHSCOPE_API_KEY="sk-xxxxxxxx"

# 或创建 .env 文件（已在 .gitignore 中）
echo DASHSCOPE_API_KEY=sk-xxxxxxxx > .env
```

### 3. 运行

```bash
# A 榜全量（默认）
python main.py

# 指定参数
python main.py --split A --workers 8 --budget-limit 5000000

# 只处理单个领域
python main.py --domain insurance

# 跳过已有步骤（增量运行）
python main.py --skip-parse --skip-index
```

## 核心模块说明

| 模块 | 功能 | Token 消耗 |
|------|------|-----------|
| `document_parser.py` | pdfplumber 解析 PDF，表格转 Markdown，条款/章节边界识别 | 不计 |
| `indexer.py` | jieba + 金融词典分词，BM25 倒排索引，领域关键词匹配，条款号精确查找 | 不计 |
| `compressor.py` | 规则压缩（保留数字/条款号句子），证据去重聚合，Prompt 上下文构建 | 不计 |
| `agent.py` | FinancialAgent：检索→压缩→Qwen3.6-plus 推理→答案提取→校验 | 计入 |
| `answer_formatter.py` | mcq/tf 取首个有效字母，multi 去重排序，JSON 答案提取 | 不计 |
| `submit.py` | 生成 answer.csv + evidence.json，TokenScore 计算 | 不计 |

## 答案格式

| 题型 | 格式要求 | 示例 |
|------|---------|------|
| 单选 (mcq) | 单个字母 A/B/C/D | `B` |
| 多选 (multi) | 字母升序拼接，无分隔符 | `AC` |
| 判断 (tf) | A=正确，B=错误 | `A` |

## Token 预算策略

- **总预算**: 5,000,000 tokens
- **满分线**: 3,600,000 tokens（TokenScore = 100）
- **零分线**: 5,000,000 tokens
- **三级压缩**: normal → moderate（80% 预算）→ aggressive（95% 预算）

## 技术约束

- 模型：仅使用 Qwen3.6-plus（阿里云百炼 / 魔搭 API）
- 检索：严禁任何 Embedding 模型（BM25 + 规则）
- 文档解析：离线处理，不计 Token（pdfplumber + PyMuPDF）
- Python：3.10+

## 许可证

本项目仅用于 AFAC2026 竞赛。
