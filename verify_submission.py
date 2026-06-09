import csv
import json
import os
import sys

def verify():
    errors = []
    warnings = []

    # 1. 检查文件存在
    if not os.path.exists('submission_a/answer.csv'):
        errors.append("answer.csv 不存在")
        return errors, warnings
    if not os.path.exists('submission_a/evidence.json'):
        warnings.append("evidence.json 不存在（建议生成）")

    # 2. 检查 answer.csv 格式
    with open('submission_a/answer.csv', 'r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        errors.append("answer.csv 为空")
        return errors, warnings

    # 第一行必须是 summary
    if rows[0][0] != 'summary':
        errors.append(f"第一行必须是 summary，实际是 {rows[0][0]}")
    else:
        try:
            prompt_tokens = int(rows[0][2]) if rows[0][2] else 0
            completion_tokens = int(rows[0][3]) if rows[0][3] else 0
            total_tokens = int(rows[0][4]) if rows[0][4] else 0
            if total_tokens <= 0:
                errors.append("summary total_tokens 必须 > 0")
            if total_tokens >= 5000000:
                errors.append(f"Token 超限: {total_tokens:,} >= 5,000,000")
            else:
                token_score = (5000000 - total_tokens) / 5000000
                print(f"[OK] Token 统计: prompt={prompt_tokens:,}, completion={completion_tokens:,}, total={total_tokens:,}")
                print(f"[OK] TokenScore: {token_score:.4f} (满分=1.0)")
        except Exception as e:
            errors.append(f"summary 行格式错误: {e}")

    # 3. 检查 100 题
    data_rows = rows[1:]
    if len(data_rows) != 100:
        errors.append(f"题目数量错误: {len(data_rows)}，应为 100")

    qids = []
    for i, row in enumerate(data_rows, 1):
        if len(row) < 2:
            errors.append(f"第{i}行格式错误: {row}")
            continue

        qid = row[0]
        answer = row[1]
        qids.append(qid)

        # 检查 qid 格式
        if not qid.startswith(('ins_a_', 'reg_a_', 'fc_a_', 'fin_a_', 'res_a_')):
            warnings.append(f"第{i}行 qid 格式异常: {qid}")

        # 检查答案
        if not answer:
            errors.append(f"{qid} 答案为空")
            continue

        # 检查非法字符
        if not all(c in 'ABCD' for c in answer):
            errors.append(f"{qid} 答案含非法字符: {answer}")
            continue

        # 检查多选排序
        if len(answer) > 1:
            sorted_answer = ''.join(sorted(answer))
            if answer != sorted_answer:
                errors.append(f"{qid} 多选未按字母序排列: {answer}，应为 {sorted_answer}")
            # 检查重复
            if len(set(answer)) != len(answer):
                errors.append(f"{qid} 多选有重复字母: {answer}")

    # 检查重复 qid
    if len(qids) != len(set(qids)):
        errors.append("存在重复 qid")

    # 4. 检查 evidence.json
    if os.path.exists('submission_a/evidence.json'):
        try:
            with open('submission_a/evidence.json', 'r', encoding='utf-8') as f:
                evidence = json.load(f)
            if len(evidence) < 50:
                warnings.append(f"evidence.json 只有 {len(evidence)} 条，建议 100 条")
            else:
                print(f"[OK] evidence.json: {len(evidence)} 条记录")
        except Exception as e:
            warnings.append(f"evidence.json 解析错误: {e}")

    # 5. 输出结果
    print(f"\n{'='*60}")
    print(f"检查结果: {len(errors)} 个错误, {len(warnings)} 个警告")
    if errors:
        print("\n错误列表:")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print("\n警告列表:")
        for w in warnings:
            print(f"  - {w}")

    if not errors:
        print("\n全部检查通过！可以提交到天池平台。")
        print(f"提交文件: submission_a/answer.csv")
        return [], warnings
    else:
        print("\n存在错误，请修复后再提交！")
        return errors, warnings

if __name__ == '__main__':
    errors, warnings = verify()
    sys.exit(1 if errors else 0)
