import json, time, os, sys

log_file = "submission_a/checkpoint.json"
last_count = 0
start_time = time.time()

while True:
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            completed = len(data.get("completed_qids", []))
            total_tokens = data.get("total_tokens", 0)

            if completed != last_count:
                elapsed = time.time() - start_time
                avg_time = elapsed / completed if completed > 0 else 0
                remaining = (100 - completed) * avg_time
                print(f"[{time.strftime('%H:%M:%S')}] "
                      f"完成: {completed}/100 | "
                      f"Token: {total_tokens:,}/5,000,000 ({total_tokens/5000000*100:.1f}%) | "
                      f"预计剩余: {remaining/60:.1f}分钟")
                last_count = completed

            if completed >= 100:
                print("全部完成！")
                break
        except Exception as e:
            print(f"监控错误: {e}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] 等待 checkpoint 文件...")
    time.sleep(10)
