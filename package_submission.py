import os
import zipfile

def package():
    zip_path = 'submission.zip'

    # 如果已存在，删除旧包
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"删除旧包: {zip_path}")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 1. 核心提交文件
        files_to_add = [
            'submission_a/answer.csv',
            'submission_a/evidence.json',
        ]

        for f in files_to_add:
            if os.path.exists(f):
                zf.write(f, os.path.basename(f))
                print(f"添加: {f}")
            else:
                print(f"缺失: {f}")

        # 2. 代码目录
        code_dirs = ['src/', 'prompts/']
        for dir_path in code_dirs:
            if os.path.exists(dir_path):
                for root, dirs, files in os.walk(dir_path):
                    for file in files:
                        if file.endswith('.py') or file.endswith('.txt'):
                            full_path = os.path.join(root, file)
                            arcname = full_path.replace('\\', '/')
                            zf.write(full_path, arcname)
                            print(f"添加代码: {arcname}")

        # 3. 配置文件
        config_files = ['requirements.txt', 'main.py']
        for f in config_files:
            if os.path.exists(f):
                zf.write(f, f)
                print(f"添加配置: {f}")

        # 检查总大小
        final_size = os.path.getsize(zip_path)
        print(f"\n{'='*60}")
        print(f"打包完成: {zip_path}")
        print(f"文件大小: {final_size/1024/1024:.2f}MB")
        if final_size > 1024 * 1024 * 1024:
            print("超过 1GB 限制！需要精简")
        else:
            print("大小符合要求 (< 1GB)")

        # 列出内容
        print(f"\n包内文件列表:")
        with zipfile.ZipFile(zip_path, 'r') as zf_read:
            for name in zf_read.namelist()[:30]:
                print(f"  {name}")
            if len(zf_read.namelist()) > 30:
                print(f"  ... 共 {len(zf_read.namelist())} 个文件")

if __name__ == '__main__':
    package()
