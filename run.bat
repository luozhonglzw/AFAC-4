@echo off
REM A榜运行
echo === A榜运行 ===
python main.py --split A --workers 3 --output-dir submission_a\

REM B榜运行
echo === B榜运行 ===
python main.py --split B --blind --workers 3 --output-dir submission_b\

pause
