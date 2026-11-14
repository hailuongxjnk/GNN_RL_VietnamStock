#!/bin/bash
# Chờ 10 lần chạy chính (đồ thị 60 và 250 phiên, mỗi loại 5 hạt giống) xong rồi chạy các phép loại bỏ
# thành phần trên đồ thị DÀI HẠN (cấu hình chính của luận văn).
cd "$(dirname "$0")"
while [ "$(ls results/full_seed*.pkl results/tf_gat_long_seed*.pkl 2>/dev/null | wc -l)" -lt 10 ]; do sleep 30; done
pkill -f "run_experiments.py$" ; sleep 3
python3 run_experiments.py tf_gcn_long tf_nograph lstm_gat_long lstm_nograph gru_nograph >> results/run.log 2>&1
