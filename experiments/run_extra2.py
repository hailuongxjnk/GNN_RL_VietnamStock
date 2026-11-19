"""
Đợt bổ sung thứ hai (sau run_extra.py):
  - kiểm định cuốn chiếu: thêm hạt giống 4–8 cho cả đồ thị dài hạn và ngắn hạn (→ 8 hạt giống mỗi cấu hình);
  - tập thanh khoản: thêm hạt giống 6–10 cho đồ thị dài hạn; đồ thị ngắn hạn trên tập thanh khoản, hạt giống 1–5.
Cùng mã và siêu tham số với run_extra.py. Nhật ký: results/run_extra2.log
"""
import os, sys, time
from multiprocessing import Pool
import run_extra as RX
RX.RE.CONFIGS.append(('full_liq', 'transformer', 'gat', 'A60_liq'))

if __name__ == '__main__':
    jobs = []
    for year in sorted(RX.WF_YEARS, reverse=True):          # năm dài trước để xếp lịch tốt hơn
        for seed in range(4, 9):
            for cfg in RX.WF_CFG:
                jobs.append(('wf', cfg, year, seed))
    for seed in range(6, 11):
        jobs.append(('cfg', 'tf_gat_long_liq', 'transformer', 'gat', 'A250_liq', seed))
    for seed in range(1, 6):
        jobs.append(('cfg', 'full_liq', 'transformer', 'gat', 'A60_liq', seed))
    log = open(os.path.join(RX.C.RESULTS, 'run_extra2.log'), 'a')
    print(len(jobs), 'jobs', file=log, flush=True)
    with Pool(4, maxtasksperchild=1) as p:
        for r in p.imap_unordered(RX.run_job, jobs):
            print(time.strftime('%H:%M:%S'), r, file=log, flush=True)
