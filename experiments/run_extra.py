"""
Thí nghiệm bổ sung (chạy một lần, mọi kết quả đều được báo cáo):
  (A) Thêm hạt giống 6–10 cho mô hình chính (đồ thị dài hạn) và đồ thị ngắn hạn → 10 hạt giống mỗi cấu hình.
  (B) Tập cổ phiếu THANH KHOẢN (build_liquid_cache.py): mô hình chính, 5 hạt giống.
  (C) Đánh giá CUỐN CHIẾU (walk-forward): mỗi năm kiểm tra Y ∈ {2020,…,2025}:
        huấn luyện: các tháng đến hết năm Y-2; kiểm định: năm Y-1; kiểm tra: năm Y.
      Hai cấu hình (đồ thị dài hạn / ngắn hạn) × 3 hạt giống × 6 năm. Không chọn năm, không bỏ năm nào.
Kết quả: results/*_seed*.pkl (A, B) và results/wf/*.pkl (C).
"""
import io, os, sys, time, pickle, contextlib, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
import common as C
import run_experiments as RE

RE.CONFIGS.append(('tf_gat_long_liq', 'transformer', 'gat', 'A250_liq'))
WF_YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
WF_CFG = {'tf_gat_long': 'A250', 'full': 'A60'}


def wf_split(all_dates, year):
    tr = [d for d in all_dates if d.year <= year - 2]
    va = [d for d in all_dates if d.year == year - 1]
    te = [d for d in all_dates if d.year == year]
    nxt = [d for d in all_dates if d.year == year + 1]
    if nxt: te = te + nxt[:1]                         # để kỳ nắm giữ cuối của năm Y có ngày kết thúc
    va = va + te[:1]
    return {'train': tr, 'val': va, 'test': te}


def run_wf(job):
    _, cfg, year, seed = job
    path = os.path.join(C.RESULTS, 'wf', f'{cfg}_y{year}_seed{seed}.pkl')
    if os.path.exists(path):
        return cfg, year, seed, 'skip'
    torch.set_num_threads(1)
    graph = WF_CFG[cfg]
    df = C.load_df(); cd = C.CachedData(graph=graph, freq='M').install()
    sp = wf_split(sorted(cd.snap.keys()), year)
    C.train_mod.split_dates = lambda df_, freq='M', **k: sp
    C.quiet_split = lambda df_, freq: sp
    torch.manual_seed(seed); np.random.seed(seed)
    init = C.make_models('transformer', 'gat'); t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        res = C.train_mod.train_ppo(df, n_iters=RE.N_ITERS, freq='M', top_k=RE.TOP_K, w_max=RE.W_MAX,
                                    cost_rate=RE.COST, seed=seed, device='cpu', init=init, verbose=False)
    bt = RE.backtest(res, df, 'test')
    out = {'config': cfg, 'year': year, 'seed': seed, 'history': res['history'],
           'n_train': len(sp['train']) - 1, 'test': bt, 'best_val_sharpe': res['best_val_sharpe'],
           'train_seconds': time.time() - t}
    pickle.dump(out, open(path, 'wb'))
    return cfg, year, seed, round(time.time() - t)


def run_job(job):
    if job[0] == 'wf':
        return run_wf(job)
    return RE.run_one(job[1:])


if __name__ == '__main__':
    jobs = []
    # (C) walk-forward: năm có quỹ đạo huấn luyện dài nhất trước để cân tải
    for year in sorted(WF_YEARS, reverse=True):
        for seed in (1, 2, 3):
            for cfg in ('tf_gat_long', 'full'):
                jobs.append(('wf', cfg, year, seed))
    # (A) thêm hạt giống
    for seed in range(6, 11):
        jobs.append(('cfg', 'tf_gat_long', 'transformer', 'gat', 'A250', seed))
        jobs.append(('cfg', 'full', 'transformer', 'gat', 'A60', seed))
    # (B) tập thanh khoản
    for seed in range(1, 6):
        jobs.append(('cfg', 'tf_gat_long_liq', 'transformer', 'gat', 'A250_liq', seed))
    print(len(jobs), 'jobs', flush=True)
    # maxtasksperchild=1: mỗi lần chạy một tiến trình mới, để các thay đổi chia dữ liệu của walk-forward
    # không ảnh hưởng sang lần chạy khác
    with Pool(4, maxtasksperchild=1) as p:
        for r in p.imap_unordered(run_job, jobs):
            print(time.strftime('%H:%M:%S'), r, flush=True)
