"""
Phép loại bỏ học tăng cường (ablation "không RL").

Câu hỏi: nếu bỏ PPO và thay bằng cách đơn giản hơn để biến biểu diễn đồ thị thành danh mục, kết quả có
kém đi không? Hai phương án thay thế, cùng tập cổ phiếu, cùng phí 10 bps, cùng top-20:

  (1) 'sup'  — HỌC CÓ GIÁM SÁT: cùng bộ mã hoá Transformer + GAT (đồ thị dài hạn A250), thay tác tử PPO
               bằng một lớp tuyến tính cho điểm từng cổ phiếu. Hàm mất mát: sai số bình phương giữa điểm
               và lợi suất kỳ nắm giữ kế tiếp đã chuẩn hoá theo mặt cắt (z-score, kẹp ±3).
               Ngân sách tối ưu giống PPO: 12 vòng × 4 lượt cập nhật toàn bộ, Adam lr 3e-4, kẹp gradient 1.
               Chọn mô hình theo tỷ số Sharpe trên tập kiểm định (tắt dropout). Danh mục: 20 mã điểm cao nhất,
               tỷ trọng đều 5%.
  (2) 'lowdeg' — QUY TẮC KHÔNG HỌC trên đồ thị: chọn 20 mã có bậc nhỏ nhất trong đồ thị dài hạn, tỷ trọng đều
               (xem lowdeg_variants.py, có xử lý thế hòa).

Chạy trên cả cách chia chính (kiểm tra 2024–2025, 10 hạt giống) và kiểm định cuốn chiếu 2020–2025 (8 hạt giống).
Code gốc không bị sửa. Kết quả: results/sup_seed*.pkl, results/wf/sup_y*_seed*.pkl
"""
import io, os, sys, time, pickle, contextlib, warnings
from multiprocessing import Pool
import numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
import common as C
import run_experiments as RE
import run_extra as RX

TOP_K, COST, N_ITERS, EPOCHS, LR = 20, 0.001, 12, 4, 3e-4


def episodes(df, dates):
    eps = []
    for j in range(len(dates) - 1):
        d, dn = dates[j], dates[j + 1]
        X, mask, syms = C.train_mod.build_snapshot_tensor(df, d)
        if int(mask.sum()) < 2:
            continue
        A = C.adj.mask_adjacency(C.train_mod.build_adjacency(df, d, syms), mask)
        R = C.train_mod.compute_holding_return(df, d, dn, syms)
        eps.append({'d': d, 'X': X, 'A': A, 'mask': mask, 'R': R, 'syms': syms})
    return eps


def scores(net, ep):
    enc, gnn, head = net
    Z = gnn(C.te.apply_mask_to_embedding(enc(ep['X']), ep['mask']), ep['A'])
    return head(Z).squeeze(-1)


def portfolio(s, mask):
    s = s.detach().numpy().copy(); m = mask.numpy()
    s[~m] = -np.inf
    w = np.zeros(len(s)); w[np.argsort(s)[::-1][:TOP_K]] = 1.0 / TOP_K
    return w


def backtest(net, eps):
    for m in net: m.eval()
    rets, turns, W, SY, D = [], [], [], [], []; wp = None
    with torch.no_grad():
        for ep in eps:
            w = portfolio(scores(net, ep), ep['mask'])
            t = float(np.abs(w - (wp if wp is not None else 0.0)).sum())
            rets.append(float(w @ ep['R']) - COST * t); turns.append(t); wp = w
            W.append(w); SY.append(ep['syms']); D.append(ep['d'])
    for m in net: m.train()
    return {'returns': rets, 'turnover': turns, 'weights': W, 'symbols': SY, 'dates': D,
            'metrics': C.perf_metrics(rets, 12)}


def train(df, sp, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    mods = C.make_models('transformer', 'gat')
    net = (mods['encoder'], mods['gnn'], nn.Linear(C.D_MODEL, 1))
    params = [p for m in net for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=LR)
    tr, va = episodes(df, sp['train']), episodes(df, sp['val'])
    for ep in tr:                                             # mục tiêu: z-score theo mặt cắt
        m = ep['mask'].numpy(); r = ep['R'][m]
        z = np.zeros(len(ep['R'])); z[m] = np.clip((r - r.mean()) / (r.std() + 1e-8), -3, 3)
        ep['y'] = torch.tensor(z, dtype=torch.float32)
    best, best_state, hist = -np.inf, None, []
    for it in range(N_ITERS):
        for _ in range(EPOCHS):
            opt.zero_grad(); tot = 0.0
            for ep in tr:
                s = scores(net, ep); m = ep['mask']
                loss = ((s[m] - ep['y'][m]) ** 2).mean() / len(tr)
                loss.backward(); tot += float(loss)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        vs = backtest(net, va)['metrics']['sharpe'] if len(va) > 1 else float('nan')
        hist.append({'loss': tot, 'val_sharpe': vs})
        if not np.isnan(vs) and vs > best:
            best = vs; best_state = [{k: v.clone() for k, v in m.state_dict().items()} for m in net]
    if best_state is not None:
        for m, st in zip(net, best_state): m.load_state_dict(st)
    return net, hist, best


def run_job(job):
    torch.set_num_threads(1)
    df = C.load_df(); cd = C.CachedData(graph='A250', freq='M').install()
    if job[0] == 'main':
        seed = job[1]; path = os.path.join(C.RESULTS, f'sup_seed{seed}.pkl')
        sp = C.quiet_split(df, 'M')
    else:
        _, year, seed = job; path = os.path.join(C.RESULTS, 'wf', f'sup_y{year}_seed{seed}.pkl')
        sp = RX.wf_split(sorted(cd.snap.keys()), year)
    if os.path.exists(path):
        return job, 'skip'
    t = time.time()
    net, hist, best = train(df, sp, seed)
    out = {'config': 'sup', 'seed': seed, 'history': hist, 'best_val_sharpe': best,
           'test': backtest(net, episodes(df, sp['test'])), 'train_seconds': time.time() - t}
    if job[0] == 'wf':
        out['year'] = job[1]
    pickle.dump(out, open(path, 'wb'))
    return job, round(time.time() - t), round(out['test']['metrics']['sharpe'], 3)


if __name__ == '__main__':
    jobs = [('wf', y, s) for y in sorted(RX.WF_YEARS, reverse=True) for s in range(1, 9)]
    jobs = [('main', s) for s in range(1, 11)] + jobs
    print(len(jobs), 'jobs', flush=True)
    with Pool(4, maxtasksperchild=1) as p:
        for r in p.imap_unordered(run_job, jobs):
            print(time.strftime('%H:%M:%S'), r, flush=True)
