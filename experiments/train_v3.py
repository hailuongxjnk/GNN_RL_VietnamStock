"""
Hai cải tiến cho thành phần học tăng cường (Mục 4.12 của luận văn). Code gốc KHÔNG bị sửa: vòng huấn luyện
dưới đây là bản sao của train_v2.train_ppo (vòng huấn luyện dùng cho mọi thí nghiệm), chỉ thêm hai lựa chọn:

  cost='tier'   — phí giao dịch phụ thuộc thanh khoản ngay TRONG huấn luyện (và khi chọn mô hình trên tập
                  kiểm định): tại ngày quyết định, 30% mã kém thanh khoản nhất theo Amihud 20 phiên chịu 50 bps,
                  40% ở giữa 25 bps, 30% thanh khoản nhất 10 bps (giống phép tính lại ở Mục 4.12/robustness2.py).
  reward='excess' — phần thưởng DSR tính trên lợi suất VƯỢT quy tắc bậc thấp (giữ mọi mã có bậc <= bậc nhỏ thứ 20
                  trên đồ thị A250, tỷ trọng đều, cùng mô hình phí). Mục tiêu: tác tử chỉ được thưởng khi làm tốt
                  hơn mốc đơn giản, thay vì được thưởng cho biến động chung của thị trường.

Ba cấu hình:
  'tier'     : phí theo thanh khoản, thưởng DSR như gốc, 12 vòng.
  'excess'   : phí 10 bps, thưởng DSR trên lợi suất vượt quy tắc, 20 vòng.
  'long20'   : phí 10 bps, thưởng DSR như gốc, 20 vòng (đối chứng: tách tác động của số vòng khỏi tác động của thưởng).
Mọi thứ khác như mô hình đề xuất: Transformer + GAT, A250, top-20, trần 10%, chọn theo Sharpe tập kiểm định.

Chạy: python train_v3.py [n_proc]
Kết quả: results/v3_<cfg>[_liq]_seed*.pkl, results/wf/v3_<cfg>_y*_seed*.pkl
"""
import io, os, sys, time, pickle, contextlib, warnings
from multiprocessing import Pool
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
import common as C
import run_experiments as RE
import run_extra as RX
T = C.train_mod

CFG = {'tier': dict(cost='tier', reward='dsr', n_iters=12),
       'excess': dict(cost='flat', reward='excess', n_iters=20),
       'long20': dict(cost='flat', reward='dsr', n_iters=20),
       'base': dict(cost='flat', reward='dsr', n_iters=12)}   # kiểm tra: phải tái lập đúng mô hình đề xuất
FLAT, K = 0.001, 20
_AMI = None


def tier_vec(df, d, syms, mask):
    global _AMI
    if _AMI is None:
        _AMI = df.pivot(index='date', columns='symbol', values='log_amihud_20d')
    a = _AMI.loc[_AMI.index < pd.Timestamp(d)].iloc[-1].reindex(syms).values.astype(float)
    m = mask & np.isfinite(a); c = np.full(len(syms), 0.0025)
    if m.sum() > 3:
        q = pd.Series(a[m]).rank(pct=True).values
        c[m] = np.where(q > 0.7, 0.005, np.where(q <= 0.3, 0.001, 0.0025))
    return c


def rule_weights(A, mask):
    A = A.numpy(); m = mask.numpy()
    deg = np.where(m, (A != 0).sum(1) - (np.diag(A) != 0), np.inf)
    sel = np.where(deg <= np.sort(deg)[K - 1])[0]
    w = np.zeros(len(m)); w[sel] = 1.0 / len(sel)
    return w


def build_episodes(df, dates):
    eps = []
    for j in range(len(dates) - 1):
        d, dn = dates[j], dates[j + 1]
        X, mask, syms = T.build_snapshot_tensor(df, d)
        if int(mask.sum()) < 2:
            continue
        A = C.adj.mask_adjacency(T.build_adjacency(df, d, syms), mask)
        R = T.compute_holding_return(df, d, dn, syms)
        eps.append({'d': d, 'X': X, 'A': A, 'mask': mask, 'R': R, 'syms': syms,
                    'c_tier': tier_vec(df, d, syms, mask.numpy()), 'w_rule': rule_weights(A, mask)})
    return eps


def cost_of(ep, dw, mode):
    return float((np.abs(dw) * (ep['c_tier'] if mode == 'tier' else FLAT)).sum())


def rule_net(eps, mode):
    out, wp = [], None
    for ep in eps:
        w = ep['w_rule']; out.append(float(w @ ep['R']) - cost_of(ep, w - (wp if wp is not None else 0.0), mode)); wp = w
    return out


def run_policy(net, eps, deterministic=True):
    enc, gnn, agent = net
    W = []
    with torch.no_grad():
        for ep in eps:
            Z = gnn(C.te.apply_mask_to_embedding(enc(ep['X']), ep['mask']), ep['A'])
            w, _, _ = agent.act(Z, ep['mask'], deterministic=deterministic)
            W.append(T.project_weights(w.numpy(), top_k=RE.TOP_K, w_max=RE.W_MAX))
    return W


def net_returns(W, eps, mode):
    out, wp = [], None
    for w, ep in zip(W, eps):
        out.append(float(w @ ep['R']) - cost_of(ep, w - (wp if wp is not None else 0.0), mode)); wp = w
    return out


def sharpe(r):
    r = np.asarray(r); return float(r.mean() / (r.std() + 1e-12) * np.sqrt(12)) if len(r) > 1 else float('nan')


def train(df, sp, cfg, seed, ppo_epochs=4, lr=3e-4, clip=0.2, c_v=0.5, c_e=0.01, gamma=0.99, lam=0.95, reward_clip=5.0):
    """Bản sao train_v2.train_ppo với phí theo thanh khoản và thưởng vượt quy tắc (tùy chọn)."""
    o = CFG[cfg]
    torch.manual_seed(seed); np.random.seed(seed)
    m = C.make_models('transformer', 'gat'); enc, gnn, agent = m['encoder'], m['gnn'], m['agent']
    torch.manual_seed(seed); np.random.seed(seed)             # như train_ppo: đặt lại hạt giống sau khi tạo mô hình
    params = list(enc.parameters()) + list(gnn.parameters()) + list(agent.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    eps, veps = build_episodes(df, sp['train']), build_episodes(df, sp['val'])
    base = rule_net(eps, o['cost']) if o['reward'] == 'excess' else [0.0] * len(eps)
    best, best_state, hist = -np.inf, None, []
    for it in range(o['n_iters']):
        dsr = T.DSR(); wp = None
        rewards, values, logp_old, actions = [], [], [], []
        with torch.no_grad():
            for k, ep in enumerate(eps):
                Z = gnn(C.te.apply_mask_to_embedding(enc(ep['X']), ep['mask']), ep['A'])
                w, lp, v = agent.act(Z, ep['mask'])
                wn = T.project_weights(w.numpy(), top_k=RE.TOP_K, w_max=RE.W_MAX)
                net = float(wn @ ep['R']) - cost_of(ep, wn - (wp if wp is not None else 0.0), o['cost'])
                rewards.append(float(np.clip(dsr.step(net - base[k]), -reward_clip, reward_clip)))
                values.append(float(v)); logp_old.append(float(lp)); actions.append(w); wp = wn
        adv, rets = T.compute_gae(rewards, values, gamma, lam)
        adv_t = torch.tensor(adv, dtype=torch.float32); adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(rets, dtype=torch.float32); lpold = torch.tensor(logp_old, dtype=torch.float32)
        for _ in range(ppo_epochs):
            opt.zero_grad()
            for k, ep in enumerate(eps):
                Z = gnn(C.te.apply_mask_to_embedding(enc(ep['X']), ep['mask']), ep['A'])
                lp, ent, v = agent.evaluate(Z, ep['mask'], actions[k])
                ratio = torch.exp(lp - lpold[k])
                a_loss = -torch.min(ratio * adv_t[k], torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[k])
                ((a_loss + c_v * (v - ret_t[k]) ** 2 - c_e * ent) / len(eps)).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        for x in (enc, gnn, agent): x.eval()                  # như train_v2 [V2-1]: tắt dropout khi chấm validation
        vs = sharpe(net_returns(run_policy((enc, gnn, agent), veps), veps, o['cost'])) if len(veps) > 1 else float('nan')
        for x in (enc, gnn, agent): x.train()
        hist.append({'reward': float(np.mean(rewards)), 'val_sharpe': vs})
        if not np.isnan(vs) and vs > best:
            best = vs; best_state = [{a: b.clone() for a, b in x.state_dict().items()} for x in (enc, gnn, agent)]
    if best_state is not None:
        for x, st in zip((enc, gnn, agent), best_state): x.load_state_dict(st)
    return (enc, gnn, agent), hist, best


def evaluate(net, eps):
    for x in net: x.eval()
    W = run_policy(net, eps)
    rf, rt = net_returns(W, eps, 'flat'), net_returns(W, eps, 'tier')
    turn = [float(np.abs(w - (W[i - 1] if i else 0.0)).sum()) for i, w in enumerate(W)]
    return {'returns': rf, 'returns_tier': rt, 'turnover': turn, 'weights': W,
            'symbols': [e['syms'] for e in eps], 'dates': [e['d'] for e in eps],
            'metrics': C.perf_metrics(rf, 12), 'metrics_tier': C.perf_metrics(rt, 12),
            'rule_flat': rule_net(eps, 'flat'), 'rule_tier': rule_net(eps, 'tier')}


def run_job(job):
    torch.set_num_threads(1)
    cfg, kind = job[0], job[1]
    df = C.load_df(); t = time.time()
    if kind == 'wf':
        year, seed = job[2], job[3]; path = os.path.join(C.RESULTS, 'wf', f'v3_{cfg}_y{year}_seed{seed}.pkl')
        if os.path.exists(path): return job, 'skip'
        cd = C.CachedData(graph='A250', freq='M').install(); sp = RX.wf_split(sorted(cd.snap.keys()), year)
    else:
        seed = job[2]; graph = 'A250' if kind == 'main' else 'A250_liq'
        path = os.path.join(C.RESULTS, f"v3_{cfg}{'' if kind == 'main' else '_liq'}_seed{seed}.pkl")
        if os.path.exists(path): return job, 'skip'
        C.CachedData(graph=graph, freq='M').install(); sp = C.quiet_split(df, 'M')
    net, hist, best = train(df, sp, cfg, seed)
    out = {'config': cfg, 'kind': kind, 'seed': seed, 'history': hist, 'best_val_sharpe': best,
           'test': evaluate(net, build_episodes(df, sp['test'])), 'train_seconds': time.time() - t}
    if kind == 'wf': out['year'] = year
    pickle.dump(out, open(path, 'wb'))
    m = out['test']
    return job, round(time.time() - t), round(m['metrics']['sharpe'], 3), round(m['metrics_tier']['sharpe'], 3)


if __name__ == '__main__':
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs = [('base', 'main', 1)] + [('tier', 'main', s) for s in range(1, 11)] + [('excess', 'main', s) for s in range(1, 9)]
    jobs += [('long20', 'main', s) for s in range(1, 9)]
    jobs += [('tier', 'liq', s) for s in range(1, 6)] + [('excess', 'liq', s) for s in range(1, 6)]
    for cfg in ('tier', 'excess'):
        jobs += [(cfg, 'wf', y, s) for y in sorted(RX.WF_YEARS, reverse=True) for s in range(1, 5)]
    print(len(jobs), 'jobs', flush=True)
    with Pool(n_proc, maxtasksperchild=1) as p:
        for r in p.imap_unordered(run_job, jobs):
            print(time.strftime('%H:%M:%S'), r, flush=True)
