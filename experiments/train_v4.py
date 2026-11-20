"""
Thay PPO bằng thuật toán học khác, giữ nguyên Transformer + GAT, đồ thị A250, top-20, trần 10%, phí và cách chọn
mô hình (Sharpe tập kiểm định, chấm 12 lần). Code gốc KHÔNG bị sửa; dùng lại các hàm của train_v3.py.

  'a2c'    : actor–critic có lợi thế (A2C). Như PPO nhưng KHÔNG cắt tỷ số xác suất và mỗi lô dữ liệu chỉ dùng
             cho MỘT bước gradient. 48 vòng = 48 bước gradient, đúng bằng PPO (12 vòng x 4 epoch).
  'direct' : tối ưu trực tiếp tỷ số Sharpe (không phải học tăng cường). Lợi suất ròng w·R − phí·|Δw| là hàm khả vi
             theo tỷ trọng, nên lấy đạo hàm thẳng của Sharpe trên toàn giai đoạn huấn luyện theo tham số mạng
             (Zhang, Zohren & Roberts 2020), thay vì ước lượng gradient bằng lấy mẫu như PPO.
             Tỷ trọng huấn luyện = kỳ vọng Dirichlet alpha/sum(alpha) với top-20 MỀM (cổng sigmoid quanh ngưỡng
             giữa mã thứ 20 và 21, tau = 0,05 trên log alpha); khi đánh giá vẫn dùng đúng project_weights như PPO.
             Trần 10% bỏ qua khi huấn luyện vì không bao giờ chạm (Mục 4.12). 48 bước gradient toàn lô.
             Gradient chính xác được tính theo hai lượt để không phải giữ đồ thị tính toán của cả 80 kỳ:
             lượt 1 (không gradient) tính mọi w_t và r_t, suy ra dS/dr_t; lượt 2 lan truyền ngược từng kỳ với
             hàm thay thế g_t·r_t(w_t, w_{t−1}) + g_{t+1}·(−phí·|w_{t+1} − w_t|). Dropout giữ nguyên giữa hai lượt
             nhờ đặt cùng hạt giống cho mỗi kỳ.
  'direct_tier' : như 'direct' nhưng phí theo thanh khoản ngay trong hàm mục tiêu.

Chạy: python train_v4.py [n_proc]
Kết quả: results/v4_<cfg>[_liq]_seed*.pkl, results/wf/v4_<cfg>_y*_seed*.pkl (cùng định dạng train_v3)
"""
import os, sys, time, pickle, warnings
from multiprocessing import Pool
import numpy as np, torch
warnings.filterwarnings('ignore')
import common as C
import run_experiments as RE
import run_extra as RX
import train_v3 as V
T = C.train_mod

CFG = {'a2c': dict(algo='a2c', cost='flat'), 'direct': dict(algo='direct', cost='flat'),
       'direct_tier': dict(algo='direct', cost='tier')}
N_STEPS, EVAL_EVERY, TAU = 48, 4, 0.05


def embed(net, ep):
    enc, gnn, _ = net
    return gnn(C.te.apply_mask_to_embedding(enc(ep['X']), ep['mask']), ep['A'])


def validate(net, veps, cost):
    for x in net: x.eval()
    vs = V.sharpe(V.net_returns(V.run_policy(net, veps), veps, cost)) if len(veps) > 1 else float('nan')
    for x in net: x.train()
    return vs


def soft_weights(agent, Z, mask):
    a = agent._alpha(Z); m = mask.bool()
    la = torch.log(a)
    v = la[m].detach().sort(descending=True).values
    thr = (v[V.K - 1] + v[V.K]) / 2 if len(v) > V.K else v[-1] - 1.0
    g = torch.sigmoid((la - thr) / TAU) * m.float()
    w = a * g
    return w / w.sum()


def cost_vec(ep, mode):
    return torch.tensor(ep['c_tier'], dtype=torch.float32) if mode == 'tier' else torch.full((len(ep['syms']),), V.FLAT)


def train_direct(net, opt, params, eps, veps, cost, seed):
    enc, gnn, agent = net
    Rt = [torch.tensor(ep['R'], dtype=torch.float32) for ep in eps]
    cv = [cost_vec(ep, cost) for ep in eps]
    best, best_state, hist = -np.inf, None, []
    for step in range(N_STEPS):
        base = seed * 100003 + step * 1000
        with torch.no_grad():                                     # lượt 1
            W, r = [], []
            for k, ep in enumerate(eps):
                torch.manual_seed(base + k); W.append(soft_weights(agent, embed(net, ep), ep['mask']))
                prev = W[k - 1] if k else torch.zeros_like(W[k])
                r.append(float(W[k] @ Rt[k] - (cv[k] * (W[k] - prev).abs()).sum()))
        r = np.array(r); n = len(r); mu, sd = r.mean(), r.std() + 1e-12
        g = np.sqrt(12) * (1.0 / (n * sd) - (r - mu) * mu / (n * sd ** 3))   # dS/dr_t
        opt.zero_grad()
        for k, ep in enumerate(eps):                              # lượt 2
            torch.manual_seed(base + k); w = soft_weights(agent, embed(net, ep), ep['mask'])
            prev = W[k - 1] if k else torch.zeros_like(w)
            sur = g[k] * (w @ Rt[k] - (cv[k] * (w - prev).abs()).sum())
            if k + 1 < n:
                sur = sur + g[k + 1] * (-(cv[k + 1] * (W[k + 1] - w).abs()).sum())
            (-sur).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        rec = {'train_sharpe_soft': float(mu / sd * np.sqrt(12))}
        if (step + 1) % EVAL_EVERY == 0:
            vs = validate(net, veps, cost); rec['val_sharpe'] = vs
            if not np.isnan(vs) and vs > best:
                best = vs; best_state = [{a: b.clone() for a, b in x.state_dict().items()} for x in net]
        hist.append(rec)
    return best, best_state, hist


def train_a2c(net, opt, params, eps, veps, cost, c_v=0.5, c_e=0.01, gamma=0.99, lam=0.95, reward_clip=5.0):
    enc, gnn, agent = net
    best, best_state, hist = -np.inf, None, []
    for it in range(N_STEPS):
        dsr = T.DSR(); wp = None
        rewards, values, actions = [], [], []
        with torch.no_grad():
            for ep in eps:
                w, _, v = agent.act(embed(net, ep), ep['mask'])
                wn = T.project_weights(w.numpy(), top_k=RE.TOP_K, w_max=RE.W_MAX)
                rnet = float(wn @ ep['R']) - V.cost_of(ep, wn - (wp if wp is not None else 0.0), cost)
                rewards.append(float(np.clip(dsr.step(rnet), -reward_clip, reward_clip)))
                values.append(float(v)); actions.append(w); wp = wn
        adv, rets = T.compute_gae(rewards, values, gamma, lam)
        adv_t = torch.tensor(adv, dtype=torch.float32); adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(rets, dtype=torch.float32)
        opt.zero_grad()                                           # MỘT bước gradient, không cắt tỷ số
        for k, ep in enumerate(eps):
            lp, ent, v = agent.evaluate(embed(net, ep), ep['mask'], actions[k])
            ((-lp * adv_t[k] + c_v * (v - ret_t[k]) ** 2 - c_e * ent) / len(eps)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        rec = {'reward': float(np.mean(rewards))}
        if (it + 1) % EVAL_EVERY == 0:
            vs = validate(net, veps, cost); rec['val_sharpe'] = vs
            if not np.isnan(vs) and vs > best:
                best = vs; best_state = [{a: b.clone() for a, b in x.state_dict().items()} for x in net]
        hist.append(rec)
    return best, best_state, hist


def train(df, sp, cfg, seed, lr=3e-4):
    o = CFG[cfg]
    torch.manual_seed(seed); np.random.seed(seed)
    m = C.make_models('transformer', 'gat'); net = (m['encoder'], m['gnn'], m['agent'])
    torch.manual_seed(seed); np.random.seed(seed)
    params = [p for x in net for p in x.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    eps, veps = V.build_episodes(df, sp['train']), V.build_episodes(df, sp['val'])
    f = train_direct if o['algo'] == 'direct' else train_a2c
    best, best_state, hist = (f(net, opt, params, eps, veps, o['cost'], seed) if o['algo'] == 'direct'
                              else f(net, opt, params, eps, veps, o['cost']))
    if best_state is not None:
        for x, st in zip(net, best_state): x.load_state_dict(st)
    return net, hist, best


def run_job(job):
    torch.set_num_threads(1)
    cfg, kind = job[0], job[1]
    df = C.load_df(); t = time.time()
    if kind == 'wf':
        year, seed = job[2], job[3]; path = os.path.join(C.RESULTS, 'wf', f'v4_{cfg}_y{year}_seed{seed}.pkl')
        if os.path.exists(path): return job, 'skip'
        cd = C.CachedData(graph='A250', freq='M').install(); sp = RX.wf_split(sorted(cd.snap.keys()), year)
    else:
        seed = job[2]; graph = 'A250' if kind == 'main' else 'A250_liq'
        path = os.path.join(C.RESULTS, f"v4_{cfg}{'' if kind == 'main' else '_liq'}_seed{seed}.pkl")
        if os.path.exists(path): return job, 'skip'
        C.CachedData(graph=graph, freq='M').install(); sp = C.quiet_split(df, 'M')
    net, hist, best = train(df, sp, cfg, seed)
    out = {'config': cfg, 'kind': kind, 'seed': seed, 'history': hist, 'best_val_sharpe': best,
           'test': V.evaluate(net, V.build_episodes(df, sp['test'])), 'train_seconds': time.time() - t}
    if kind == 'wf': out['year'] = year
    pickle.dump(out, open(path, 'wb'))
    m = out['test']
    return job, round(time.time() - t), round(m['metrics']['sharpe'], 3), round(m['metrics_tier']['sharpe'], 3)


if __name__ == '__main__':
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs = [(c, 'main', s) for s in range(1, 11) for c in ('direct', 'a2c', 'direct_tier')]
    jobs += [(c, 'liq', s) for s in range(1, 6) for c in ('direct', 'a2c', 'direct_tier')]
    for cfg in ('direct', 'a2c'):
        jobs += [(cfg, 'wf', y, s) for y in sorted(RX.WF_YEARS, reverse=True) for s in range(1, 5)]
    print(len(jobs), 'jobs', flush=True)
    with Pool(n_proc, maxtasksperchild=1) as p:
        for r in p.imap_unordered(run_job, jobs):
            print(time.strftime('%H:%M:%S'), r, flush=True)
