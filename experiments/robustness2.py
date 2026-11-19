"""
Kiểm tra độ vững bổ sung (không huấn luyện lại mô hình):

  (A) Độ nhạy của cấu trúc đồ thị theo cửa sổ ước lượng (120/250/500 phiên) và ngưỡng tương quan
      (0,2/0,3/0,4), đo bằng quy tắc bậc thấp không học (phiên bản 'all', không phụ thuộc thứ tự tên mã) —
      vì quy tắc không cần huấn luyện nên có thể quét toàn bộ lưới. Kèm mật độ đồ thị và tỷ lệ cạnh giữ lại qua tháng.
  (B) Khoảng tin cậy bootstrap khối (stationary bootstrap, độ dài khối trung bình 3 tháng, 5000 lần) cho chênh lệch
      tỷ số Sharpe giữa mô hình và các mốc, trên giai đoạn kiểm tra và chuỗi cuốn chiếu.
  (C) Độ nhạy theo chi phí giao dịch: tính lại lợi suất ròng với phí 20/30/50 bps (trọng số giữ nguyên), và với
      phí phụ thuộc thanh khoản (theo nhóm Amihud tại ngày quyết định: 30% kém thanh khoản nhất 50 bps,
      40% giữa 25 bps, 30% thanh khoản nhất 10 bps).
Kết quả: results/robustness2.json, thesis/tables/sens_graph.tex, bootstrap.tex, costs.tex
"""
import os, glob, json, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common as C

R = C.RESULTS; TAB = os.path.join(C.ROOT, 'thesis', 'tables')
f = lambda x, d=2: (f"{x:.{d}f}".replace('.', '{,}').replace('-', '$-$'))
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
df = C.load_df(); cd = C.CachedData('A250')
allM = sorted(cd.snap.keys()); WF = [d for d in allM if d.year >= 2020]; TEST = C.quiet_split(df, 'M')['test']
S = {}

# ───────── (A) độ nhạy đồ thị ─────────
def rule_all(adjs, dates, K=20, cost=0.001):
    rets, turns, prev = [], [], {}
    for j in range(len(dates) - 1):
        d, dn = dates[j], dates[j + 1]
        _, mask, syms = cd.snapshot(df, d); m = mask.numpy(); A = adjs[d]
        deg = np.where(m, ((A != 0).sum(1) - (np.diag(A) != 0)), np.inf)
        sel = np.where(deg <= np.sort(deg)[K - 1])[0]
        w = {syms[i]: 1.0 / len(sel) for i in sel}
        t = sum(abs(w.get(s, 0) - prev.get(s, 0)) for s in set(w) | set(prev))
        Rr = C.train_mod.compute_holding_return(df, d, dn, [syms[i] for i in sel])
        rets.append(float(np.mean(Rr)) - cost * t); turns.append(t); prev = w
    return rets, turns

if not os.path.exists(os.path.join(R, 'sens_adj.pkl')):
    cache = {}
    for win in (120, 250, 500):
        for thr in (0.2, 0.3, 0.4):
            adjs = {}
            for d in WF:
                _, mask, syms = cd.snapshot(df, d)
                n_past = int((pd.to_datetime(df['date'].unique()) < pd.Timestamp(d)).sum())
                A = C._ORIG_ADJ(df, d, syms, corr_window=min(win, n_past), min_obs=min(30, win), threshold=thr)
                adjs[d] = C.adj.mask_adjacency(A, mask).numpy().astype(np.float32)
            cache[(win, thr)] = adjs; print('adj', win, thr, flush=True)
    pickle.dump(cache, open(os.path.join(R, 'sens_adj.pkl'), 'wb'))
cache = pickle.load(open(os.path.join(R, 'sens_adj.pkl'), 'rb'))
sens = {}
for (win, thr), adjs in cache.items():
    rt, tt = rule_all(adjs, TEST); rw, tw = rule_all(adjs, WF)
    dens, keep = [], []
    for k, d in enumerate(TEST[:-1]):
        A = adjs[d]; _, mask, syms = cd.snapshot(df, d); n = int(mask.sum())
        E = (A != 0) & ~np.eye(len(A), dtype=bool); dens.append(E.sum() / max(n * (n - 1), 1))
        if k:
            Ep = adjs[TEST[k - 1]]; Ep = (Ep != 0) & ~np.eye(len(Ep), dtype=bool)
            if Ep.shape == E.shape and Ep.sum(): keep.append((E & Ep).sum() / Ep.sum())
    sens[f'{win}_{thr}'] = {'test': C.perf_metrics(rt, 12), 'wf': C.perf_metrics(rw, 12),
                            'turn_test': float(np.mean(tt[1:])), 'density': float(np.mean(dens)), 'persist': float(np.mean(keep))}
    print(win, thr, round(sens[f'{win}_{thr}']['test']['sharpe'], 2), round(sens[f'{win}_{thr}']['wf']['sharpe'], 2),
          round(np.mean(dens), 3), round(np.mean(keep), 3), flush=True)
S['sens'] = sens
tb = ("\\begin{tabular}{@{}ccccccc@{}}\n\\toprule\nCửa sổ (phiên) & Ngưỡng & Mật độ (\\%) & Cạnh giữ lại (\\%) & "
      "Sharpe 2024--2025 & Sharpe 2020--2025 & Luân chuyển \\\\\n\\midrule\n")
for win in (120, 250, 500):
    for thr in (0.2, 0.3, 0.4):
        v = sens[f'{win}_{thr}']; b = (lambda s: f"\\textbf{{{s}}}") if (win, thr) == (250, 0.3) else (lambda s: s)
        tb += f"{b(str(win))} & {b(f(thr,1))} & {f(v['density']*100,1)} & {f(v['persist']*100,0)} & {f(v['test']['sharpe'])} & {f(v['wf']['sharpe'])} & {f(v['turn_test'])} \\\\\n"
    if win != 500: tb += "\\addlinespace\n"
tb += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(TAB, 'sens_graph.tex'), 'w').write(tb)

# ───────── (B) bootstrap ─────────
def sharpe(r): return r.mean() / (r.std() + 1e-12) * np.sqrt(12)
def boot_diff(a, b, n=5000, L=3, seed=0):
    a, b = np.asarray(a), np.asarray(b); T = len(a); rng = np.random.default_rng(seed); out = np.empty(n)
    for k in range(n):
        idx = np.empty(T, int); i = rng.integers(T)
        for t in range(T):
            idx[t] = i; i = rng.integers(T) if rng.random() < 1 / L else (i + 1) % T
        out[k] = sharpe(a[idx]) - sharpe(b[idx])
    return float(sharpe(a) - sharpe(b)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), float((out <= 0).mean())

bl = pickle.load(open(os.path.join(R, 'baselines.pkl'), 'rb'))['M']; bw = pickle.load(open(os.path.join(R, 'baselines_wf.pkl'), 'rb'))
ld = pickle.load(open(os.path.join(R, 'lowdeg_variants.pkl'), 'rb'))
ppo = [pickle.load(open(fn, 'rb')) for fn in glob.glob(os.path.join(R, 'tf_gat_long_seed*.pkl'))]
ppo60 = [pickle.load(open(fn, 'rb')) for fn in glob.glob(os.path.join(R, 'full_seed*.pkl'))]
def wf_chain(cfg):
    by = {}
    for fn in glob.glob(os.path.join(R, 'wf', f'{cfg}_y*_seed*.pkl')):
        o = pickle.load(open(fn, 'rb')); by.setdefault(o['seed'], {})[o['year']] = o['test']
    return {s: ys for s, ys in by.items() if all(y in ys for y in YEARS)}
wfp = wf_chain('tf_gat_long'); wfs = wf_chain('full')
chain = lambda ys, k='returns': sum((ys[y][k] for y in YEARS), [])
ens_t = np.mean([o['test']['returns'] for o in ppo], 0); ens_w = np.mean([chain(v) for v in wfp.values()], 0)
ens_t60 = np.mean([o['test']['returns'] for o in ppo60], 0); ens_w60 = np.mean([chain(v) for v in wfs.values()], 0)
boot = {}
for per, ens, ens60, B, rule, seeds in (('test', ens_t, ens_t60, bl, ld['test_A250']['all']['returns'], [o['test']['returns'] for o in ppo]),
                                        ('wf', ens_w, ens_w60, bw, ld['wf_A250']['all']['returns'], [chain(v) for v in wfp.values()])):
    boot[per] = {}
    for lab, other in (('Equal-weight', B['Equal-weight']['returns']), ('VN-Index', B['VN-Index']['returns']),
                       ('Mean-Variance', B['Mean-Variance']['returns']), ('Quy tắc bậc thấp', rule), ('Đồ thị ngắn hạn', ens60)):
        d, lo, hi, p = boot_diff(ens, other)
        ps = [boot_diff(s, other, n=1000, seed=1)[3] for s in seeds]
        boot[per][lab] = {'diff': d, 'lo': lo, 'hi': hi, 'p': p, 'seeds_p05': int(sum(x < 0.05 for x in ps)), 'n_seeds': len(seeds)}
        print(per, lab, [round(x, 3) for x in (d, lo, hi, p)], boot[per][lab]['seeds_p05'], flush=True)
S['boot'] = boot
names = {'Equal-weight': 'Phân bổ đều (1/N)', 'VN-Index': 'VN-Index', 'Mean-Variance': 'Trung bình--phương sai',
         'Quy tắc bậc thấp': 'Quy tắc bậc thấp (đồ thị dài hạn)', 'Đồ thị ngắn hạn': 'Mô hình, đồ thị ngắn hạn'}
tb = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{2024--2025 (23 tháng)} & \\multicolumn{3}{c}{2020--2025 (71 tháng)} \\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nMô hình so với & Chênh lệch & KTC 95\\% & $p$ & Chênh lệch & KTC 95\\% & $p$ \\\\\n\\midrule\n")
for k, lab in names.items():
    a, b = boot['test'][k], boot['wf'][k]
    tb += f"{lab} & {f(a['diff'])} & [{f(a['lo'])}; {f(a['hi'])}] & {f(a['p'])} & {f(b['diff'])} & [{f(b['lo'])}; {f(b['hi'])}] & {f(b['p'])} \\\\\n"
tb += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(TAB, 'bootstrap.tex'), 'w').write(tb)

# ───────── (C) chi phí ─────────
amihud = df.pivot(index='date', columns='symbol', values='log_amihud_20d')
def tier_cost(d, syms, mask):
    a = amihud.loc[amihud.index < d].iloc[-1].reindex(syms).values.astype(float)
    m = mask & np.isfinite(a); c = np.full(len(syms), 0.0025)
    if m.sum() > 3:
        q = pd.Series(a[m]).rank(pct=True).values; cc = np.where(q > 0.7, 0.005, np.where(q <= 0.3, 0.001, 0.0025)); c[m] = cc
    return dict(zip(syms, c))
def recost_weights(seq, dates, R_of):
    """seq: list of dict symbol->weight per date; trả lợi suất gộp và phí theo bậc thanh khoản"""
    gross, tiered, prev = [], [], {}
    for j, w in enumerate(seq):
        d = dates[j]; _, mask, syms = cd.snapshot(df, d); tc = tier_cost(d, syms, mask.numpy())
        trade = {s: abs(w.get(s, 0) - prev.get(s, 0)) for s in set(w) | set(prev)}
        tiered.append(sum(v * tc.get(s, 0.005) for s, v in trade.items())); gross.append(R_of(j, w)); prev = w
    return np.array(gross), np.array(tiered)
def model_seq(bt):
    return [{s: float(x) for s, x in zip(sy, w) if x > 0} for w, sy in zip(bt['weights'], bt['symbols'])]
def hold_ret(dates):
    def f_(j, w):
        Rr = C.train_mod.compute_holding_return(df, dates[j], dates[j + 1], list(w)); return float(np.dot(list(w.values()), Rr))
    return f_
costs = {}
for per, dates, bts, B, rule in (('test', TEST, [o['test'] for o in ppo], bl, ld['test_A250']['all']),
                                 ('wf', WF, None, bw, ld['wf_A250']['all'])):
    out = {}
    def flat(rets, turns):
        net, turn = np.array(rets), np.array(turns); g = net + 0.001 * turn
        return {c: C.perf_metrics(g - c / 1e4 * turn, 12)['sharpe'] for c in (10, 20, 30, 50)}
    if per == 'test':
        mods = [(o['returns'], o['turnover'], model_seq(o)) for o in bts]
    else:
        mods = []
        for ys in wfp.values():
            seq = sum((model_seq(ys[y]) for y in YEARS), [])
            mods.append((chain(ys), chain(ys, 'turnover'), seq))
    fl = [flat(r, t) for r, t, _ in mods]
    out['model'] = {c: float(np.mean([x[c] for x in fl])) for c in (10, 20, 30, 50)}
    tiers = []
    for r, t, seq in mods:
        g = np.array(r) + 0.001 * np.array(t)
        _, tc = recost_weights(seq, dates, lambda j, w: 0.0)
        tiers.append(C.perf_metrics(g - tc, 12)['sharpe'])
    out['model']['tier'] = float(np.mean(tiers))
    for nm in ('Equal-weight', 'Mean-Variance', 'Min-Variance', 'Momentum top-20'):
        out[nm] = flat(B[nm]['returns'], B[nm]['turnover'])
    out['rule'] = flat(rule['returns'], rule['turnover'])
    seq = [{s: 1.0 / len(h) for s in h} for h in rule['held']]
    g = np.array(rule['returns']) + 0.001 * np.array(rule['turnover'])
    _, tc = recost_weights(seq, dates, lambda j, w: 0.0); out['rule']['tier'] = C.perf_metrics(g - tc, 12)['sharpe']
    seq = []
    for d in dates[:-1]:
        _, mask, syms = cd.snapshot(df, d); el = [s for s, m in zip(syms, mask.numpy()) if m]; seq.append({s: 1.0 / len(el) for s in el})
    g = np.array(B['Equal-weight']['returns']) + 0.001 * np.array(B['Equal-weight']['turnover'])
    _, tc = recost_weights(seq, dates, lambda j, w: 0.0); out['Equal-weight']['tier'] = C.perf_metrics(g - tc, 12)['sharpe']
    costs[per] = out; print(per, json.dumps(out, default=float), flush=True)
S['costs'] = costs
labs = {'model': '\\textbf{Mô hình đề xuất}', 'rule': 'Quy tắc bậc thấp', 'Equal-weight': 'Phân bổ đều (1/N)',
        'Mean-Variance': 'Trung bình--phương sai', 'Min-Variance': 'Phương sai nhỏ nhất', 'Momentum top-20': 'Động lượng top-20'}
tb = ("\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n & \\multicolumn{4}{c}{2024--2025} & \\multicolumn{4}{c}{2020--2025} \\\\\n"
      "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\nChiến lược & 10 bps & 30 bps & 50 bps & Theo thanh khoản & 10 bps & 30 bps & 50 bps & Theo thanh khoản \\\\\n\\midrule\n")
for k, lab in labs.items():
    row = [lab]
    for per in ('test', 'wf'):
        v = costs[per][k]; row += [f(v[10]), f(v[30]), f(v[50]), f(v['tier']) if 'tier' in v else '--']
    tb += " & ".join(row) + " \\\\\n"
tb += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(TAB, 'costs.tex'), 'w').write(tb)
json.dump(S, open(os.path.join(R, 'robustness2.json'), 'w'), indent=1, default=float)
print('done')
