"""
Độ nhạy theo số mã nắm giữ (top_k) và trần tỷ trọng mỗi mã (w_max), KHÔNG huấn luyện lại.

Các lần chạy đã lưu tham số Dirichlet alpha của mô hình tại mỗi ngày quyết định (RE.backtest). Hành động tất định
của tác tử là kỳ vọng Dirichlet, w = alpha / sum(alpha) trên các mã đủ điều kiện; danh mục thực = project_weights(w,
top_k, w_max). Ở đây chỉ thay (top_k, w_max) trong bước chiếu, giữ nguyên mô hình đã huấn luyện với (20, 10%),
rồi tính lại lợi suất sau phí 10 bps. Kết quả cho biết luận văn có phụ thuộc vào lựa chọn 20 mã và trần 10% không;
nó KHÔNG cho biết một mô hình được huấn luyện với ràng buộc khác sẽ đạt bao nhiêu.

Chạy cho: kiểm tra 2024–2025 (10 hạt giống), chuỗi cuốn chiếu 2020–2025 (8 hạt giống), tập thanh khoản (10 hạt giống).
Kết quả: results/sens_weights.json, ../thesis/tables/sens_weights.tex
"""
import os, glob, json, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common as C
T = C.train_mod
R_DIR = C.RESULTS; TAB = os.path.join(C.ROOT, 'thesis', 'tables')
TOPK = [10, 20, 30, 40]; WMAX = [0.05, 0.10, 0.20, 1.0]; COST = 0.001
df = C.load_df()
_R = {}


def hold(d, dn, syms):
    k = (d, dn)
    if k not in _R: _R[k] = T.compute_holding_return(df, d, dn, syms)
    return _R[k]


def series(bt, cd, dates_next, k, wm):
    rets, turns, wp = [], [], None
    for j, (d, a) in enumerate(zip(bt['dates'], bt['alpha'])):
        _, mask, syms = cd.snapshot(df, d); m = mask.numpy()
        w = np.zeros(len(syms)); w[m] = a / a.sum()
        wn = T.project_weights(w, top_k=k, w_max=wm)
        t = float(np.abs(wn - (wp if wp is not None else 0.0)).sum())
        rets.append(float(wn @ hold(d, dates_next[d], syms)) - COST * t); turns.append(t); wp = wn
    return rets, turns


def next_map(cd):
    ks = sorted(cd.snap.keys()); return {ks[i]: ks[i + 1] for i in range(len(ks) - 1)}


def conc(bt):
    """mức tập trung của danh mục gốc (20, 10%): tỷ trọng lớn nhất trung bình, tỷ lệ mã chạm trần"""
    mx = [w.max() for w in bt['weights']]; cap = [np.mean(w[w > 0] >= 0.0999) for w in bt['weights']]
    return float(np.mean(mx)), float(np.mean(cap))


out = {}
for name, graph, pat in [('test', 'A250', 'tf_gat_long_seed*.pkl'), ('liq', 'A250_liq', 'tf_gat_long_liq_seed*.pkl')]:
    cd = C.CachedData(graph=graph); nx = next_map(cd); _R.clear()
    runs = [pickle.load(open(f, 'rb')) for f in sorted(glob.glob(os.path.join(R_DIR, pat)))]
    chk = max(abs(a - b) for o in runs for a, b in zip(series(o['test'], cd, nx, 20, 0.10)[0], o['test']['returns']))
    res = {}
    for k in TOPK:
        for wm in WMAX:
            if k * wm < 1 - 1e-9: continue
            sh = [C.perf_metrics(series(o['test'], cd, nx, k, wm)[0], 12)['sharpe'] for o in runs]
            res[f'{k}_{wm}'] = [float(np.mean(sh)), float(np.std(sh))]
    out[name] = {'grid': res, 'n': len(runs), 'check_max_abs_diff': float(chk),
                 'conc': [float(np.mean([conc(o['test'])[i] for o in runs])) for i in (0, 1)]}
    print(name, len(runs), 'check', chk, out[name]['conc'], flush=True)

# cuốn chiếu: nối 6 năm cho mỗi hạt giống
cd = C.CachedData(graph='A250'); nx = next_map(cd); _R.clear()
wf = {}
for f in glob.glob(os.path.join(R_DIR, 'wf', 'tf_gat_long_y*_seed*.pkl')):
    o = pickle.load(open(f, 'rb')); wf.setdefault(o['seed'], {})[o['year']] = o
wf = {s: v for s, v in wf.items() if len(v) == 6}
res = {}
for k in TOPK:
    for wm in WMAX:
        if k * wm < 1 - 1e-9: continue
        sh = []
        for s, ys in wf.items():
            r = []
            for y in sorted(ys):
                bt = ys[y]['test']; keep = [i for i, d in enumerate(bt['dates']) if d.year == y]
                sub = {'dates': [bt['dates'][i] for i in keep], 'alpha': [bt['alpha'][i] for i in keep]}
                r += series(sub, cd, nx, k, wm)[0]
            sh.append(C.perf_metrics(r, 12)['sharpe'])
        res[f'{k}_{wm}'] = [float(np.mean(sh)), float(np.std(sh))]
out['wf'] = {'grid': res, 'n': len(wf)}
print('wf', len(wf), res['20_0.1'], flush=True)
json.dump(out, open(os.path.join(R_DIR, 'sens_weights.json'), 'w'), indent=1)

f = lambda x: f"{x:.2f}".replace('.', '{,}').replace('-', '$-$')
tb = "\\begin{tabular}{@{}l" + "c" * len(WMAX) * 3 + "@{}}\n\\toprule\n"
tb += " & " + " & ".join(f"\\multicolumn{{{len(WMAX)}}}{{c}}{{{t}}}" for t in
                         ('Kiểm tra 2024--2025', 'Cuốn chiếu 2020--2025', 'Tập thanh khoản')) + " \\\\\n"
tb += "".join(f"\\cmidrule(lr){{{2 + i * len(WMAX)}-{1 + (i + 1) * len(WMAX)}}}" for i in range(3)) + "\n"
lab = lambda wm: 'không trần' if wm >= 1 else f"{int(round(wm * 100))}\\%"
tb += "Số mã & " + " & ".join(lab(wm) for _ in range(3) for wm in WMAX) + " \\\\\n\\midrule\n"
for k in TOPK:
    row = [str(k)]
    for per in ('test', 'wf', 'liq'):
        for wm in WMAX:
            v = out[per]['grid'].get(f'{k}_{wm}')
            s = '--' if v is None else f(v[0])
            row.append(f"\\textbf{{{s}}}" if (k, wm) == (20, 0.10) else s)
    tb += " & ".join(row) + " \\\\\n"
tb += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(TAB, 'sens_weights.tex'), 'w').write(tb)
