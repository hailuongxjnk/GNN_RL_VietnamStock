"""
Quy tắc 'bậc thấp' trên đồ thị (không học), đánh giá trên các đồ thị / tập cổ phiếu khác nhau.

LƯU Ý về các mã cùng bậc: trên đồ thị dài hạn, tại phần lớn các ngày quyết định có khoảng 30 mã bậc 0
(không có cạnh nào), nhiều hơn 20. Phiên bản đầu của quy tắc chọn 20 mã đầu tiên theo thứ tự chữ cái trong
nhóm này, nên kết quả phụ thuộc vào thứ tự tên mã. Phiên bản này thay bằng hai cách không phụ thuộc thứ tự:
  'all'    — giữ TOÀN BỘ các mã có bậc không lớn hơn bậc của mã thứ 20 (≥ 20 mã), tỷ trọng đều; tất định.
  'random' — chọn ngẫu nhiên 20 mã trong nhóm cùng bậc ở ranh giới (giữ lại mã cũ nếu vẫn thuộc nhóm để tránh
             luân chuyển giả), lặp 300 lần → phân phối kết quả.
Kết quả: results/lowdeg_variants.pkl
"""
import os, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')
import common as C

K, COST, NRAND = 20, 0.001, 300

def degrees(cd, df, d):
    X, mask, syms = cd.snapshot(df, d); m = mask.numpy()
    A = C.adj.mask_adjacency(cd.adjacency(df, d, syms), mask).numpy()
    deg = ((A != 0).sum(1) - (np.diag(A) != 0)).astype(float)
    return np.where(m, deg, np.inf), m, syms

def run(graph, dates):
    df = C.load_df(); cd = C.CachedData(graph=graph, freq='M'); rng = np.random.default_rng(0)
    out_all = {'returns': [], 'turnover': [], 'n_held': [], 'held': []}; prev = None
    rr = np.zeros((NRAND, len(dates) - 1)); rprev = [None] * NRAND; rpick = [None] * NRAND; rheld = [[] for _ in range(NRAND)]
    n_tied = []
    for j in range(len(dates) - 1):
        d, dn = dates[j], dates[j + 1]
        deg, m, syms = degrees(cd, df, d)
        R = C.train_mod.compute_holding_return(df, d, dn, syms)
        kth = np.sort(deg)[K - 1]
        sure = np.where(deg < kth)[0]; tied = np.where(deg == kth)[0]; n_tied.append(len(tied))
        sel = np.concatenate([sure, tied])
        w = np.zeros(len(syms)); w[sel] = 1.0 / len(sel)
        t = float(np.abs(w - (prev if prev is not None else 0.0)).sum()) if prev is None or len(prev) == len(w) else 1.0
        # các ngày có số mã khác nhau: căn theo tên mã
        if prev is not None:
            pw = dict(zip(out_all['_syms'], prev)); cw = dict(zip(syms, w))
            t = sum(abs(cw.get(s, 0) - pw.get(s, 0)) for s in set(pw) | set(cw))
        out_all['returns'].append(float(w @ R) - COST * t); out_all['turnover'].append(t)
        out_all['n_held'].append(len(sel)); out_all['held'].append([syms[i] for i in sel]); prev = w; out_all['_syms'] = syms
        need = K - len(sure); tied_names = [syms[i] for i in tied]; pos = {s: i for i, s in enumerate(syms)}
        for i in range(NRAND):
            keep = [s for s in (rpick[i] or []) if s in set(tied_names)][:need]
            pool = [s for s in tied_names if s not in set(keep)]
            pick = [syms[k] for k in sure] + keep + list(rng.choice(pool, need - len(keep), replace=False))
            wr = {s: 1.0 / K for s in pick}
            tr = sum(abs(wr.get(s, 0) - (rprev[i] or {}).get(s, 0)) for s in set(wr) | set(rprev[i] or {}))
            rr[i, j] = sum(wr[s] * R[pos[s]] for s in pick) - COST * tr
            rprev[i] = wr; rpick[i] = [s for s in pick if s in set(tied_names)]; rheld[i].append(pick)
    out_all.pop('_syms')
    out_all['metrics'] = C.perf_metrics(out_all['returns'], 12)
    rm = [C.perf_metrics(r, 12) for r in rr]
    return {'all': out_all, 'random': {'returns': rr, 'metrics': rm, 'held': rheld[:20]}, 'n_tied': n_tied}

if __name__ == '__main__':
    df = C.load_df(); test = C.quiet_split(df, 'M')['test']
    allM = sorted(C.CachedData('A250').snap.keys()); wf = [d for d in allM if d.year >= 2020]
    res = {}
    for name, g, ds in (('test_A250', 'A250', test), ('test_A60', 'A60', test), ('wf_A250', 'A250', wf),
                        ('wf_A60', 'A60', wf), ('liq_A250', 'A250_liq', test), ('liq_A60', 'A60_liq', test)):
        r = run(g, ds); res[name] = r; m = r['all']['metrics']; s = np.array([x['sharpe'] for x in r['random']['metrics']])
        print(f"{name:<10} ALL sharpe {m['sharpe']:.2f} cum {m['cum_return']*100:.1f} vol {m['ann_vol']*100:.1f} mdd {m['max_drawdown']*100:.1f} "
              f"n_held {np.mean(r['all']['n_held']):.0f} turn {np.mean(r['all']['turnover'][1:]):.2f} | RANDOM-TIE sharpe p5/p50/p95 "
              f"{np.percentile(s,5):.2f}/{np.median(s):.2f}/{np.percentile(s,95):.2f} | tied {np.mean(r['n_tied']):.0f}", flush=True)
    pickle.dump(res, open(os.path.join(C.RESULTS, 'lowdeg_variants.pkl'), 'wb'))
