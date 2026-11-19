"""
Quy tắc 'bậc thấp top-20' (xem baselines_noRL.py) trên các đồ thị / tập cổ phiếu khác, để kiểm tra
lợi ích có đến từ đồ thị DÀI HẠN hay không và có còn trên tập thanh khoản tốt hay không.
Kết quả: results/lowdeg_variants.pkl
"""
import os, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')
import common as C

def run(graph, dates, k=20, cost=0.001):
    df = C.load_df(); cd = C.CachedData(graph=graph, freq='M')
    rets, turns, prev, held = [], [], None, []
    for j in range(len(dates) - 1):
        d, dn = dates[j], dates[j + 1]
        X, mask, syms = cd.snapshot(df, d); m = mask.numpy()
        A = C.adj.mask_adjacency(cd.adjacency(df, d, syms), mask).numpy()
        R = C.train_mod.compute_holding_return(df, d, dn, syms)
        deg = (A != 0).sum(1) - (np.diag(A) != 0)
        w = np.zeros(len(syms)); w[np.argsort(np.where(m, deg, np.inf), kind='stable')[:k]] = 1.0 / k
        t = float(np.abs(w - (prev if prev is not None else 0.0)).sum())
        rets.append(float(w @ R) - cost * t); turns.append(t); prev = w
        held.append([syms[i] for i in np.where(w > 0)[0]])
    return {'returns': rets, 'turnover': turns, 'held': held, 'metrics': C.perf_metrics(rets, 12)}

if __name__ == '__main__':
    df = C.load_df(); test = C.quiet_split(df, 'M')['test']
    allM = sorted(C.CachedData('A250').snap.keys()); wf = [d for d in allM if d.year >= 2020]
    res = {}
    for name, g, ds in (('test_A250', 'A250', test), ('test_A60', 'A60', test), ('wf_A250', 'A250', wf),
                        ('wf_A60', 'A60', wf), ('liq_A250', 'A250_liq', test), ('liq_A60', 'A60_liq', test)):
        res[name] = run(g, ds); m = res[name]['metrics']
        print(f"{name:<10} sharpe {m['sharpe']:.2f} cum {m['cum_return']*100:.1f} vol {m['ann_vol']*100:.1f} mdd {m['max_drawdown']*100:.1f} turn {np.mean(res[name]['turnover'][1:]):.2f}", flush=True)
    pickle.dump(res, open(os.path.join(C.RESULTS, 'lowdeg_variants.pkl'), 'wb'))
