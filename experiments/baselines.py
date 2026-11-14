"""
Các chiến lược đối chứng KHÔNG học (không phụ thuộc seed), cùng quy ước với evaluate.py:
quyết định tại ngày d chỉ dùng dữ liệu < d, nắm giữ trong (d, d_next], phí 10 bps × turnover.

  - Equal-weight trên toàn bộ mã eligible (giống evaluate.py)
  - VN-Index mua và nắm giữ (giống evaluate.py)
  - Minimum-Variance và Mean-Variance (tangency) của Markowitz với hiệp phương sai co rút
    Ledoit–Wolf, long-only, trần 10%/mã
  - Momentum top-20 (đều trọng số)
  - Chọn NGẪU NHIÊN 20 mã eligible, đều trọng số (1000 lần mô phỏng) → phân phối "không kỹ năng"

Kết quả: results/baselines.pkl
"""
import pickle, os, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')
import common as C

COST, TOP_K, W_MAX, COV_WIN = 0.001, 20, 0.10, 250


def optimise(mu, S, kind):
    n = len(mu); w0 = np.full(n, 1.0 / n)
    cons = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    bnds = [(0.0, W_MAX)] * n
    if kind == 'minvar':
        f = lambda w: w @ S @ w * 1e4
        g = lambda w: 2 * S @ w * 1e4
    else:  # tangency: cực đại Sharpe tiên nghiệm
        def f(w):
            return -(mu @ w) / np.sqrt(w @ S @ w + 1e-16)
        g = None
    r = minimize(f, w0, jac=g, bounds=bnds, constraints=cons, method='SLSQP',
                 options={'maxiter': 300, 'ftol': 1e-10})
    w = np.clip(r.x, 0, None)
    return w / w.sum()


def run(freq, clean=False):
    df = C.load_df(); cd = C.CachedData(freq=freq)
    # clean=True: quyết định vẫn dựa trên dữ liệu gốc, chỉ lãi/lỗ thực hiện được tính trên lợi suất đã hiệu chỉnh
    df_eval = C.clean_corporate_actions(df)[0] if clean else df
    dates = C.quiet_split(df, freq)['test']
    ppy = 52 if freq == 'W' else 12
    wide = df.pivot(index='date', columns='symbol', values='log_return')
    mom = df.pivot(index='date', columns='symbol', values='momentum_60d')
    days = wide.index
    rng = np.random.default_rng(0)
    n_rand = 1000
    names = ['Equal-weight', 'VN-Index', 'Min-Variance', 'Mean-Variance', 'Momentum top-20']
    rets = {k: [] for k in names}; turns = {k: [] for k in names}; prev = {k: None for k in names}
    rand_rets = np.zeros((n_rand, len(dates) - 1)); rand_prev = [None] * n_rand
    hold_rets = np.zeros((n_rand, len(dates) - 1)); hold_prev = [None] * n_rand
    hold_pick = [None] * n_rand          # tên 20 mã đang giữ của từng danh mục ngẫu nhiên
    n_elig, used = [], []
    for j in range(len(dates) - 1):
        d, dn = dates[j], dates[j + 1]
        X, mask, syms = cd.snapshot(df, d)
        mask = mask.numpy()
        R = C.train_mod.compute_holding_return(df_eval, d, dn, syms)
        el = np.where(mask)[0]; n_elig.append(len(el)); used.append(d)
        hist = wide.loc[days < d, syms].iloc[-COV_WIN:, el].fillna(0.0).values
        mu = hist.mean(0)
        S = LedoitWolf().fit(hist).covariance_
        W = {}
        w = np.zeros(len(syms)); w[el] = 1.0 / len(el); W['Equal-weight'] = w
        for kind, nm in (('minvar', 'Min-Variance'), ('tangency', 'Mean-Variance')):
            w = np.zeros(len(syms)); w[el] = optimise(mu, S, kind); W[nm] = w
        m = mom.loc[days < d, syms].iloc[-1].values.astype(float)
        m = np.where(mask & ~np.isnan(m), m, -np.inf)
        w = np.zeros(len(syms)); w[np.argsort(m)[::-1][:TOP_K]] = 1.0 / TOP_K; W['Momentum top-20'] = w
        for nm, w in W.items():
            t = float(np.abs(w - (prev[nm] if prev[nm] is not None else 0.0)).sum())
            rets[nm].append(float(w @ R) - COST * t); turns[nm].append(t); prev[nm] = w
        rets['VN-Index'].append(C.eval_mod.vnindex_holding_return(df, d, dn)); turns['VN-Index'].append(0.0)
        for i in range(n_rand):
            w = np.zeros(len(syms)); w[rng.choice(el, TOP_K, replace=False)] = 1.0 / TOP_K
            t = float(np.abs(w - (rand_prev[i] if rand_prev[i] is not None else 0.0)).sum())
            rand_rets[i, j] = float(w @ R) - COST * t; rand_prev[i] = w
        # biến thể "giữ nguyên": chọn 20 mã một lần, chỉ thay mã khi mã cũ hết eligible
        el_names = [syms[k] for k in el]; pos = {s_: k for k, s_ in enumerate(syms)}
        for i in range(n_rand):
            keep = [s_ for s_ in (hold_pick[i] or []) if s_ in set(el_names)]
            pool = [s_ for s_ in el_names if s_ not in set(keep)]
            keep += list(rng.choice(pool, TOP_K - len(keep), replace=False))
            hold_pick[i] = keep
            w = pd.Series(0.0, index=syms); w[keep] = 1.0 / TOP_K
            wp = hold_prev[i].reindex(syms).fillna(0.0) if hold_prev[i] is not None else 0.0
            t = float(np.abs(w - wp).sum())
            hold_rets[i, j] = float(w.values @ R) - COST * t; hold_prev[i] = w
    out = {nm: {'returns': rets[nm], 'turnover': turns[nm], 'metrics': C.perf_metrics(rets[nm], ppy)}
           for nm in names}
    out['_random20'] = {'returns': rand_rets,
                        'metrics': [C.perf_metrics(r, ppy) for r in rand_rets]}
    out['_random20_hold'] = {'returns': hold_rets,
                             'metrics': [C.perf_metrics(r, ppy) for r in hold_rets]}
    out['_dates'] = used; out['_n_eligible'] = n_elig
    return out


if __name__ == '__main__':
    import sys
    clean = 'clean' in sys.argv
    res = {}
    for f in (('M',) if clean else ('M', 'W')):
        res[f] = run(f, clean=clean)
        print('==', f)
        for k, v in res[f].items():
            if not k.startswith('_'):
                m = v['metrics']
                print(f"{k:<18} cum {m['cum_return']*100:6.1f}%  sharpe {m['sharpe']:5.2f}  "
                      f"mdd {m['max_drawdown']*100:6.1f}%  calmar {m['calmar']:5.2f}  "
                      f"turn {np.mean(v['turnover'][1:]):.2f}")
        s = np.array([m['sharpe'] for m in res[f]['_random20']['metrics']])
        print('random-20 Sharpe: mean %.2f  p5 %.2f  p50 %.2f  p95 %.2f' % (s.mean(), *np.percentile(s, [5, 50, 95])))
        s = np.array([m['sharpe'] for m in res[f]['_random20_hold']['metrics']])
        print('random-20 (hold) Sharpe: mean %.2f  p5 %.2f  p50 %.2f  p95 %.2f' % (s.mean(), *np.percentile(s, [5, 50, 95])))
    with open(os.path.join(C.RESULTS, 'baselines_clean.pkl' if clean else 'baselines.pkl'), 'wb') as f:
        pickle.dump(res, f)
