"""
Kiểm tra độ vững: tính lại lãi/lỗ trên tập kiểm tra với lợi suất ĐÃ LOẠI các cú sụt giả do sự kiện
doanh nghiệp (xem common.clean_corporate_actions). Quyết định của mọi chiến lược giữ nguyên (đã ra trên
dữ liệu gốc); chỉ lợi suất thực hiện thay đổi.   Kết quả: results/clean.json
"""
import glob, json, os, pickle, warnings
import numpy as np, torch
warnings.filterwarnings('ignore')
import common as C
import run_experiments as RE

COST = 0.001
if __name__ == '__main__':
    torch.set_num_threads(1)
    df = C.load_df(); dfc, ev = C.clean_corporate_actions(df)
    tdates = C.quiet_split(df, 'M')['test']
    S = {'n_events': int(len(ev)), 'n_events_eligible': int(ev['mask'].sum()),
         'n_events_test_eligible': int(((ev.date >= tdates[0]) & (ev['mask'] == 1)).sum()),
         'events_by_year': {int(k): int(v) for k, v in ev.groupby(ev.date.dt.year).size().items()}}
    cfgs = {}
    for f in sorted(glob.glob(os.path.join(C.RESULTS, '*_seed*.pkl'))):
        r = pickle.load(open(f, 'rb')); t = r['test']; rets = []; hits = 0
        for k, (w, sy, d) in enumerate(zip(t['weights'], t['symbols'], t['dates'])):
            dn = tdates[tdates.index(d) + 1]
            Rc = C.train_mod.compute_holding_return(dfc, d, dn, sy)
            Rr = C.train_mod.compute_holding_return(df, d, dn, sy)
            hits += int(((np.abs(Rc - Rr) > 1e-9) & (w > 0)).sum())
            rets.append(float(w @ Rc) - COST * t['turnover'][k])
        cfgs.setdefault(r['config'], []).append({'seed': r['seed'], 'raw': t['metrics'], 'clean': C.perf_metrics(rets, 12),
                                                 'n_event_positions': hits})
    S['configs'] = {}
    for c, lst in cfgs.items():
        d = {'per_seed': lst}
        for k in ('sharpe', 'cum_return', 'max_drawdown', 'calmar', 'ann_vol'):
            v = np.array([x['clean'][k] for x in lst])
            d[k] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0}
        d['n_event_positions_mean'] = float(np.mean([x['n_event_positions'] for x in lst]))
        S['configs'][c] = d
    # mạng chưa huấn luyện, đánh giá trên lợi suất đã hiệu chỉnh
    C.CachedData(freq='M').install(); un = []
    for seed in range(1, 31):
        torch.manual_seed(seed); np.random.seed(seed)
        un.append(RE.backtest(C.make_models('transformer', 'gat'), df, 'test', df_eval=dfc)['metrics'])
    sh = np.array([m['sharpe'] for m in un])
    S['untrained_clean'] = {'sharpe_mean': float(sh.mean()), 'sharpe_std': float(sh.std(ddof=1)), 'sharpe_min': float(sh.min()),
                            'sharpe_max': float(sh.max()), 'cum_mean': float(np.mean([m['cum_return'] for m in un]))}
    bp = os.path.join(C.RESULTS, 'baselines_clean.pkl')
    if os.path.exists(bp):
        b = pickle.load(open(bp, 'rb'))['M']
        S['baselines'] = {k: v['metrics'] for k, v in b.items() if not k.startswith('_')}
        for key in ('_random20', '_random20_hold'):
            S['random' + key] = {k: {'p5': float(np.percentile([m[k] for m in b[key]['metrics']], 5)),
                                     'p50': float(np.percentile([m[k] for m in b[key]['metrics']], 50)),
                                     'p95': float(np.percentile([m[k] for m in b[key]['metrics']], 95))}
                                 for k in ('sharpe', 'cum_return', 'max_drawdown', 'calmar')}
        rs = np.array([m['sharpe'] for m in b['_random20_hold']['metrics']])
        for c in S['configs']:
            for x in S['configs'][c]['per_seed']:
                x['pct_random_hold'] = float((rs < x['clean']['sharpe']).mean())
    json.dump(S, open(os.path.join(C.RESULTS, 'clean.json'), 'w'), indent=1, default=str)
    for c, d in S['configs'].items():
        print(f"{c:<14} clean sharpe {d['sharpe']['mean']:.2f}±{d['sharpe']['std']:.2f}  cum {d['cum_return']['mean']*100:.1f}  "
              f"raw→clean per seed {[ (round(x['raw']['sharpe'],2), round(x['clean']['sharpe'],2)) for x in d['per_seed']]}  events held {d['n_event_positions_mean']:.1f}")
    print('untrained clean', S['untrained_clean'])
    if 'baselines' in S:
        for k, m in S['baselines'].items(): print(k, {a: round(b, 3) for a, b in m.items()})
        print(S['random_random20_hold'])
