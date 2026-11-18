"""Tổng hợp thí nghiệm bổ sung → results/extra.json, bảng LaTeX và hình."""
import glob, json, os, pickle
import numpy as np, pandas as pd
import common as C
from plotstyle import *

R = C.RESULTS; S = {}
f = lambda x, d=2: (f"{x:.{d}f}").replace('.', '{,}').replace('-', '$-$')
TAB = os.path.join(C.ROOT, 'thesis', 'tables')
KEYS = ['sharpe', 'cum_return', 'ann_return', 'ann_vol', 'max_drawdown', 'calmar']
def stats(ms):
    d = {}
    for k in KEYS:
        v = np.array([m[k] for m in ms]); d[k] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                                                  'min': float(v.min()), 'max': float(v.max())}
    return d
def lo_se(r, ppy=12):
    r = np.asarray(r); s1 = r.mean() / r.std(); return float(np.sqrt((1 + 0.5 * s1 ** 2) / len(r)) * np.sqrt(ppy))

# ───────── (C) walk-forward ─────────
wf = {}
for fn in glob.glob(os.path.join(R, 'wf', '*.pkl')):
    r = pickle.load(open(fn, 'rb')); wf.setdefault(r['config'], {}).setdefault(r['seed'], {})[r['year']] = r
bw = pickle.load(open(os.path.join(R, 'baselines_wf.pkl'), 'rb'))['M']
dates = list(bw['_dates']); years = sorted(set(d.year for d in dates))
def by_year(rets, ds):
    s = pd.Series(rets, index=pd.to_datetime(ds)); return {y: s[s.index.year == y].values for y in years}
base_names = {'Equal-weight': 'Phân bổ đều (1/N)', 'VN-Index': 'VN-Index', 'Mean-Variance': 'Trung bình--phương sai',
              'Min-Variance': 'Phương sai nhỏ nhất', 'Momentum top-20': 'Động lượng top-20'}
rows = {}
for b in base_names:
    by = by_year(bw[b]['returns'], dates)
    rows[b] = {'year': {y: C.perf_metrics(by[y], 12) for y in years}, 'all': C.perf_metrics(bw[b]['returns'], 12)}
rand = np.array(bw['_random20_hold']['returns'])
rows['random'] = {'year': {y: {'sharpe_p50': float(np.median([C.perf_metrics(r[[i for i, d in enumerate(dates) if d.year == y]], 12)['sharpe'] for r in rand]))} for y in years},
                  'all_sharpe': np.array([C.perf_metrics(r, 12)['sharpe'] for r in rand])}
for g in ('A250', 'A60'):
    p = os.path.join(R, f'untrained_wf_{g}.pkl')
    if os.path.exists(p):
        u = pickle.load(open(p, 'rb'))
        rows['untrained_' + g] = {'all': stats([C.perf_metrics(o['returns'], 12) for o in u]),
                                  'year': {y: stats([C.perf_metrics(by_year(o['returns'], o['dates'])[y], 12) for o in u]) for y in years}}
model = {}
for cfg, seeds in wf.items():
    st = {}
    for s, ys in seeds.items():
        if sorted(ys) != years: continue
        rets, ds = [], []
        for y in years:
            t = ys[y]['test']; rets += t['returns']; ds += t['dates']
        st[s] = {'returns': rets, 'dates': ds, 'all': C.perf_metrics(rets, 12),
                 'year': {y: ys[y]['test']['metrics'] for y in years},
                 'turnover': float(np.mean([np.mean(ys[y]['test']['turnover'][1:]) for y in years])),
                 'best_iter': {y: int(np.nanargmax([h['val_sharpe'] for h in ys[y]['history']])) for y in years}}
    model[cfg] = st
S['wf_seeds'] = {c: sorted(v) for c, v in model.items()}
ew_all = np.array(bw['Equal-weight']['returns'])
S['wf'] = {}
for cfg, st in model.items():
    if not st: continue
    d = {'all': stats([v['all'] for v in st.values()]),
         'year': {y: stats([v['year'][y] for v in st.values()]) for y in years},
         'turnover': float(np.mean([v['turnover'] for v in st.values()])),
         'lo_se': float(np.mean([lo_se(v['returns']) for v in st.values()])),
         'per_seed': {s: {'sharpe': v['all']['sharpe'], 'cum': v['all']['cum_return'], 'mdd': v['all']['max_drawdown'],
                          'pct_random': float((rows['random']['all_sharpe'] < v['all']['sharpe']).mean()),
                          't_vs_ew': float((lambda dd: dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))))(np.array(v['returns']) - ew_all)),
                          'months_beat_ew': int((np.array(v['returns']) - ew_all > 0).sum()),
                          'best_iter': v['best_iter']} for s, v in st.items()}}
    ens = np.mean([v['returns'] for v in st.values()], axis=0)
    d['ensemble'] = C.perf_metrics(ens, 12)
    S['wf'][cfg] = d
S['wf_baselines'] = {b: {'all': rows[b]['all'], 'year': rows[b]['year']} for b in base_names}
S['wf_random'] = {'all_p5_p50_p95': [float(np.percentile(rows['random']['all_sharpe'], q)) for q in (5, 50, 95)],
                  'year_p50': {y: rows['random']['year'][y]['sharpe_p50'] for y in years}}
for g in ('A250', 'A60'):
    if 'untrained_' + g in rows: S['wf_untrained_' + g] = rows['untrained_' + g]
S['wf_n_months'] = len(dates); S['wf_lo_se_ew'] = lo_se(ew_all)
# so sánh cặp dài hạn - ngắn hạn: chênh lệch Sharpe theo hạt giống
if 'tf_gat_long' in model and 'full' in model:
    common_s = sorted(set(model['tf_gat_long']) & set(model['full']))
    S['wf_long_minus_short'] = [model['tf_gat_long'][s]['all']['sharpe'] - model['full'][s]['all']['sharpe'] for s in common_s]

# bảng theo năm (Sharpe) + tổng
if 'tf_gat_long' in S['wf']:
    L = S['wf']['tf_gat_long']; Sh = S['wf'].get('full')
    head = "\\begin{tabular}{@{}l" + "c" * len(years) + "c@{}}\n\\toprule\nChiến lược & " + " & ".join(map(str, years)) + " & 2020--2025 \\\\\n\\midrule\n"
    body = head
    def r_model(d, lab):
        return lab + " & " + " & ".join(f"{f(d['year'][y]['sharpe']['mean'])}" for y in years) + f" & {f(d['all']['sharpe']['mean'])} $\\pm$ {f(d['all']['sharpe']['std'])} \\\\\n"
    body += r_model(L, '\\textbf{Mô hình, đồ thị dài hạn}')
    if Sh: body += r_model(Sh, 'Mô hình, đồ thị ngắn hạn')
    for g, lab in (('A250', 'Mạng chưa huấn luyện, dài hạn'), ('A60', 'Mạng chưa huấn luyện, ngắn hạn')):
        if 'wf_untrained_' + g in S: body += r_model(S['wf_untrained_' + g], lab)
    body += "\\midrule\n"
    for b, lab in base_names.items():
        body += lab + " & " + " & ".join(f(rows[b]['year'][y]['sharpe']) for y in years) + f" & {f(rows[b]['all']['sharpe'])} \\\\\n"
    body += "20 mã ngẫu nhiên (trung vị) & " + " & ".join(f(S['wf_random']['year_p50'][y]) for y in years) + f" & {f(S['wf_random']['all_p5_p50_p95'][1])} \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(TAB, 'wf_sharpe.tex'), 'w').write(body)
    # bảng tổng 6 năm đầy đủ chỉ số
    body = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nChiến lược & Sharpe & Lợi suất (\\%) & CAGR (\\%) & Biến động (\\%) & MDD (\\%) & Calmar \\\\\n\\midrule\n")
    def full_row(d, lab):
        q = lambda k, s=1, dd=2: f"{f(d[k]['mean']*s, dd)} $\\pm$ {f(d[k]['std']*s, dd)}"
        return f"{lab} & {q('sharpe')} & {q('cum_return',100,1)} & {q('ann_return',100,1)} & {q('ann_vol',100,1)} & {q('max_drawdown',100,1)} & {q('calmar')} \\\\\n"
    body += full_row(L['all'], '\\textbf{Mô hình, đồ thị dài hạn}')
    if Sh: body += full_row(Sh['all'], 'Mô hình, đồ thị ngắn hạn')
    for g, lab in (('A250', 'Mạng chưa huấn luyện, dài hạn'), ('A60', 'Mạng chưa huấn luyện, ngắn hạn')):
        if 'wf_untrained_' + g in S: body += full_row(S['wf_untrained_' + g]['all'], lab)
    body += "\\midrule\n"
    for b, lab in base_names.items():
        m = rows[b]['all']
        body += f"{lab} & {f(m['sharpe'])} & {f(m['cum_return']*100,1)} & {f(m['ann_return']*100,1)} & {f(m['ann_vol']*100,1)} & {f(m['max_drawdown']*100,1)} & {f(m['calmar'])} \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    open(os.path.join(TAB, 'wf_total.tex'), 'w').write(body)
    # hình đường tài sản 2020–2025
    xs = list(dates) + [pd.Timestamp('2025-12-31')]
    eq = lambda r: np.concatenate([[0], (np.cumprod(1 + np.asarray(r)) - 1) * 100])
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    E = np.array([eq(v['returns']) for v in model['tf_gat_long'].values()])
    ax.fill_between(xs, E.min(0), E.max(0), color=C1, alpha=0.15, lw=0)
    ax.plot(xs, E.mean(0), color=C1, lw=2.2, label='Mô hình, đồ thị dài hạn (trung bình; dải: min–max)')
    if 'full' in model and model['full']:
        ax.plot(xs, np.mean([eq(v['returns']) for v in model['full'].values()], axis=0), color=C7, lw=1.6, ls='--', label='Mô hình, đồ thị ngắn hạn')
    ax.plot(xs, eq(bw['Equal-weight']['returns']), color=C2, label='Phân bổ đều (1/N)')
    ax.plot(xs, eq(bw['VN-Index']['returns']), color=C3, label='VN-Index')
    ax.plot(xs, eq(bw['Mean-Variance']['returns']), color=C4, label='Trung bình–phương sai')
    for y in years[1:]: ax.axvline(pd.Timestamp(f'{y}-01-01'), color=GRID, lw=0.8)
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylabel('Lợi suất tích lũy (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8.5)
    fig.savefig(os.path.join(FIG, 'equity_walkforward.pdf')); plt.close(fig)

# ───────── (B) tập thanh khoản ─────────
liq = [pickle.load(open(fn, 'rb')) for fn in sorted(glob.glob(os.path.join(R, 'tf_gat_long_liq_seed*.pkl')))]
bl = pickle.load(open(os.path.join(R, 'baselines_liq.pkl'), 'rb'))['M'] if os.path.exists(os.path.join(R, 'baselines_liq.pkl')) else None
if liq and bl:
    S['liq'] = {'model': stats([r['test']['metrics'] for r in liq]), 'n': len(liq),
                'turnover': float(np.mean([np.mean(r['test']['turnover'][1:]) for r in liq])),
                'baselines': {b: bl[b]['metrics'] for b in base_names},
                'random_p5_p50_p95': [float(np.percentile([m['sharpe'] for m in bl['_random20_hold']['metrics']], q)) for q in (5, 50, 95)],
                'n_eligible': float(np.mean(bl['_n_eligible']))}
    rs = np.array([m['sharpe'] for m in bl['_random20_hold']['metrics']])
    S['liq']['pct_random'] = [float((rs < r['test']['metrics']['sharpe']).mean()) for r in liq]
    S['liq']['sharpe_values'] = [r['test']['metrics']['sharpe'] for r in liq]
    up = os.path.join(R, 'untrained_liq.pkl')
    if os.path.exists(up): S['liq']['untrained'] = stats([o['metrics'] for o in pickle.load(open(up, 'rb'))])
    body = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nChiến lược (tập thanh khoản) & Số lần & Sharpe & Lợi suất (\\%) & Biến động (\\%) & MDD (\\%) & Calmar \\\\\n\\midrule\n")
    q = lambda d, k, s=1, dd=2: f"{f(d[k]['mean']*s, dd)} $\\pm$ {f(d[k]['std']*s, dd)}"
    m = S['liq']['model']
    body += f"\\textbf{{Mô hình, đồ thị dài hạn}} & {len(liq)} & {q(m,'sharpe')} & {q(m,'cum_return',100,1)} & {q(m,'ann_vol',100,1)} & {q(m,'max_drawdown',100,1)} & {q(m,'calmar')} \\\\\n"
    if 'untrained' in S['liq']:
        u = S['liq']['untrained']
        body += f"Mạng chưa huấn luyện & 30 & {q(u,'sharpe')} & {q(u,'cum_return',100,1)} & {q(u,'ann_vol',100,1)} & {q(u,'max_drawdown',100,1)} & {q(u,'calmar')} \\\\\n"
    body += "\\midrule\n"
    for b, lab in base_names.items():
        mm = bl[b]['metrics']
        body += f"{lab} & -- & {f(mm['sharpe'])} & {f(mm['cum_return']*100,1)} & {f(mm['ann_vol']*100,1)} & {f(mm['max_drawdown']*100,1)} & {f(mm['calmar'])} \\\\\n"
    p = S['liq']['random_p5_p50_p95']
    body += f"20 mã ngẫu nhiên, giữ nguyên & 1000 & {f(p[1])} [{f(p[0])}; {f(p[2])}] & -- & -- & -- & -- \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    open(os.path.join(TAB, 'liquid.tex'), 'w').write(body)

json.dump(S, open(os.path.join(R, 'extra.json'), 'w'), indent=1, default=str, ensure_ascii=False)
print(json.dumps({k: S[k] for k in S if k in ('wf_seeds', 'wf_n_months', 'wf_lo_se_ew', 'wf_random', 'wf_long_minus_short')}, indent=0, default=str)[:1500])
for c, d in S.get('wf', {}).items():
    print(c, 'ALL', {k: round(d['all'][k]['mean'], 3) for k in KEYS}, 'std sh', round(d['all']['sharpe']['std'], 3), 'turn', round(d['turnover'], 2), 'se', round(d['lo_se'], 2))
    print('   by year', {y: round(d['year'][y]['sharpe']['mean'], 2) for y in years})
    print('   ens', {k: round(v, 3) for k, v in d['ensemble'].items()})
    for s, p in d['per_seed'].items(): print('   seed', s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()})
for g in ('A250', 'A60'):
    if 'wf_untrained_' + g in S: print('untrained', g, round(S['wf_untrained_' + g]['all']['sharpe']['mean'], 2), round(S['wf_untrained_' + g]['all']['sharpe']['std'], 2), {y: round(S['wf_untrained_' + g]['year'][y]['sharpe']['mean'], 2) for y in years})
for b in base_names: print(b, {k: round(v, 3) for k, v in rows[b]['all'].items()}, {y: round(rows[b]['year'][y]['sharpe'], 2) for y in years})
if 'liq' in S: print('LIQ', json.dumps(S['liq'], default=str)[:1800])
