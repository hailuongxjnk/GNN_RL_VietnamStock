"""Tổng hợp kết quả các lần chạy → results/summary.json + hình cho Chương 4."""
import glob, io, json, os, pickle, contextlib, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings('ignore')
import common as C
from plotstyle import *

NAMES = {'full': 'Transformer + GAT (đề xuất)', 'lstm_gat': 'LSTM + GAT', 'tf_gcn': 'Transformer + GCN',
         'tf_nograph': 'Transformer, không đồ thị', 'tf_gat_long': 'Transformer + GAT, đồ thị 250 phiên',
         'lstm_nograph': 'LSTM, không đồ thị', 'gru_nograph': 'GRU, không đồ thị'}
runs = {}
for f in sorted(glob.glob(os.path.join(C.RESULTS, '*_seed*.pkl'))):
    r = pickle.load(open(f, 'rb')); runs.setdefault(r['config'], {})[r['seed']] = r
base = pickle.load(open(os.path.join(C.RESULTS, 'baselines.pkl'), 'rb'))
df = C.load_df()
S = {'n_runs': {k: sorted(v) for k, v in runs.items()}}
KEYS = ['sharpe', 'cum_return', 'ann_return', 'ann_vol', 'max_drawdown', 'calmar']

def agg(cfg):
    rs = runs[cfg]; out = {'seeds': sorted(rs)}
    for k in KEYS:
        v = np.array([rs[s]['test']['metrics'][k] for s in sorted(rs)])
        out[k] = {'mean': float(v.mean()), 'std': float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                  'min': float(v.min()), 'max': float(v.max()), 'values': v.round(4).tolist()}
    t = np.array([np.mean(rs[s]['test']['turnover'][1:]) for s in sorted(rs)])
    out['turnover'] = {'mean': float(t.mean()), 'std': float(t.std(ddof=1)) if len(t) > 1 else 0.0}
    v = np.array([rs[s]['val']['metrics']['sharpe'] for s in sorted(rs)])
    out['val_sharpe'] = {'mean': float(v.mean()), 'values': v.round(3).tolist()}
    out['best_iter'] = [int(np.nanargmax([h['val_sharpe'] for h in rs[s]['history']])) for s in sorted(rs)]
    return out

S['configs'] = {c: agg(c) for c in runs}
bm = base['M']
S['baselines_M'] = {k: dict(v['metrics'], turnover=float(np.mean(v['turnover'][1:]))) for k, v in bm.items() if not k.startswith('_')}
S['baselines_W'] = {k: dict(v['metrics'], turnover=float(np.mean(v['turnover'][1:]))) for k, v in base['W'].items() if not k.startswith('_')}
for f in ('M', 'W'):
    for key in ('_random20', '_random20_hold'):
        d = {}
        for k in KEYS:
            v = np.array([m[k] for m in base[f][key]['metrics']])
            d[k] = {'mean': float(v.mean()), 'p5': float(np.percentile(v, 5)), 'p50': float(np.percentile(v, 50)),
                    'p95': float(np.percentile(v, 95))}
        S[f'random{key}_{f}'] = d
rand_sh = np.array([m['sharpe'] for m in bm['_random20_hold']['metrics']])
rand_sh_resh = np.array([m['sharpe'] for m in bm['_random20']['metrics']])

# ── sai số chuẩn Sharpe (Lo 2002) và so sánh cặp với Equal-weight ──
def lo_se(r, ppy):
    r = np.asarray(r); sr1 = r.mean() / r.std(); return float(np.sqrt((1 + 0.5 * sr1 ** 2) / len(r)) * np.sqrt(ppy))
S['lo_se'] = {'Equal-weight': lo_se(bm['Equal-weight']['returns'], 12), 'VN-Index': lo_se(bm['VN-Index']['returns'], 12)}
if 'full' in runs:
    ew = np.array(bm['Equal-weight']['returns']); vn = np.array(bm['VN-Index']['returns'])
    per = []
    for s, r in sorted(runs['full'].items()):
        a = np.array(r['test']['returns']); d = a - ew; dv = a - vn
        per.append({'seed': s, 'sharpe': r['test']['metrics']['sharpe'], 'cum': r['test']['metrics']['cum_return'],
                    'mdd': r['test']['metrics']['max_drawdown'], 'calmar': r['test']['metrics']['calmar'],
                    'val_sharpe': r['val']['metrics']['sharpe'],
                    'best_iter': int(np.nanargmax([h['val_sharpe'] for h in r['history']])),
                    'lo_se': lo_se(a, 12), 'turnover': float(np.mean(r['test']['turnover'][1:])),
                    't_vs_ew': float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
                    't_vs_vn': float(dv.mean() / (dv.std(ddof=1) / np.sqrt(len(dv)))),
                    'pct_random_hold': float((rand_sh < r['test']['metrics']['sharpe']).mean()),
                    'pct_random_reshuffle': float((rand_sh_resh < r['test']['metrics']['sharpe']).mean()),
                    'months_beat_ew': int((d > 0).sum()), 'n': len(d)})
    S['full_per_seed'] = per
    # trung bình các seed như một "quỹ" gộp
    ens = np.mean([r['test']['returns'] for r in runs['full'].values()], axis=0)
    S['full_ensemble'] = C.perf_metrics(ens, 12)

# ── hành vi chính sách ──
def behaviour(cfg):
    out = []
    for s, r in sorted(runs[cfg].items()):
        t = r['test']; W = t['weights']
        cv = [float(a.std() / a.mean()) for a in t['alpha']]
        nz = [w[w > 0] for w in W]
        held = [set(np.array(sy)[w > 0]) for w, sy in zip(W, t['symbols'])]
        stay = [len(held[i] & held[i - 1]) / 20 for i in range(1, len(held))]
        out.append({'seed': s, 'alpha_mean': float(np.mean([a.mean() for a in t['alpha']])), 'alpha_cv': float(np.mean(cv)),
                    'w_min': float(min(x.min() for x in nz)), 'w_max': float(max(x.max() for x in nz)),
                    'w_std': float(np.mean([x.std() for x in nz])), 'n_held': float(np.mean([len(x) for x in nz])),
                    'retention': float(np.mean(stay)), 'held': held})
    return out
if 'full' in runs:
    B = behaviour('full')
    seeds = [b['seed'] for b in B]
    ov = []
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            ov.append(np.mean([len(a & b) / 20 for a, b in zip(B[i]['held'], B[j]['held'])]))
    n_el = np.mean(bm['_n_eligible'])
    S['behaviour'] = {'per_seed': [{k: v for k, v in b.items() if k != 'held'} for b in B],
                      'overlap_between_seeds': float(np.mean(ov)) if ov else None,
                      'overlap_expected_random': float(20 / n_el), 'n_eligible_test': float(n_el)}
    # thanh khoản của các mã được chọn so với toàn bộ mã eligible (phân vị Amihud; cao = kém thanh khoản)
    cd = C.CachedData(); pct = []
    wide = df.pivot(index='date', columns='symbol', values='log_amihud_20d')
    for b, (s, r) in zip(B, sorted(runs['full'].items())):
        for d, held, sy in zip(r['test']['dates'], b['held'], r['test']['symbols']):
            m = cd.snap[pd.Timestamp(d)]['mask'].numpy()
            row = wide.loc[wide.index < d].iloc[-1].reindex(sy)
            el = row[m].dropna(); rk = el.rank(pct=True)
            pct.extend(rk.reindex(list(held)).dropna().tolist())
    S['behaviour']['illiquidity_percentile_of_picks'] = float(np.mean(pct))
    from collections import Counter
    cnt = Counter(x for b in B for h in b['held'] for x in h)
    S['behaviour']['most_held'] = cnt.most_common(15)
    S['behaviour']['n_distinct_held'] = len(cnt)

# ── phân tích phần thưởng DSR: độ lớn theo bước, tỷ lệ bị cắt ──
ew_tr = []
cdx = C.CachedData().install()
dates = C.quiet_split(df, 'M')['train']
for j in range(len(dates) - 1):
    X, mask, sy = cdx.snapshot(df, dates[j]); m = mask.numpy().astype(float)
    R = C.train_mod.compute_holding_return(df, dates[j], dates[j + 1], sy); ew_tr.append(float((m / m.sum()) @ R))
dsr = C.train_mod.DSR(); raw = [dsr.step(r) for r in ew_tr]
raw = np.array(raw)
S['dsr'] = {'n_steps': len(raw), 'frac_clipped': float((np.abs(raw) > 5).mean()),
            'frac_clipped_first20': float((np.abs(raw[:20]) > 5).mean()), 'frac_clipped_after20': float((np.abs(raw[20:]) > 5).mean()),
            'median_abs_first10': float(np.median(np.abs(raw[1:11]))), 'median_abs_last10': float(np.median(np.abs(raw[-10:]))),
            'max_abs': float(np.abs(raw).max())}
fig, ax = plt.subplots(figsize=(7.2, 3.1))
ax.axhspan(-5, 5, color='#f0efec', lw=0)
ax.vlines(range(len(raw)), 0, np.clip(raw, -60, 60), color=C1, lw=1.6)
ax.axhline(5, color=MUTED, lw=0.8, ls='--'); ax.axhline(-5, color=MUTED, lw=0.8, ls='--')
ax.text(len(raw) - 1, 6.5, 'biên độ cắt ±5', ha='right', fontsize=9, color=MUTED)
ax.set_xlabel('Bước quyết định trong quỹ đạo huấn luyện (tháng)'); ax.set_ylabel('$D_t$ trước khi cắt')
fig.savefig(os.path.join(FIG, 'dsr_magnitude.pdf')); plt.close(fig)

# ── HÌNH: đường giá trị tài sản (tháng) ──
dts = bm['_dates']; xs = list(dts) + [C.quiet_split(df, 'M')['test'][-1]]
def eq(r): return np.concatenate([[0], (np.cumprod(1 + np.asarray(r)) - 1) * 100])
if 'full' in runs:
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    E = np.array([eq(r['test']['returns']) for r in runs['full'].values()])
    ax.fill_between(xs, E.min(0), E.max(0), color=C1, alpha=0.15, lw=0)
    for e in E: ax.plot(xs, e, color=C1, lw=0.7, alpha=0.55)
    ax.plot(xs, E.mean(0), color=C1, lw=2.2, label=f'Mô hình đề xuất (trung bình {len(E)} hạt giống; dải: min–max)')
    ax.plot(xs, eq(bm['Equal-weight']['returns']), color=C2, label='Phân bổ đều (1/N)')
    ax.plot(xs, eq(bm['VN-Index']['returns']), color=C3, label='VN-Index')
    ax.plot(xs, eq(bm['Mean-Variance']['returns']), color=C4, label='Mean–Variance (Ledoit–Wolf)')
    ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylabel('Lợi suất tích lũy (%)')
    ax.legend(loc='upper left', fontsize=8.5)
    fig.savefig(os.path.join(FIG, 'equity_monthly.pdf')); plt.close(fig)

    # ── HÌNH: lịch sử huấn luyện ──
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    cols = [C1, C2, C3, C4, C5]
    for (s, r), c in zip(sorted(runs['full'].items()), cols):
        it = range(1, len(r['history']) + 1)
        axes[0].plot(it, [h['reward'] for h in r['history']], color=c, marker='o', ms=3, lw=1.3, label=f'hạt giống {s}')
        axes[1].plot(it, [h['val_sharpe'] for h in r['history']], color=c, marker='o', ms=3, lw=1.3)
    axes[0].set_title('Phần thưởng trung bình (huấn luyện)'); axes[1].set_title('Tỷ số Sharpe trên tập kiểm định')
    for a in axes: a.set_xlabel('Vòng lặp PPO'); a.axhline(0, color=MUTED, lw=0.8)
    axes[0].legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(1.1, -0.28))
    fig.savefig(os.path.join(FIG, 'training_history.pdf')); plt.close(fig)

    # ── HÌNH: phân phối Sharpe của danh mục ngẫu nhiên ──
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.hist(rand_sh, bins=40, color='#cde2fb', edgecolor='white', lw=0.8)
    top = ax.get_ylim()[1]
    for b in S['full_per_seed']:
        ax.axvline(b['sharpe'], color=C1, lw=1.8)
        ax.text(b['sharpe'], top * 1.01, str(b['seed']), ha='center', va='bottom', fontsize=8.5, color=INK)
    ax.axvline(bm['Equal-weight']['metrics']['sharpe'], color=C2, lw=1.8, ls='--')
    ax.axvline(bm['VN-Index']['metrics']['sharpe'], color=C3, lw=1.8, ls='--')
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=C1, lw=1.8), Line2D([], [], color=C2, lw=1.8, ls='--'), Line2D([], [], color=C3, lw=1.8, ls='--'),
               plt.Rectangle((0, 0), 1, 1, color='#cde2fb')],
              ['Mô hình đề xuất (từng hạt giống)', 'Phân bổ đều (1/N)', 'VN-Index', '1.000 danh mục 20 mã ngẫu nhiên'],
              fontsize=8.5, loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.set_xlabel('Tỷ số Sharpe quy năm trên tập kiểm tra'); ax.set_ylabel('Số danh mục')
    fig.savefig(os.path.join(FIG, 'random_portfolios.pdf')); plt.close(fig)

# ── HÌNH: Sharpe kiểm định vs kiểm tra trên mọi lần chạy ──
pts = [(r['val']['metrics']['sharpe'], r['test']['metrics']['sharpe'], c) for c, rs in runs.items() for r in rs.values()]
if len(pts) > 3:
    v, t = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    S['val_test_corr'] = {'pearson': float(np.corrcoef(v, t)[0, 1]), 'n': len(pts),
                          'spearman': float(pd.Series(v).corr(pd.Series(t), method='spearman'))}
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.scatter(v, t, s=34, color=C1, edgecolor='white', lw=1.2, zorder=3)
    ax.axhline(bm['Equal-weight']['metrics']['sharpe'], color=C2, lw=1.2, ls='--')
    ax.text(ax.get_xlim()[1], bm['Equal-weight']['metrics']['sharpe'] + 0.04, 'Phân bổ đều', ha='right', fontsize=8.5, color=INK)
    ax.set_xlabel('Tỷ số Sharpe trên tập kiểm định (2023)'); ax.set_ylabel('Tỷ số Sharpe trên tập kiểm tra (2024–2025)')
    fig.savefig(os.path.join(FIG, 'val_vs_test.pdf')); plt.close(fig)

# ── HÌNH: ablation (Sharpe từng seed) ──
order = [c for c in ['full', 'tf_gat_long', 'tf_gcn', 'tf_nograph', 'lstm_gat', 'lstm_nograph', 'gru_nograph'] if c in runs]
if order:
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(order) + 1.3))
    ax.axvspan(np.percentile(rand_sh, 5), np.percentile(rand_sh, 95), color='#f0efec', lw=0)
    for i, c in enumerate(order[::-1]):
        v = S['configs'][c]['sharpe']['values']
        ax.scatter(v, [i] * len(v), s=30, color=C1, edgecolor='white', lw=1, zorder=3)
        ax.scatter([np.mean(v)], [i], marker='|', s=420, color=INK, lw=2, zorder=4)
    ax.axvline(bm['Equal-weight']['metrics']['sharpe'], color=C2, lw=1.2, ls='--')
    ax.axvline(bm['VN-Index']['metrics']['sharpe'], color=C3, lw=1.2, ls='--')
    ax.set_yticks(range(len(order))); ax.set_yticklabels([NAMES[c] for c in order[::-1]], color=INK)
    ax.set_xlabel('Tỷ số Sharpe trên tập kiểm tra (chấm: từng hạt giống; vạch: trung bình; nền xám: phân vị 5–95% của danh mục ngẫu nhiên;\n nét đứt cam: phân bổ đều; nét đứt xanh lục: VN-Index)', fontsize=8.5)
    ax.grid(axis='y', visible=False)
    fig.savefig(os.path.join(FIG, 'ablation_sharpe.pdf')); plt.close(fig)

# ── THIẾT LẬP TUẦN: checkpoint có sẵn trong repo ──
with contextlib.redirect_stdout(io.StringIO()):
    res = C.train_mod.load_checkpoint(os.path.join(C.ROOT, 'model_weekly.pt'))
    C.CachedData(freq='W').install()
    out = C.eval_mod.evaluate(res, df, split='test', freq='W', top_k=20, w_max=0.10, verbose=False)
a = out['series']['agent']
S['weekly_checkpoint'] = {'agent': C.perf_metrics(a, 52), 'lo_se': lo_se(a, 52),
                          'history': res['history'],
                          'pct_random_hold': float((np.array([m['sharpe'] for m in base['W']['_random20_hold']['metrics']]) < C.perf_metrics(a, 52)['sharpe']).mean())}
bw = base['W']; xw = list(bw['_dates']) + [C.quiet_split(df, 'W')['test'][-1]]
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.plot(xw, eq(a), color=C1, lw=2, label='Mô hình đề xuất (checkpoint tuần, 1 hạt giống)')
ax.plot(xw, eq(bw['Equal-weight']['returns']), color=C2, label='Phân bổ đều (1/N)')
ax.plot(xw, eq(bw['VN-Index']['returns']), color=C3, label='VN-Index')
ax.axhline(0, color=MUTED, lw=0.8); ax.set_ylabel('Lợi suất tích lũy (%)'); ax.legend(loc='upper left', fontsize=8.5)
fig.savefig(os.path.join(FIG, 'equity_weekly.pdf')); plt.close(fig)

json.dump(S, open(os.path.join(C.RESULTS, 'summary.json'), 'w'), indent=1, ensure_ascii=False, default=str)
pr = lambda d: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}
print('RUNS', S['n_runs'])
for c, a in S['configs'].items():
    print(f"{c:<14} n={len(a['seeds'])} sharpe {a['sharpe']['mean']:.2f}±{a['sharpe']['std']:.2f} {a['sharpe']['values']} | cum {a['cum_return']['mean']*100:.1f}±{a['cum_return']['std']*100:.1f} | mdd {a['max_drawdown']['mean']*100:.1f}±{a['max_drawdown']['std']*100:.1f} | calmar {a['calmar']['mean']:.2f}±{a['calmar']['std']:.2f} | vol {a['ann_vol']['mean']*100:.1f} | turn {a['turnover']['mean']:.2f} | best_it {a['best_iter']} | val {a['val_sharpe']['values']}")
for k in ('baselines_M', 'baselines_W'):
    print(k)
    for n, m in S[k].items(): print('  ', n, pr(m))
for k in S:
    if k.startswith('random'): print(k, {a: pr(b) for a, b in S[k].items() if a in ('sharpe', 'cum_return', 'max_drawdown', 'calmar')})
for k in ('lo_se', 'full_ensemble', 'dsr', 'val_test_corr'):
    if k in S: print(k, pr(S[k]) if not isinstance(list(S[k].values())[0], dict) else S[k])
if 'full_per_seed' in S:
    for p in S['full_per_seed']: print(pr(p))
    print({k: v for k, v in S['behaviour'].items() if k != 'per_seed'})
    for p in S['behaviour']['per_seed']: print(pr(p))
print('weekly', pr(S['weekly_checkpoint']['agent']), S['weekly_checkpoint']['lo_se'], S['weekly_checkpoint']['pct_random_hold'])
