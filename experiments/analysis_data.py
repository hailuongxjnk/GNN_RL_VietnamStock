"""Thống kê mô tả dữ liệu và đồ thị tương quan → hình cho Chương 3 + results/data_stats.json."""
import json, os
import numpy as np, pandas as pd
import common as C
from plotstyle import *

df = C.load_df(); cd = C.CachedData('A60'); cd250 = C.CachedData('A250')
spM, spW = C.quiet_split(df, 'M'), C.quiet_split(df, 'W')
stats = {}
stats['rows'] = len(df); stats['n_symbols'] = int(df.symbol.nunique())
stats['n_days'] = int(df.date.nunique())
stats['splits'] = {f: {k: [str(v[0].date()), str(v[-1].date()), len(v)] for k, v in sp.items()}
                   for f, sp in (('M', spM), ('W', spW))}
stats['missing'] = df[C.FEATURE_COLS].isna().mean().round(4).to_dict()
elig_rows = df[df['mask'] == 1]
stats['missing_eligible'] = elig_rows[C.FEATURE_COLS].isna().mean().round(4).to_dict()
desc = elig_rows[C.FEATURE_COLS].replace([np.inf, -np.inf], np.nan).describe().T[['mean', 'std', 'min', '50%', 'max']]
stats['describe_eligible'] = desc.round(4).to_dict('index')

# ── số mã eligible theo thời gian (theo decision date tháng) ──
all_m = sorted(sum(spM.values(), []))
rows = []
for d in all_m:
    s = cd.snap[pd.Timestamp(d)]
    m = s['mask'].numpy(); n = int(m.sum())
    r = {'date': d, 'n_elig': n, 'n_all': len(s['symbols'])}
    for key in ('A60', 'A250'):
        A = (s if key == 'A60' else cd250.snap[pd.Timestamp(d)])[key].numpy()[np.ix_(m, m)]
        off = ~np.eye(n, dtype=bool)
        r[key + '_density'] = float((A[off] != 0).mean())
        r[key + '_deg'] = float((A != 0).sum(1).mean() - 1)
        r[key + '_w'] = float(A[off][A[off] != 0].mean()) if (A[off] != 0).any() else 0.0
        r[key + '_isolated'] = float(((A != 0).sum(1) == 1).mean())
    rows.append(r)
g = pd.DataFrame(rows).set_index('date')
g.to_csv(os.path.join(C.RESULTS, 'graph_stats_monthly.csv'))
stats['eligible'] = {'min': int(g.n_elig.min()), 'max': int(g.n_elig.max()),
                     'by_year': g.n_elig.groupby(g.index.year).mean().round(1).to_dict()}
stats['graph'] = {k: {'mean': float(g[k].mean()), 'min': float(g[k].min()), 'max': float(g[k].max()),
                      'argmax': str(g[k].idxmax().date()), 'argmin': str(g[k].idxmin().date())}
                  for k in ['A60_density', 'A250_density', 'A60_deg', 'A250_deg', 'A60_isolated']}
for nm, sp in (('train', spM['train']), ('val', spM['val']), ('test', spM['test'])):
    sub = g.loc[sp[0]:sp[-1]]
    stats['graph'][nm] = {'A60_density': float(sub.A60_density.mean()), 'A250_density': float(sub.A250_density.mean()),
                          'A60_deg': float(sub.A60_deg.mean()), 'n_elig': float(sub.n_elig.mean())}

def shade(ax):
    ax.axvspan(spM['val'][0], spM['test'][0], color='#f0efec', lw=0)
    ax.axvspan(spM['test'][0], g.index[-1], color='#dddcd6', lw=0)
    tr = ax.get_xaxis_transform()
    for x0, x1, t in ((g.index[0], spM['val'][0], 'Huấn luyện'), (spM['val'][0], spM['test'][0], 'KĐ'),
                      (spM['test'][0], g.index[-1], 'Kiểm tra')):
        ax.text(x0 + (x1 - x0) / 2, 1.02, t, transform=tr, va='bottom', ha='center', fontsize=9, color=MUTED)

fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(g.index, g.n_elig, color=C1)
ax.set_ylabel('Số mã đủ điều kiện'); ax.set_ylim(0, 200); shade(ax)
fig.savefig(os.path.join(FIG, 'eligible_universe.pdf')); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.plot(g.index, g.A60_density * 100, color=C1, label='Cửa sổ 60 phiên (mô hình đề xuất)')
ax.plot(g.index, g.A250_density * 100, color=C2, label='Cửa sổ 250 phiên')
ax.set_ylabel('Mật độ đồ thị (%)'); ax.set_ylim(0, 100); shade(ax)
ax.legend(loc='upper left', fontsize=8.5)
fig.savefig(os.path.join(FIG, 'graph_density.pdf')); plt.close(fig)

# ── hai ảnh chụp ma trận kề: ngày thưa nhất và dày nhất (trong 2019–2025) ──
sub = g.loc['2019':]
d_lo, d_hi = sub.A60_density.idxmin(), sub.A60_density.idxmax()
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('seq', ['#ffffff'] + SEQ)
for ax, d in zip(axes, (d_lo, d_hi)):
    s = cd.snap[pd.Timestamp(d)]; m = s['mask'].numpy()
    A = s['A60'].numpy()[np.ix_(m, m)].copy(); np.fill_diagonal(A, 0)
    order = np.argsort(-(A != 0).sum(1)); A = A[np.ix_(order, order)]
    im = ax.imshow(A, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
    ax.set_title(f"{d.date()} — mật độ {g.loc[d, 'A60_density']*100:.0f}%", fontsize=10)
    ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"{int(m.sum())} cổ phiếu (xếp theo bậc)")
cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02); cb.set_label('Trọng số cạnh $|\\rho_{ij}|$'); cb.outline.set_visible(False)
fig.savefig(os.path.join(FIG, 'adjacency_snapshots.pdf')); plt.close(fig)
stats['adj_snap'] = [str(d_lo.date()), str(d_hi.date())]

# ── VN-Index và phân phối lợi suất các mã trong giai đoạn test ──
t0, t1 = spM['test'][0], spM['test'][-1]
sub = df[(df.date > t0) & (df.date <= t1)]
cum = np.exp(sub.groupby('symbol')['log_return'].sum()) - 1
lastmask = df[df.date == df.date[df.date < t0].max()].set_index('symbol')['mask']
cum_el = cum[lastmask.reindex(cum.index).fillna(0) == 1]
vn = df.drop_duplicates('date').set_index('date')['vnindex_return'] / 100
vn_cum = float(np.prod(1 + vn[(vn.index > t0) & (vn.index <= t1)]) - 1)
stats['test_cross_section'] = {'median': float(cum_el.median()), 'mean': float(cum_el.mean()),
    'pct_beat_vnindex': float((cum_el > vn_cum).mean()), 'vnindex': vn_cum, 'n': int(len(cum_el)),
    'top5': cum_el.sort_values(ascending=False).head(5).round(3).to_dict(),
    'pct_negative': float((cum_el < 0).mean())}
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.hist(np.clip(cum_el * 100, -80, 200), bins=40, color=C1, edgecolor='white', linewidth=0.8)
ax.axvline(vn_cum * 100, color=C2, lw=2); ax.axvline(cum_el.median() * 100, color=INK, lw=1.2, ls='--')
ax.text(vn_cum * 100 + 3, ax.get_ylim()[1] * 0.9, f'VN-Index: {vn_cum*100:.0f}%', color=INK, fontsize=9)
ax.text(cum_el.median() * 100 - 3, ax.get_ylim()[1] * 0.9, f'Trung vị: {cum_el.median()*100:.1f}%', color=INK, fontsize=9, ha='right')
ax.set_xlabel('Lợi suất tích lũy 01/2024–12/2025 của từng mã đủ điều kiện (%, cắt tại 200%)'); ax.set_ylabel('Số mã')
fig.savefig(os.path.join(FIG, 'test_cross_section.pdf')); plt.close(fig)

json.dump(stats, open(os.path.join(C.RESULTS, 'data_stats.json'), 'w'), indent=1, ensure_ascii=False, default=str)
print(json.dumps({k: stats[k] for k in ('eligible', 'graph', 'test_cross_section', 'adj_snap', 'splits', 'missing_eligible')}, indent=1, ensure_ascii=False, default=str))
