"""
Phân tích QUAN HỆ TƯƠNG QUAN DÀI HẠN giữa các cổ phiếu (trọng tâm của đề tài):
  (1) độ bền của cạnh: tỷ lệ cạnh còn giữ nguyên giữa hai tháng liên tiếp (Jaccard) — đồ thị 60 vs 250 phiên;
  (2) nhóm cổ phiếu có tương quan dài hạn: phân cụm phân cấp trên ma trận tương quan của cả giai đoạn
      huấn luyện 2016–2022 (không dùng dữ liệu kiểm định/kiểm tra);
  (3) tương quan dài hạn ước lượng trên 2016–2022 có còn đúng ở 2024–2025 không (tính ổn định ngoài mẫu).
Kết quả: results/longterm.json + hình.
"""
import json, os, pickle
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
import common as C
from plotstyle import *

df = C.load_df(); sp = C.quiet_split(df, 'M')
S = {}
snaps = {g: pickle.load(open(os.path.join(C.CACHE, f'snap_M_{g}.pkl'), 'rb')) for g in ('A60', 'A250')}
dates = sorted(snaps['A60'])
rows = []
for a, b in zip(dates[:-1], dates[1:]):
    r = {'date': b}
    for g in ('A60', 'A250'):
        sa, sb = snaps[g][a], snaps[g][b]
        ia = {s: i for i, s in enumerate(sa['symbols'])}; ib = {s: i for i, s in enumerate(sb['symbols'])}
        common = [s for s in sa['symbols'] if s in ib and sa['mask'][ia[s]] and sb['mask'][ib[s]]]
        xa = sa[g].numpy()[np.ix_([ia[s] for s in common], [ia[s] for s in common])] != 0
        xb = sb[g].numpy()[np.ix_([ib[s] for s in common], [ib[s] for s in common])] != 0
        iu = np.triu_indices(len(common), 1)
        ea, eb = xa[iu], xb[iu]
        r[g + '_jaccard'] = float((ea & eb).sum() / max((ea | eb).sum(), 1))
        r[g + '_kept'] = float((ea & eb).sum() / max(ea.sum(), 1))
    rows.append(r)
P = pd.DataFrame(rows).set_index('date'); P.to_csv(os.path.join(C.RESULTS, 'edge_persistence.csv'))
S['persistence'] = {c: float(P[c].mean()) for c in P.columns}
S['persistence_min'] = {c: float(P[c].min()) for c in P.columns}
fig, ax = plt.subplots(figsize=(7.2, 3.2))
ax.plot(P.index, P.A250_jaccard * 100, color=C1, label='Đồ thị dài hạn (250 phiên)')
ax.plot(P.index, P.A60_jaccard * 100, color=C2, label='Đồ thị ngắn hạn (60 phiên)')
ax.set_ylim(0, 100); ax.set_ylabel('Độ giống nhau của tập cạnh giữa\nhai tháng liên tiếp (Jaccard, %)')
ax.legend(loc='lower right', fontsize=8.5)
fig.savefig(os.path.join(FIG, 'edge_persistence.pdf')); plt.close(fig)

# ── (2) nhóm cổ phiếu tương quan dài hạn trên giai đoạn huấn luyện ──
wide = df.pivot(index='date', columns='symbol', values='log_return')
tr = wide.loc[:sp['train'][-1]]; te_ = wide.loc[sp['test'][0]:]
ok = [s for s in wide.columns if tr[s].notna().sum() >= 750 and te_[s].notna().sum() >= 300]   # ≥3 năm train, đủ dữ liệu test
Ctr = tr[ok].corr(min_periods=250); Cte = te_[ok].corr(min_periods=120)
S['n_stocks_longterm'] = len(ok)
iu = np.triu_indices(len(ok), 1)
a, b = Ctr.values[iu], Cte.values[iu]; m = ~np.isnan(a) & ~np.isnan(b)
S['oos_stability'] = {'pearson': float(np.corrcoef(a[m], b[m])[0, 1]), 'n_pairs': int(m.sum()),
                      'mean_corr_train': float(a[m].mean()), 'mean_corr_test': float(b[m].mean())}
strong = a[m] >= 0.30
S['oos_stability']['share_strong_pairs_train'] = float(strong.mean())
S['oos_stability']['strong_pairs_still_ge_0.30_in_test'] = float((b[m][strong] >= 0.30).mean())
S['oos_stability']['weak_pairs_ge_0.30_in_test'] = float((b[m][~strong] >= 0.30).mean())
# tương quan ngắn hạn 60 phiên: độ ổn định giữa hai cửa sổ 60 phiên liền kề (trung bình trên test) để so sánh
# Thành phần thị trường chung chi phối tương quan thô, nên nhóm được tìm trên tương quan của PHẦN DƯ sau khi
# loại ảnh hưởng của VN-Index (hồi quy một nhân tố trên giai đoạn huấn luyện), phân cụm Ward.
vn = df.drop_duplicates('date').set_index('date')['vnindex_return'] / 100
def resid(W):
    x = vn.reindex(W.index); out = {}
    for s_ in W.columns:
        y = W[s_]; k = y.notna() & x.notna()
        beta = np.cov(y[k], x[k])[0, 1] / x[k].var()
        out[s_] = y - beta * x
    return pd.DataFrame(out)
Rtr = resid(tr[ok]).corr(min_periods=250); Rte = resid(te_[ok]).corr(min_periods=120)
D = np.sqrt(np.clip(2 * (1 - Rtr.fillna(0).values), 0, None)); np.fill_diagonal(D, 0); D = (D + D.T) / 2
Z = linkage(squareform(D, checks=False), method='ward')
lab = fcluster(Z, t=14, criterion='maxclust')
order = leaves_list(Z)
groups = {}
for s_, l in zip(ok, lab): groups.setdefault(int(l), []).append(s_)
G = []
for l, mem in groups.items():
    if len(mem) < 3: continue
    f = lambda Cm: float(np.nanmean(Cm.loc[mem, mem].values[np.triu_indices(len(mem), 1)]))
    G.append({'size': len(mem), 'members': mem, 'raw_corr_train': f(Ctr), 'raw_corr_test': f(Cte),
              'resid_corr_train': f(Rtr), 'resid_corr_test': f(Rte)})
G.sort(key=lambda g: -g['resid_corr_train'])
S['groups'] = G
ra, rb = Rtr.values[iu], Rte.values[iu]; mm = ~np.isnan(ra) & ~np.isnan(rb)
S['resid_oos_pearson'] = float(np.corrcoef(ra[mm], rb[mm])[0, 1])
S['resid_mean_all_pairs'] = {'train': float(ra[mm].mean()), 'test': float(rb[mm].mean())}
fig, ax = plt.subplots(figsize=(5.4, 4.6))
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('seq', ['#ffffff'] + SEQ)
M = Ctr.values[np.ix_(order, order)].copy(); np.fill_diagonal(M, np.nan)
im = ax.imshow(np.clip(M, 0, 0.7), cmap=cmap, vmin=0, vmax=0.7, interpolation='nearest')
ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel(f'{len(ok)} cổ phiếu, sắp theo phân cụm phân cấp')
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03); cb.set_label('Tương quan lợi suất 2016–2022'); cb.outline.set_visible(False)
fig.savefig(os.path.join(FIG, 'longterm_corr_heatmap.pdf')); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.8))
idx = np.random.default_rng(0).choice(m.sum(), min(4000, m.sum()), replace=False)
ax.scatter(a[m][idx], b[m][idx], s=5, color=C1, alpha=0.35, lw=0)
lim = [-0.2, 0.9]; ax.plot(lim, lim, color=MUTED, lw=0.8, ls='--'); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('Tương quan 2016–2022 (huấn luyện)'); ax.set_ylabel('Tương quan 2024–2025 (kiểm tra)')
fig.savefig(os.path.join(FIG, 'longterm_oos.pdf')); plt.close(fig)

json.dump(S, open(os.path.join(C.RESULTS, 'longterm.json'), 'w'), indent=1, ensure_ascii=False)
print(S['persistence'], S['persistence_min']); print(S['oos_stability'])
print(S['resid_oos_pearson'], S['resid_mean_all_pairs'])
for g in G: print(g['size'], [round(g[k], 2) for k in ('raw_corr_train','raw_corr_test','resid_corr_train','resid_corr_test')], ' '.join(g['members']))
