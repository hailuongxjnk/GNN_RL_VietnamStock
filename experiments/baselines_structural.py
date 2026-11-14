"""
Đối chứng 'cấu trúc': các quy tắc chọn 20 mã KHÔNG học, dùng đúng thông tin mà mô hình có thể đang khai thác,
để diễn giải kết quả của mô hình trên đồ thị dài hạn.
  - Ít liên kết nhất: 20 mã đủ điều kiện có bậc nhỏ nhất trên đồ thị tương quan dài hạn (250 phiên)
  - Biến động thấp nhất: 20 mã có volatility_60d nhỏ nhất
  - Như trên nhưng chỉ trong nửa thanh khoản tốt hơn (Amihud dưới trung vị)
Đồng thời đo đặc điểm (độ biến động, thanh khoản, tỷ lệ phiên giá đứng yên, bậc đồ thị) của các mã mô hình chọn.
Kết quả: results/structural.json
"""
import glob, json, os, pickle
import numpy as np, pandas as pd
import common as C

COST, K = 0.001, 20
df = C.load_df(); cd = C.CachedData('A250'); cd60 = C.CachedData('A60')
dates = C.quiet_split(df, 'M')['test']
piv = {c: df.pivot(index='date', columns='symbol', values=c) for c in ('volatility_60d', 'log_amihud_20d', 'log_return')}
days = piv['log_return'].index
zero = (piv['log_return'].abs() < 1e-9).astype(float).where(piv['log_return'].notna())
names = ['Ít liên kết nhất (đồ thị dài hạn)', 'Biến động thấp nhất', 'Ít liên kết nhất, nhóm thanh khoản tốt', 'Liên kết nhiều nhất (đồ thị dài hạn)']
rets = {n: [] for n in names}; prev = {n: None for n in names}; turns = {n: [] for n in names}
feat = []        # đặc điểm của toàn bộ mã eligible tại mỗi ngày
for j in range(len(dates) - 1):
    d, dn = dates[j], dates[j + 1]
    s = cd.snap[pd.Timestamp(d)]; m = s['mask'].numpy(); sy = s['symbols']
    A = s['A250'].numpy(); deg = ((A != 0) & m[None, :]).sum(1) - 1
    A6 = cd60.snap[pd.Timestamp(d)]['A60'].numpy(); deg60 = ((A6 != 0) & m[None, :]).sum(1) - 1
    last = days[days < d][-1]
    vol = piv['volatility_60d'].loc[last].reindex(sy).values.astype(float)
    ami = piv['log_amihud_20d'].loc[last].reindex(sy).values.astype(float)
    zr = zero.loc[days < d].iloc[-60:].mean().reindex(sy).values.astype(float)
    R = C.train_mod.compute_holding_return(df, d, dn, sy)
    el = np.where(m)[0]
    F = pd.DataFrame({'date': d, 'symbol': np.array(sy)[el], 'deg250': deg[el], 'deg60': deg60[el], 'vol': vol[el],
                      'amihud': ami[el], 'zero': zr[el]})
    for c in ('deg250', 'deg60', 'vol', 'amihud', 'zero'): F[c + '_pct'] = F[c].rank(pct=True)
    feat.append(F)
    liquid = ami <= np.nanmedian(ami[el])
    def pick(score, ok):
        sc = np.where(ok & ~np.isnan(score), score, np.inf)
        w = np.zeros(len(sy)); w[np.argsort(sc)[:K]] = 1.0 / K; return w
    W = {names[0]: pick(deg + 1e-6 * np.nan_to_num(vol, nan=9), m), names[1]: pick(vol, m),
         names[2]: pick(deg + 1e-6 * np.nan_to_num(vol, nan=9), m & liquid), names[3]: pick(-deg.astype(float), m)}
    for n, w in W.items():
        t = float(np.abs(w - (prev[n] if prev[n] is not None else 0.0)).sum())
        rets[n].append(float(w @ R) - COST * t); turns[n].append(t); prev[n] = w
feat = pd.concat(feat)
S = {'baselines': {n: dict(C.perf_metrics(rets[n], 12), turnover=float(np.mean(turns[n][1:]))) for n in names}}
# đặc điểm các mã được mô hình chọn (phân vị trong tập eligible cùng ngày; 0.5 = giống thị trường)
S['picks'] = {}
fx = feat.set_index(['date', 'symbol'])
for cfg in ('tf_gat_long', 'full'):
    rows = []
    for f in sorted(glob.glob(os.path.join(C.RESULTS, f'{cfg}_seed*.pkl'))):
        r = pickle.load(open(f, 'rb'))['test']
        for d, w, sy in zip(r['dates'], r['weights'], r['symbols']):
            for sname in np.array(sy)[w > 0]:
                if (d, sname) in fx.index: rows.append(fx.loc[(d, sname)])
    if rows:
        P = pd.DataFrame(rows)
        S['picks'][cfg] = {c: float(P[c].mean()) for c in P.columns if c.endswith('_pct') or c in ('deg250', 'vol', 'zero')}
S['universe'] = {c: float(feat[c].mean()) for c in ('deg250', 'vol', 'zero')}
for g in ('A250', 'A60'):
    p = os.path.join(C.RESULTS, f'untrained_{g}.pkl')
    if os.path.exists(p):
        rows = [fx.loc[(d, s_)] for o in pickle.load(open(p, 'rb')) for d, h in zip(dates, o['held']) for s_ in h if (d, s_) in fx.index]
        P = pd.DataFrame(rows); S['picks']['untrained_' + g] = {c: float(P[c].mean()) for c in P.columns if c.endswith('_pct')}
json.dump(S, open(os.path.join(C.RESULTS, 'structural.json'), 'w'), indent=1, ensure_ascii=False)
for n, mtr in S['baselines'].items(): print(n, {k: round(v, 3) for k, v in mtr.items()})
for k, v in S['picks'].items(): print(k, {a: round(b, 3) for a, b in v.items()})
print('universe', S['universe'])
