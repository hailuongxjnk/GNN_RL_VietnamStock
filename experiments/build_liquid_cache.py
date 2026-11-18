"""
Biến thể TẬP CỔ PHIẾU THANH KHOẢN: tại mỗi ngày quyết định (tháng), trong các mã đủ điều kiện, loại thêm
  - 30% mã kém thanh khoản nhất theo Amihud 20 phiên (log_amihud_20d) ở phiên cuối trước ngày quyết định;
  - các mã có hơn 25% số phiên giá đứng yên (lợi suất = 0) trong 60 phiên trước ngày quyết định;
  - các mã thiếu Amihud.
Chỉ dùng dữ liệu trước ngày quyết định. Kết quả: cache/snap_M_A250_liq.pkl (khoá đồ thị 'A250_liq').
"""
import os, pickle, json
import numpy as np, pandas as pd, torch
import common as C

df = C.load_df()
snap = pickle.load(open(os.path.join(C.CACHE, 'snap_M_A250.pkl'), 'rb'))
ami = df.pivot(index='date', columns='symbol', values='log_amihud_20d')
lr = df.pivot(index='date', columns='symbol', values='log_return')
zero = (lr.abs() < 1e-9).astype(float).where(lr.notna())
days = ami.index
out, stat = {}, []
for d, s in snap.items():
    sy = s['symbols']; m = s['mask'].numpy().copy()
    past = days[days < d]
    a = ami.loc[past[-1]].reindex(sy).values.astype(float)
    z = zero.loc[past[-60:]].mean().reindex(sy).values.astype(float)
    el = np.where(m)[0]
    thr = np.nanpercentile(a[el], 70) if np.isfinite(a[el]).any() else np.inf
    liquid = np.isfinite(a) & (a <= thr) & (np.nan_to_num(z, nan=1.0) <= 0.25)
    m2 = m & liquid
    out[d] = {'X': s['X'], 'mask': torch.tensor(m2), 'symbols': sy, 'A250_liq': s['A250']}
    stat.append((d, int(m.sum()), int(m2.sum())))
pickle.dump(out, open(os.path.join(C.CACHE, 'snap_M_A250_liq.pkl'), 'wb'))
st = pd.DataFrame(stat, columns=['date', 'n_elig', 'n_liquid']).set_index('date').sort_index()
st.to_csv(os.path.join(C.RESULTS, 'liquid_universe.csv'))
print(st.describe().round(1)); print(st.loc['2024':].mean().round(1))
