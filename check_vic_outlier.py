import numpy as np
import pandas as pd
from build_final_df import build_final_df

df = build_final_df(cache_path='final_df.pkl')
df['date'] = pd.to_datetime(df['date'])

test_start, test_end = pd.Timestamp('2024-01-01'), pd.Timestamp('2025-12-31')
sub = df[(df['date'] >= test_start) & (df['date'] <= test_end)].copy()

# VN-Index cumulative return over the test period (daily vnindex_return, in %)
vn = sub.drop_duplicates('date').sort_values('date')
vn_cum = (1 + vn['vnindex_return'] / 100.0).prod() - 1
print(f"VN-Index cumulative return {test_start.date()}..{test_end.date()}: {vn_cum*100:.1f}%")

# per-stock cumulative log_return over same window, ranked
stock_cum = (
    sub.groupby('symbol')['log_return']
    .apply(lambda r: np.exp(r.sum()) - 1)
    .sort_values(ascending=False)
)
print(f"\nSo mã có du lieu: {len(stock_cum)}")
print("\nTop 15 mã tăng mạnh nhất giai đoạn test:")
print(stock_cum.head(15).apply(lambda x: f"{x*100:.1f}%"))

print(f"\nVIC return: {stock_cum.get('VIC', float('nan'))*100:.1f}%")
print(f"Median cổ phiếu: {stock_cum.median()*100:.1f}%")
print(f"Mean cổ phiếu: {stock_cum.mean()*100:.1f}%")

# eligibility / market cap weight check for VIC
vic_mask = sub[(sub['symbol'] == 'VIC')]['mask']
print(f"\nVIC eligible (mask=1) trong bao nhieu % ngay cua giai doan test: "
      f"{100*vic_mask.mean():.1f}%")

# how many stocks had return >= vn_cum (i.e., beat the index) vs below
n_beat = (stock_cum >= vn_cum).sum()
print(f"\nSo ma THANG VN-Index trong giai doan nay: {n_beat}/{len(stock_cum)} "
      f"({100*n_beat/len(stock_cum):.1f}%)")
