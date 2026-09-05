"""
Dựng lại final_df từ dữ liệu cache sẵn trong raw_data/ + stock_value/ + stock_bs/ + stock_is/,
tái hiện đúng logic các cell 2,3,5,7,8,9,14 trong Full_code_model.ipynb — nhưng KHÔNG gọi
API Fireant sống (cell 2 gốc dùng API key đã bị xoá / hết hạn). Thay vào đó, lịch giao dịch
(base_date) được lấy trực tiếp từ cột 'date' đã có sẵn trong stock_value/VIC.parquet
(chính là kết quả cell 4 gốc đã crawl từ trước, cùng một nguồn dữ liệu).

Chạy 1 lần, cache ra final_df.pkl để các script train/eval sau dùng lại cho nhanh.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def build_final_df(cache_path: str = None) -> pd.DataFrame:
    if cache_path and os.path.exists(cache_path):
        print(f"Đã có cache, nạp lại từ {cache_path}")
        return pd.read_pickle(cache_path)

    # ── cell 5: giá + chỉ số kỹ thuật từ stock_value/*.parquet ──
    value_dir = os.path.join(HERE, "stock_value")
    df_list = [
        pd.read_parquet(os.path.join(value_dir, f))
        for f in os.listdir(value_dir) if f.endswith(".parquet")
    ]
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"[1/6] stock_value: {len(df_list)} cổ phiếu, {len(merged_df):,} dòng")

    # ── cell 2 (thay thế API sống): lịch giao dịch lấy từ VIC.parquet ──
    vic = merged_df[merged_df['symbol'] == 'VIC'].sort_values('date')
    base_date = pd.to_datetime(vic['date'])
    data_set_date = base_date[base_date >= pd.Timestamp('2016-01-01')]
    print(f"[2/6] base_date: {base_date.min().date()} → {base_date.max().date()} "
          f"({len(base_date)} phiên); data_set_date từ 2016: {len(data_set_date)} phiên")

    # ── cell 3: danh sách cổ phiếu đạt tiêu chí vốn hoá vừa+lớn ──
    market_cap = pd.read_excel(os.path.join(HERE, "raw_data", "Total_MKC.xlsx"), sheet_name='mkc')
    market_cap['Dates'] = pd.to_datetime(market_cap['Dates'])
    market_cap = market_cap.set_index("Dates")

    list_stock = market_cap.reset_index().melt(
        id_vars='Dates', var_name='symbol', value_name='market_cap'
    )
    list_stock = list_stock.rename(columns={'Dates': 'date'})
    list_stock = list_stock.dropna(subset=['market_cap'])
    list_stock = list_stock[['symbol', 'date', 'market_cap']]

    EXCLUDE = ['E1VFVN30', 'FUEVFVND', 'GE2', 'FLC', 'SQC', 'TBH', 'ROS', 'CAB', 'GTN', 'PME', 'BHS']
    list_stock = list_stock[~list_stock['symbol'].isin(EXCLUDE)]
    list_stock['size_category'] = np.where(list_stock['market_cap'] > 5e6, 'check', None)
    print(f"[3/6] list_stock (vốn hoá vừa+lớn, sau loại trừ): "
          f"{list_stock['symbol'].nunique()} mã")

    # ── cell 7: merge P/E, P/B ──
    pe_merged = pd.read_csv(os.path.join(HERE, "raw_data", "pe_pb_result.csv"), parse_dates=['date'])
    merged_df = merged_df.merge(pe_merged, on=['date', 'symbol'], how='left')
    print(f"[4/6] sau merge P/E,P/B: {merged_df.shape}")

    # ── cell 8: mask eligibility (vốn hoá + niêm yết >= 6 tháng / 126 phiên) ──
    unique_dates = sorted(list_stock['date'].unique())
    timeline_df = pd.DataFrame({'start_date': unique_dates})
    timeline_df['end_date'] = timeline_df['start_date'].shift(-1)
    timeline_df['end_date'] = timeline_df['end_date'].fillna(
        timeline_df['start_date'] + pd.Timedelta(days=182)
    )
    list_stock_timemap = pd.merge(list_stock, timeline_df, left_on='date', right_on='start_date', how='left')

    merged_df = merged_df.sort_values('date')
    list_stock_timemap = list_stock_timemap.sort_values('date')

    new_df = pd.merge_asof(
        merged_df,
        list_stock_timemap[['symbol', 'date', 'size_category', 'end_date']],
        on='date', by='symbol', direction='backward',
    )
    out_of_bound = new_df['date'] >= new_df['end_date']
    new_df.loc[out_of_bound, 'size_category'] = pd.NA
    new_df = new_df.drop(columns=['end_date']).sort_values(['symbol', 'date'])

    is_not_nan = new_df['size_category'].notna()
    is_not_nan_6m_ago = new_df.groupby('symbol')['log_return'].shift(126).notna()
    new_df['mask'] = ((is_not_nan) & (is_not_nan_6m_ago)).astype(int)
    new_df = new_df.drop(columns=['size_category'])
    new_df = new_df[new_df['date'].isin(data_set_date)].copy()
    print(f"[5/6] sau mask: {new_df.shape}, mask=1: {(new_df['mask'] == 1).sum():,} dòng, "
          f"{new_df.loc[new_df['mask'] == 1, 'symbol'].nunique()} mã từng eligible")

    # ── cell 9: macro (CPI, bond yield, interbank, VN-Index, USD/VND) ──
    df_cpi = pd.read_excel(os.path.join(HERE, "raw_data", "cpi_yoy.xlsx"))
    df_cpi['date'] = pd.to_datetime(df_cpi['date'])
    filtered_df_cpi = df_cpi[df_cpi['date'].isin(base_date)].copy()

    df_bond = pd.read_excel(os.path.join(HERE, "raw_data", "vietnam_bond_yield.xlsx"))
    df_bond['Date'] = pd.to_datetime(df_bond['Date'])
    filtered_df_bond = df_bond[df_bond['Date'].isin(base_date)].copy()
    filtered_df_bond = filtered_df_bond.drop(columns=["Open", "High", "Low", "Change %"])
    filtered_df_bond.rename(columns={'Price': 'bond_yield', "Date": "date"}, inplace=True)

    df_interbankrate = pd.read_excel(os.path.join(HERE, "raw_data", "interbank_rate.xlsx"))
    df_interbankrate['date'] = pd.to_datetime(df_interbankrate['date'])
    filtered_df_interbankrate = df_interbankrate[df_interbankrate['date'].isin(base_date)].copy()
    filtered_df_interbankrate = filtered_df_interbankrate.drop(columns=["maturity"])

    df_vnindex = pd.read_excel(os.path.join(HERE, "raw_data", "vnindex_daily_returns.xlsx"))
    df_vnindex['date'] = pd.to_datetime(df_vnindex['date'])
    filtered_df_vnindex = df_vnindex[df_vnindex['date'].isin(base_date)].copy()
    filtered_df_vnindex = filtered_df_vnindex.drop(columns=["vnindex"])
    filtered_df_vnindex.rename(columns={"return_pct": "vnindex_return"}, inplace=True)

    df_fx = pd.read_excel(os.path.join(HERE, "raw_data", "usd_vnd_fx_rate.xlsx"))
    df_fx['date'] = pd.to_datetime(df_fx['date'])
    filtered_df_fx = df_fx[df_fx['date'].isin(base_date)].copy()
    filtered_df_fx = filtered_df_fx.drop(columns=["Open", "High", "Low", "Change %"])
    filtered_df_fx.rename(columns={"Close": "usdvnd_fx_rate"}, inplace=True)

    macro_df = (
        pd.concat([filtered_df_cpi, filtered_df_bond, filtered_df_interbankrate,
                   filtered_df_vnindex, filtered_df_fx], axis=0)
        .groupby("date", as_index=False).first()
    )
    macro_df = macro_df.sort_values(by='date').reset_index(drop=True)
    macro_df = macro_df.set_index('date').ffill().reset_index()
    macro_df = macro_df.set_index('date').bfill().reset_index()
    filtered_macro_df = macro_df[macro_df['date'].isin(data_set_date)].reset_index(drop=True)

    new_df = new_df.merge(filtered_macro_df, on='date', how='left')
    print(f"[6/6] sau merge macro: {new_df.shape}")

    # ── cell 14: BCTC (Net income growth, ROE, A/E) ──
    financial_df = pd.read_excel(os.path.join(HERE, "raw_data", "financial_data.xlsx"))
    financial_df_clean = financial_df.dropna(subset=['date'])
    financial_df_clean = financial_df_clean.groupby(['symbol', 'date']).first().reset_index()

    ticker = financial_df_clean['symbol'].unique()
    all_date = pd.date_range(start=base_date.min(), end=pd.Timestamp('2025-12-31'), freq='D')
    idx = pd.MultiIndex.from_product([ticker, all_date], names=['symbol', 'date'])
    financial_df_expanded = (
        financial_df_clean.set_index(['symbol', 'date'])
        .reindex(idx).groupby(level='symbol').ffill().reset_index()
    )

    final_df = new_df.merge(financial_df_expanded, on=['symbol', 'date'], how='left')
    final_df = final_df.drop(columns=['quarter', 'year'], errors='ignore')

    print(f"\nfinal_df: {final_df.shape}  |  {final_df['symbol'].nunique()} mã  |  "
          f"{final_df['date'].min().date()} → {final_df['date'].max().date()}")

    if cache_path:
        final_df.to_pickle(cache_path)
        print(f"Đã lưu cache → {cache_path}")

    return final_df


if __name__ == '__main__':
    build_final_df(cache_path=os.path.join(HERE, 'final_df.pkl'))
