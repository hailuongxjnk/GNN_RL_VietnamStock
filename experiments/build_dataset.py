"""
Dựng lại final_df đúng theo các cell 2, 3, 5, 7, 8, 9, 14 của Full_code_model.ipynb,
nhưng KHÔNG gọi API: lịch giao dịch (base_date) lấy từ stock_value/VIC.parquet
(mọi file parquet đã được reindex theo lịch của VIC khi crawl).

Chạy:  python experiments/build_dataset.py   →  experiments/cache/final_df.pkl
"""
import os
import datetime
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw_data")
OUT = os.path.join(ROOT, "experiments", "cache", "final_df.pkl")

EXCLUDED = ['E1VFVN30', 'FUEVFVND', 'GE2', 'FLC', 'SQC', 'TBH', 'ROS', 'CAB', 'GTN', 'PME', 'BHS']


def _rsi(prices, n=14):
    delta = prices.diff(); gain = delta.clip(lower=0); loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / n, adjust=False).mean(); al = loss.ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al)


def adjust_for_corporate_actions(df, thr=-0.17, market_floor=-4.0, pb_gap=0.10):
    """
    SỬA LỖI DỮ LIỆU: giá trong stock_value chưa điều chỉnh theo cổ tức cổ phiếu / cổ phiếu thưởng / chia tách.
    Nhận diện ngày giao dịch không hưởng quyền (lợi suất < thr, VN-Index không sụp, P/B không giảm tương ứng),
    đặt lợi suất ngày đó về 0, dựng lại chuỗi giá điều chỉnh từ lợi suất tích luỹ và TÍNH LẠI các chỉ báo
    phụ thuộc giá (biến động, RSI, động lượng, tỷ số MA) đúng công thức trong notebook.
    df: bảng dài có cột date, symbol, log_return, PX_TO_BOOK_RATIO, vnindex_return (kể cả năm 2015).
    """
    out = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    pb = out['PX_TO_BOOK_RATIO'].where(out['PX_TO_BOOK_RATIO'] > 0)
    dlpb = np.log(pb).groupby(out['symbol']).diff()
    ev = (out['log_return'] < thr) & (out['vnindex_return'] > market_floor) & \
         (dlpb.isna() | ((dlpb - out['log_return']) > pb_gap))
    events = out.loc[ev, ['date', 'symbol', 'log_return']].assign(dlog_pb=dlpb[ev])
    out.loc[ev, 'log_return'] = 0.0
    parts = []
    for sym, g in out.groupby('symbol', sort=False):
        t = g[g['log_return'].notna()]
        lr = t['log_return']; px = np.exp(lr.cumsum())
        new = pd.DataFrame({
            'volatility_20d': lr.rolling(20).std() * np.sqrt(252),
            'volatility_60d': lr.rolling(60).std() * np.sqrt(252),
            'rsi_14': _rsi(px, 14),
            'momentum_20d': np.log(px / px.shift(20)),
            'momentum_60d': np.log(px / px.shift(60)),
            'ma_ratio': px.rolling(20).mean() / px.rolling(50).mean()}, index=t.index)
        parts.append(new)
    new = pd.concat(parts)
    for c in new.columns:
        keep_nan = out[c].isna()                    # giữ NaN ở những chỗ bản gốc vốn NaN (giai đoạn khởi tạo)
        out.loc[new.index, c] = new[c]
        out.loc[keep_nan, c] = np.nan
    return out, events


def build_final_df(verbose=True, adjust=False):
    start_date = datetime.datetime(2015, 1, 1)
    end_date = datetime.datetime(2025, 12, 31)

    # ── cell 2: lịch giao dịch ──
    vic = pd.read_parquet(os.path.join(ROOT, "stock_value", "VIC.parquet"))
    base_date = pd.to_datetime(vic['date']).sort_values().reset_index(drop=True)
    data_set_date = base_date[base_date >= datetime.datetime(2016, 1, 1)]

    # ── cell 3: vốn hoá, cập nhật 6 tháng/lần ──
    market_cap = pd.read_excel(os.path.join(RAW, 'Total_MKC.xlsx'), sheet_name='mkc')
    market_cap['Dates'] = pd.to_datetime(market_cap['Dates'])
    list_stock = market_cap.melt(id_vars='Dates', var_name='symbol', value_name='market_cap')
    list_stock = list_stock.rename(columns={'Dates': 'date'}).dropna(subset=['market_cap'])
    list_stock = list_stock[['symbol', 'date', 'market_cap']]
    list_stock = list_stock[~list_stock['symbol'].isin(EXCLUDED)]
    list_stock['size_category'] = np.where(list_stock['market_cap'] > 5e+06, 'check', None)

    # ── cell 5: giá + chỉ báo kỹ thuật ──
    sv = os.path.join(ROOT, "stock_value")
    merged_df = pd.concat(
        [pd.read_parquet(os.path.join(sv, f)) for f in sorted(os.listdir(sv)) if f.endswith('.parquet')],
        ignore_index=True)
    merged_df['date'] = pd.to_datetime(merged_df['date'])

    # ── cell 7: P/E, P/B ──
    pe = pd.read_csv(os.path.join(RAW, 'pe_pb_result.csv'), parse_dates=['date'])
    merged_df = merged_df.merge(pe, on=['date', 'symbol'], how='left')

    events = None
    if adjust:
        vni0 = pd.read_excel(os.path.join(RAW, "vnindex_daily_returns.xlsx"))
        vni0['date'] = pd.to_datetime(vni0['date']).astype('datetime64[ns]')
        merged_df['date'] = merged_df['date'].astype('datetime64[ns]')
        tmp = merged_df.merge(vni0[['date', 'return_pct']].rename(columns={'return_pct': 'vnindex_return'}),
                              on='date', how='left')
        tmp['vnindex_return'] = tmp['vnindex_return'].fillna(0.0)
        tmp, events = adjust_for_corporate_actions(tmp)
        merged_df = tmp.drop(columns=['vnindex_return'])

    # ── cell 8: mask eligible ──
    unique_dates = sorted(list_stock['date'].unique())
    timeline_df = pd.DataFrame({'start_date': unique_dates})
    timeline_df['end_date'] = timeline_df['start_date'].shift(-1)
    timeline_df['end_date'] = timeline_df['end_date'].fillna(
        timeline_df['start_date'] + pd.Timedelta(days=182))
    ls_tm = pd.merge(list_stock, timeline_df, left_on='date', right_on='start_date', how='left')
    merged_df = merged_df.sort_values('date')
    ls_tm = ls_tm.sort_values('date')
    for d in (merged_df, ls_tm):
        d['date'] = d['date'].astype('datetime64[ns]')
    new_df = pd.merge_asof(merged_df, ls_tm[['symbol', 'date', 'size_category', 'end_date']],
                           on='date', by='symbol', direction='backward')
    out_of_bound = new_df['date'] >= new_df['end_date']
    new_df.loc[out_of_bound, 'size_category'] = None
    new_df = new_df.drop(columns=['end_date']).sort_values(['symbol', 'date'])
    is_not_nan = new_df['size_category'].notna()
    is_6m = new_df.groupby('symbol')['log_return'].shift(126).notna()
    new_df['mask'] = (is_not_nan & is_6m).astype(int)
    new_df = new_df.drop(columns=['size_category'])
    new_df = new_df[new_df['date'].isin(data_set_date)].copy()

    # ── cell 9: vĩ mô ──
    cpi = pd.read_excel(os.path.join(RAW, "cpi_yoy.xlsx")); cpi['date'] = pd.to_datetime(cpi['date'])
    cpi = cpi[cpi['date'].isin(base_date)]
    bond = pd.read_excel(os.path.join(RAW, "vietnam_bond_yield.xlsx")); bond['Date'] = pd.to_datetime(bond['Date'])
    bond = bond[bond['Date'].isin(base_date)].drop(columns=["Open", "High", "Low", "Change %"])
    bond = bond.rename(columns={'Price': 'bond_yield', "Date": "date"})
    ib = pd.read_excel(os.path.join(RAW, "interbank_rate.xlsx")); ib['date'] = pd.to_datetime(ib['date'])
    ib = ib[ib['date'].isin(base_date)].drop(columns=["maturity"])
    vni = pd.read_excel(os.path.join(RAW, "vnindex_daily_returns.xlsx")); vni['date'] = pd.to_datetime(vni['date'])
    vni = vni[vni['date'].isin(base_date)].drop(columns=["vnindex"]).rename(columns={"return_pct": "vnindex_return"})
    fx = pd.read_excel(os.path.join(RAW, "usd_vnd_fx_rate.xlsx")); fx['date'] = pd.to_datetime(fx['date'])
    fx = fx[fx['date'].isin(base_date)].drop(columns=["Open", "High", "Low", "Change %"])
    fx = fx.rename(columns={"Close": "usdvnd_fx_rate"})
    macro = pd.concat([cpi, bond, ib, vni, fx], axis=0).groupby("date", as_index=False).first()
    macro = macro.sort_values('date').set_index('date').ffill().bfill().reset_index()
    macro = macro[macro['date'].isin(data_set_date)].reset_index(drop=True)
    macro['date'] = macro['date'].astype('datetime64[ns]')
    new_df = new_df.merge(macro, on='date', how='left')

    # ── cell 14: chỉ số tài chính, gắn theo NGÀY CÔNG BỐ rồi ffill ──
    fin = pd.read_excel(os.path.join(RAW, 'financial_data.xlsx'))
    fin = fin.dropna(subset=['date'])
    fin['date'] = pd.to_datetime(fin['date']).astype('datetime64[ns]')
    fin = fin.groupby(['symbol', 'date']).first().reset_index()
    all_date = pd.date_range(start=start_date, end=end_date, freq='D').astype('datetime64[ns]')
    idx = pd.MultiIndex.from_product([fin['symbol'].unique(), all_date], names=['symbol', 'date'])
    fin_x = fin.set_index(['symbol', 'date']).reindex(idx).groupby(level='symbol').ffill().reset_index()
    final_df = new_df.merge(fin_x, on=['symbol', 'date'], how='left').drop(columns=['quarter', 'year'])
    final_df = final_df.sort_values(['symbol', 'date']).reset_index(drop=True)
    if verbose:
        print(final_df.shape, final_df['date'].min().date(), final_df['date'].max().date(),
              final_df['symbol'].nunique(), 'mã')
        print(final_df.columns.tolist())
    if adjust:
        return final_df, events
    return final_df


if __name__ == '__main__':
    import sys
    if 'raw' in sys.argv:                 # dữ liệu gốc như notebook (chưa điều chỉnh) — để đối chiếu bài báo
        build_final_df().to_pickle(OUT.replace('final_df', 'final_df_raw'))
    else:                                 # mặc định: dữ liệu đã điều chỉnh sự kiện doanh nghiệp
        df, events = build_final_df(adjust=True)
        df.to_pickle(OUT)
        events.to_csv(os.path.join(ROOT, 'experiments', 'results', 'corporate_action_events.csv'), index=False)
        print(len(events), 'sự kiện doanh nghiệp đã điều chỉnh →', OUT)
