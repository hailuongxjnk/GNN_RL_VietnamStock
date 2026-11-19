"""
Các đối chứng không học bổ sung (tất định, không phụ thuộc hạt giống), cùng quy ước với baselines.py:
quyết định tại d chỉ dùng thông tin trước d, nắm giữ trong (d, d_next], phí 10 bps × luân chuyển.

  (1) 'Bậc thấp top-20'  — quy tắc không học trên đồ thị dài hạn A250: 20 mã đủ điều kiện có ít cạnh nhất
                           (ít liên kết tương quan nhất), tỷ trọng đều. Đây là cách diễn đạt bằng một dòng
                           hành vi mà mô hình đã học được (Mục 4.5), nên dùng để kiểm tra PPO có thêm gì không.
  (2) 'Vốn hoá (tự dựng)' — danh mục trọng số vốn hoá trên các mã đủ điều kiện, mô phỏng VN-Index.
      Vốn hoá lấy từ Bloomberg (Total_MKC.xlsx, cập nhật 6 tháng/lần), ước lượng tại d bằng vốn hoá ở kỳ
      công bố gần nhất trước d nhân với biến động giá từ kỳ đó đến d.
  (3) 'Vốn hoá, bỏ họ Vingroup' — như (2) nhưng loại VIC, VHM, VRE, VPL. Năm 2025 riêng nhóm này chiếm khoảng
      14–20% vốn hoá và đóng góp phần lớn mức tăng của VN-Index.
  (4) 'Phân bổ đều, bỏ họ Vingroup' — để kiểm tra việc bỏ nhóm này không làm thay đổi mốc phân bổ đều.

Chạy cho giai đoạn kiểm tra chính 2024–2025 và cho chuỗi cuốn chiếu 01/2020–12/2025.
Kết quả: results/baselines_noRL.pkl
"""
import os, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import common as C

COST, TOP_K = 0.001, 20
VIN = ['VIC', 'VHM', 'VRE', 'VPL']


def load_mcap():
    m = pd.read_excel(os.path.join(C.ROOT, 'raw_data', 'Total_MKC.xlsx'), sheet_name='mkc')
    m['Dates'] = pd.to_datetime(m['Dates'])
    return m.set_index('Dates').sort_index()


def run(dates):
    df = C.load_df(); cd = C.CachedData(graph='A250', freq='M')
    mc = load_mcap()
    lr = df.pivot(index='date', columns='symbol', values='log_return').fillna(0.0)
    clr = lr.cumsum()
    names = ['Bậc thấp top-20', 'Vốn hoá (tự dựng)', 'Vốn hoá, bỏ họ Vingroup', 'Phân bổ đều, bỏ họ Vingroup']
    rets = {k: [] for k in names}; turns = {k: [] for k in names}; prev = {k: None for k in names}
    vin_share, vnindex, vin_ret = [], [], []
    for j in range(len(dates) - 1):
        d, dn = dates[j], dates[j + 1]
        X, mask, syms = cd.snapshot(df, d); mask = mask.numpy()
        A = C.adj.mask_adjacency(cd.adjacency(df, d, syms), C.torch.tensor(mask)).numpy()
        R = C.train_mod.compute_holding_return(df, d, dn, syms)
        el = np.where(mask)[0]
        W = {}
        # (1) bậc thấp: số láng giềng trong đồ thị dài hạn (không tính chính nó)
        deg = (A != 0).sum(1) - (np.diag(A) != 0)
        key = np.where(mask, deg, np.inf)
        w = np.zeros(len(syms)); w[np.argsort(key, kind='stable')[:TOP_K]] = 1.0 / TOP_K
        W['Bậc thấp top-20'] = w
        # (2)(3) vốn hoá ước lượng tại d
        snap_dates = mc.index[mc.index < d]; s0 = snap_dates[-1]
        days = clr.index
        i0 = days[days <= s0][-1]; i1 = days[days < d][-1]
        cap = mc.loc[s0].reindex(syms).values.astype(float) * np.exp((clr.loc[i1] - clr.loc[i0]).reindex(syms).fillna(0).values)
        cap = np.where(mask & np.isfinite(cap) & (cap > 0), cap, 0.0)
        isvin = np.array([s in VIN for s in syms])
        W['Vốn hoá (tự dựng)'] = cap / cap.sum()
        c2 = np.where(isvin, 0.0, cap); W['Vốn hoá, bỏ họ Vingroup'] = c2 / c2.sum()
        e2 = (mask & ~isvin).astype(float); W['Phân bổ đều, bỏ họ Vingroup'] = e2 / e2.sum()
        vin_share.append(float(cap[isvin].sum() / cap.sum()))
        vin_ret.append(float((cap * isvin) @ R / max(cap[isvin].sum(), 1e-12)))
        for nm, w in W.items():
            t = float(np.abs(w - (prev[nm] if prev[nm] is not None else 0.0)).sum())
            # chỉ số vốn hoá: phần luân chuyển do giá trôi không phát sinh phí thật; chỉ tính phí cho quy tắc
            c = COST * t if nm in ('Bậc thấp top-20', 'Phân bổ đều, bỏ họ Vingroup') else 0.0
            rets[nm].append(float(w @ R) - c); turns[nm].append(t); prev[nm] = w
        vnindex.append(C.eval_mod.vnindex_holding_return(df, d, dn))
    out = {nm: {'returns': rets[nm], 'turnover': turns[nm], 'metrics': C.perf_metrics(rets[nm], 12)} for nm in names}
    out['_vin_share'] = vin_share; out['_vin_return'] = vin_ret; out['_vnindex'] = vnindex
    out['_dates'] = list(dates[:-1])
    out['_corr_proxy_vni'] = float(np.corrcoef(rets['Vốn hoá (tự dựng)'], vnindex)[0, 1])
    return out


if __name__ == '__main__':
    df = C.load_df()
    allM = sorted(C.CachedData('A250').snap.keys())
    res = {'test': run(C.quiet_split(df, 'M')['test']),
           'wf': run([d for d in allM if d.year >= 2020])}
    for k, r in res.items():
        print('==', k, 'corr(proxy, VN-Index) = %.3f' % r['_corr_proxy_vni'],
              'Vin share first/last = %.3f / %.3f' % (r['_vin_share'][0], r['_vin_share'][-1]))
        for nm, v in r.items():
            if not nm.startswith('_'):
                m = v['metrics']
                print(f"{nm:<30} cum {m['cum_return']*100:6.1f}%  sharpe {m['sharpe']:5.2f}  vol {m['ann_vol']*100:5.1f}  "
                      f"mdd {m['max_drawdown']*100:6.1f}%  turn {np.mean(v['turnover'][1:]):.2f}")
        print('VN-Index', C.perf_metrics(r['_vnindex'], 12))
    pickle.dump(res, open(os.path.join(C.RESULTS, 'baselines_noRL.pkl'), 'wb'))
