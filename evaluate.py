"""
Đánh giá (backtest) — so agent với baseline
===========================================
Chạy mô hình đã train trên một tập (mặc định test), tính lợi nhuận hàng tháng
cho 3 chiến lược rồi so các chỉ số:

  - Agent (GNN+RL)     : trọng số do agent quyết định (deterministic).
  - Equal-weight (1/N) : chia đều vốn cho các mã eligible mỗi tháng.
  - VN-Index           : mua-và-giữ thị trường (passive), lấy từ cột vnindex_return.

Chỉ số:
  - Cumulative return : lợi nhuận tích luỹ cả kỳ.
  - Sharpe (annualized): lợi nhuận/độ biến động, quy năm (×√12 vì monthly).
  - Max drawdown      : mức sụt sâu nhất từ đỉnh.
"""

import numpy as np
import pandas as pd
import torch

from temporal_encoder import build_snapshot_tensor, apply_mask_to_embedding, split_dates
from adjacency import build_adjacency, mask_adjacency
from train import compute_holding_return
from rl_agent import project_weights


# ── lợi nhuận VN-Index từ d0 → d1 (cột vnindex_return ở đơn vị %, simple return) ──
def vnindex_holding_return(df, d0, d1):
    sub   = df[(df['date'] > d0) & (df['date'] <= d1)].drop_duplicates('date')
    daily = sub['vnindex_return'] / 100.0          # 1.16 (%) → 0.0116
    return float(np.prod(1.0 + daily) - 1.0)


# ── chỉ số đánh giá từ chuỗi lợi nhuận hàng tháng ──
def metrics(returns, periods_per_year=12):
    r      = np.asarray(returns, dtype=float)
    equity = np.cumprod(1 + r)
    peak   = np.maximum.accumulate(equity)
    mdd    = float(((equity - peak) / peak).min())
    return {
        'cum_return':   float(equity[-1] - 1),
        'sharpe':       float(r.mean() / (r.std() + 1e-12) * np.sqrt(periods_per_year)),
        'max_drawdown': mdd,
    }


# ── backtest chính ──
def evaluate(result, df, split='test', cost_rate=0.001, top_k=None, w_max=1.0, freq='M', verbose=True):
    df = df.copy(); df['date'] = pd.to_datetime(df['date'])
    enc, gnn, agent = result['encoder'], result['gnn'], result['agent']
    enc.eval(); gnn.eval(); agent.eval()
    device = next(enc.parameters()).device

    dates = split_dates(df, freq=freq)[split]

    ret_agent, ret_ew, ret_vn = [], [], []
    w_prev_a = w_prev_e = None

    with torch.no_grad():
        for k in range(len(dates) - 1):
            d, d_next = dates[k], dates[k + 1]
            X, mask, syms = build_snapshot_tensor(df, d)
            if int(mask.sum()) < 2:
                continue
            A = mask_adjacency(build_adjacency(df, d, syms), mask)
            R = compute_holding_return(df, d, d_next, syms)            # (N,)

            # --- Agent ---
            X, A, mask = X.to(device), A.to(device), mask.to(device)
            Z = gnn(apply_mask_to_embedding(enc(X), mask), A)
            w, _, _ = agent.act(Z, mask, deterministic=True)
            wa = project_weights(w.cpu().numpy(), top_k=top_k, w_max=w_max)
            gross_a = float((wa * R).sum())
            turn_a  = float(np.abs(wa - (w_prev_a if w_prev_a is not None else 0.0)).sum())
            ret_agent.append(gross_a - cost_rate * turn_a)
            w_prev_a = wa

            # --- Equal-weight (1/N trên mã eligible) ---
            we = mask.cpu().numpy().astype(float)
            we = we / we.sum()
            gross_e = float((we * R).sum())
            turn_e  = float(np.abs(we - (w_prev_e if w_prev_e is not None else 0.0)).sum())
            ret_ew.append(gross_e - cost_rate * turn_e)
            w_prev_e = we

            # --- VN-Index (passive) ---
            ret_vn.append(vnindex_holding_return(df, d, d_next))

    ppy = 52 if freq == 'W' else 12          # số kỳ/năm để quy đổi Sharpe
    out = {
        'Agent (GNN+RL)':      metrics(ret_agent, ppy),
        'Equal-weight (1/N)':  metrics(ret_ew, ppy),
        'VN-Index':            metrics(ret_vn, ppy),
    }
    series = {'agent': ret_agent, 'equal_weight': ret_ew, 'vnindex': ret_vn,
              'dates': [d for d in dates[:len(ret_agent)]]}

    if verbose:
        print(f"=== Backtest trên tập {split.upper()} ({len(ret_agent)} kỳ) ===\n")
        print(f"{'Chiến lược':<22}{'Cum.Return':>12}{'Sharpe':>10}{'MaxDD':>10}")
        print("-" * 54)
        for name, m in out.items():
            print(f"{name:<22}{m['cum_return']*100:>11.1f}%{m['sharpe']:>10.2f}"
                  f"{m['max_drawdown']*100:>9.1f}%")

    return {'metrics': out, 'series': series}


# ── soi trọng số phân bổ của agent tại 1 decision date ──
def inspect_weights(result, df, decision_date=None, split='test', top=10, top_k=None, w_max=1.0, freq='M'):
    df = df.copy(); df['date'] = pd.to_datetime(df['date'])
    enc, gnn, agent = result['encoder'], result['gnn'], result['agent']
    enc.eval(); gnn.eval(); agent.eval()
    device = next(enc.parameters()).device

    if decision_date is None:                      # mặc định: ngày cuối tập split
        decision_date = split_dates(df, freq=freq)[split][-1]
    decision_date = pd.Timestamp(decision_date)

    X, mask, syms = build_snapshot_tensor(df, decision_date)
    A = mask_adjacency(build_adjacency(df, decision_date, syms), mask)
    X, A, mask = X.to(device), A.to(device), mask.to(device)
    with torch.no_grad():
        Z = gnn(apply_mask_to_embedding(enc(X), mask), A)
        w, _, _ = agent.act(Z, mask, deterministic=True)
    w = project_weights(w.cpu().numpy(), top_k=top_k, w_max=w_max)

    nz    = w[w > 0]                               # trọng số các mã được phân bổ
    eff   = 1.0 / np.sum(w ** 2)                   # số mã "hiệu dụng" (1/HHI)
    order = np.argsort(w)[::-1]
    cum   = np.cumsum(w[order])
    n90   = int(np.searchsorted(cum, 0.90) + 1)    # bao nhiêu mã chiếm 90% vốn

    print(f"=== Trọng số agent tại {decision_date.date()} ===")
    print(f"Số mã eligible         : {int(mask.sum())}")
    print(f"Số mã được phân bổ >0  : {len(nz)}")
    print(f"Weight min / max       : {nz.min()*100:.2f}% / {nz.max()*100:.2f}%")
    print(f"Weight trung bình      : {nz.mean()*100:.2f}%")
    print(f"Số mã hiệu dụng (1/HHI): {eff:.1f}")
    print(f"Số mã chiếm 90% vốn    : {n90}")
    print(f"\nTop {top} phân bổ:")
    for i in order[:top]:
        print(f"  {syms[i]:<7} {w[i]*100:5.2f}%")

    return w, syms


if __name__ == '__main__':
    from train import train_ppo
    df = pd.read_pickle('fake_final_df.pkl')
    res = train_ppo(df, n_iters=3, verbose=False)
    evaluate(res, df, split='test')
