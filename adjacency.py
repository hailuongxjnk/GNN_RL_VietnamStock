"""
Adjacency Builder cho GNN + RL Portfolio Optimization
=====================================================
Xây ma trận kề A (N, N) — "ai liên quan với ai" — từ tương quan lợi suất.

Nguyên tắc (đã thống nhất trong thiết kế):
  - A xây RIÊNG, chỉ từ log_return — KHÔNG trộn với fundamental.
    Mục đích của A là mô hình hoá đồng-biến-động GIÁ, không phải đồng-biến
    -động chỉ số tài chính.
  - A ĐỘNG theo thời gian (rolling): tại mỗi decision date, tính lại tương
    quan từ cửa sổ lợi suất gần nhất TRƯỚC ngày đó → tránh look-ahead, và
    phản ánh structural break của thị trường VN (COVID, 2022...).
  - A phải khớp ĐÚNG thứ tự `symbols` mà build_snapshot_tensor trả về —
    nếu lệch thứ tự, GNN sẽ ghép nhầm cổ phiếu với nhau.

Quan hệ với phần trước:
    build_snapshot_tensor(...) → X (N,T,F), mask (N,), symbols
    encoder(X)                 → H (N, D)        ← "mỗi cổ phiếu trông ra sao"
    build_adjacency(...)       → A (N, N)        ← "ai liên quan với ai"
    GNN(H, A)                  → Z (N, D')       ← state cho RL

Hàm chính:
  build_adjacency        : df + decision_date + symbols → A (N, N)
  mask_adjacency         : cắt cạnh nối tới cổ phiếu không eligible
  normalize_adjacency    : chuẩn hoá đối xứng cho GCN (Â = D^-1/2 A D^-1/2)
  adjacency_to_edge_index: chuyển A đặc → (edge_index, edge_weight) cho PyG/GAT
"""

import torch
import numpy as np
import pandas as pd

# Tái sử dụng cấu hình từ encoder để mọi cửa sổ khớp nhau
try:
    from temporal_encoder import LOOKBACK
except Exception:
    LOOKBACK = 60


# ─────────────────────────────────────────────────────────────
# HELPER: lấy đúng cửa sổ phiên trước decision_date
#   (logic trùng với build_snapshot_tensor để 2 phần khớp nhau)
# ─────────────────────────────────────────────────────────────

def _get_window_dates(df: pd.DataFrame, decision_date, window: int) -> list:
    decision_date = pd.Timestamp(decision_date)
    all_trading_days = sorted(pd.to_datetime(df['date']).unique())
    past_days = [d for d in all_trading_days if d < decision_date]
    if len(past_days) < window:
        raise ValueError(
            f"Không đủ phiên lịch sử tại {decision_date.date()}: "
            f"cần {window}, có {len(past_days)}"
        )
    return past_days[-window:]   # `window` phiên gần nhất TRƯỚC decision_date


# ─────────────────────────────────────────────────────────────
# BƯỚC CHÍNH: BUILD ADJACENCY TẠI MỘT DECISION DATE
# ─────────────────────────────────────────────────────────────

def build_adjacency(
    df:            pd.DataFrame,
    decision_date: pd.Timestamp,
    symbols:       list,                 # PHẢI là symbols từ build_snapshot_tensor
    corr_window:   int   = LOOKBACK,     # số phiên để ước lượng tương quan
    method:        str   = 'pearson',    # 'pearson' | 'spearman'
    use_abs:       bool  = True,         # dùng |corr| làm độ mạnh cạnh
    threshold:     float = 0.30,         # bỏ cạnh có |corr| < threshold
    min_obs:       int   = 30,           # cần >= min_obs phiên chung, nếu không → cạnh 0
    self_loops:    bool  = True,         # đường chéo = 1 (GAT cần self-loop)
    return_type:   str   = 'tensor',     # 'tensor' | 'numpy'
):
    """
    Tính ma trận kề A (N, N) từ tương quan log_return trong cửa sổ
    `corr_window` phiên ngay TRƯỚC decision_date.

    A[i, j] = độ liên quan giữa cổ phiếu i và j.
      - Cùng tăng/giảm mạnh  → |corr| cao → cạnh mạnh
      - Ít liên quan          → |corr| thấp → bị threshold cắt về 0
      - Thiếu dữ liệu chung   → cạnh 0 (qua min_obs)

    LƯU Ý look-ahead: chỉ dùng phiên < decision_date. Tương quan tại tháng t
    KHÔNG nhìn thấy dữ liệu tương lai.

    Returns
    -------
    A : (N, N) — torch.float32 hoặc np.float32, đối xứng, theo đúng thứ tự `symbols`
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # ── 1. Cửa sổ phiên trước decision_date ──────────────────
    window_dates = _get_window_dates(df, decision_date, corr_window)
    window_set   = set(window_dates)

    # ── 2. Pivot log_return → ma trận rộng (T, N), đúng thứ tự symbols ──
    #     index = ngày, columns = cổ phiếu. NaN = không có dữ liệu phiên đó.
    df_win = df[df['date'].isin(window_set)]
    returns_wide = (
        df_win
        .pivot_table(index='date', columns='symbol', values='log_return')
        .reindex(index=window_dates)     # giữ đúng thứ tự thời gian
        .reindex(columns=symbols)        # giữ ĐÚNG thứ tự N của snapshot tensor
    )

    # ── 3. Tương quan từng cặp (pairwise-complete, có min_periods) ──
    #     pandas tự bỏ NaN theo từng cặp; cặp < min_obs phiên chung → NaN
    C = returns_wide.corr(method=method, min_periods=min_obs)
    C = C.reindex(index=symbols, columns=symbols)   # chắc chắn đủ N×N đúng thứ tự
    C = C.values.astype(np.float32)

    N = len(symbols)
    C = np.nan_to_num(C, nan=0.0)        # cặp không đủ dữ liệu → 0

    # ── 4. Độ mạnh cạnh + threshold để làm thưa đồ thị ───────
    strength = np.abs(C) if use_abs else C
    keep     = np.abs(C) >= threshold    # chỉ giữ cạnh đủ mạnh
    A        = np.where(keep, strength, 0.0).astype(np.float32)

    # ── 5. Đối xứng hoá (dọn sai số float) ───────────────────
    A = 0.5 * (A + A.T)

    # ── 6. Self-loop: đường chéo = 1 ─────────────────────────
    if self_loops:
        np.fill_diagonal(A, 1.0)
    else:
        np.fill_diagonal(A, 0.0)

    if return_type == 'numpy':
        return A
    return torch.tensor(A, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────
# ÁP MASK: cắt cạnh tới/từ cổ phiếu không eligible
#   (song song với apply_mask_to_embedding bên encoder)
# ─────────────────────────────────────────────────────────────

def mask_adjacency(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Giữ cạnh (i, j) chỉ khi CẢ i và j đều eligible.
    A_masked[i,j] = A[i,j] * mask[i] * mask[j]

    Cổ phiếu không eligible bị cô lập hoàn toàn → không lan truyền thông
    tin trong GNN, đồng bộ với việc embedding của nó đã bị zero-out.
    """
    m = mask.float()
    return A * m.unsqueeze(0) * m.unsqueeze(1)


# ─────────────────────────────────────────────────────────────
# CHUẨN HOÁ CHO GCN  (GAT KHÔNG cần — GAT tự học trọng số attention)
# ─────────────────────────────────────────────────────────────

def normalize_adjacency(A: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Chuẩn hoá đối xứng kiểu GCN:  Â = D^-1/2 · A · D^-1/2
    (Giả định A ĐÃ có self-loop sẵn — build_adjacency mặc định self_loops=True.)

    Chỉ dùng cho baseline GCN. Với GAT thì đưa A trực tiếp (làm mask cạnh),
    GAT tự học trọng số nên không cần bước này.
    """
    deg        = A.sum(dim=1)               # bậc của mỗi nút
    d_inv_sqrt = (deg + eps).pow(-0.5)
    return d_inv_sqrt.unsqueeze(1) * A * d_inv_sqrt.unsqueeze(0)


# ─────────────────────────────────────────────────────────────
# CHUYỂN SANG EDGE_INDEX  (cho PyTorch Geometric / GATConv)
# ─────────────────────────────────────────────────────────────

def adjacency_to_edge_index(A: torch.Tensor):
    """
    A đặc (N, N) → (edge_index (2, E), edge_weight (E,)) cho PyG.
    Chỉ lấy các cạnh có trọng số khác 0.
    """
    src_dst     = (A != 0).nonzero(as_tuple=False).t().contiguous()  # (2, E)
    edge_weight = A[src_dst[0], src_dst[1]]
    return src_dst, edge_weight


# ─────────────────────────────────────────────────────────────
# DEMO / SANITY CHECK với dummy data
# ─────────────────────────────────────────────────────────────

def _make_dummy_df(n_symbols=6, n_days=120, seed=0):
    """Tạo long-format df giả có cấu trúc tương quan để test."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range('2020-01-01', periods=n_days)   # ngày giao dịch giả
    syms  = [f"S{k}" for k in range(n_symbols)]

    # 2 "nhân tố" để vài cổ phiếu tương quan với nhau
    f1 = rng.normal(0, 0.01, n_days)
    f2 = rng.normal(0, 0.01, n_days)
    loadings = rng.uniform(-1, 1, (n_symbols, 2))

    rows = []
    for ki, s in enumerate(syms):
        ret = loadings[ki, 0]*f1 + loadings[ki, 1]*f2 + rng.normal(0, 0.005, n_days)
        for d, r in zip(dates, ret):
            rows.append({'date': d, 'symbol': s, 'log_return': r})
    return pd.DataFrame(rows), syms, dates


def demo_with_dummy_data():
    print("=" * 60)
    print("DEMO: build_adjacency với dummy data")
    print("=" * 60)

    df, syms, dates = _make_dummy_df()
    symbols = sorted(syms)
    decision_date = dates[-1] + pd.Timedelta(days=1)   # ngay sau phiên cuối

    A = build_adjacency(df, decision_date, symbols,
                        corr_window=60, threshold=0.3)

    N = len(symbols)
    n_edges  = int((A != 0).sum().item()) - N            # trừ self-loop
    density  = n_edges / (N * (N - 1)) if N > 1 else 0.0
    avg_deg  = (A != 0).sum(dim=1).float().mean().item() - 1  # trừ self-loop

    print(f"\nSymbols        : {symbols}")
    print(f"A shape        : {tuple(A.shape)}")
    print(f"Đối xứng       : {torch.allclose(A, A.t())}")
    print(f"Đường chéo = 1 : {torch.allclose(A.diag(), torch.ones(N))}")
    print(f"Số cạnh        : {n_edges}  (density {density:.1%})")
    print(f"Bậc TB         : {avg_deg:.2f}")
    print(f"\nMa trận A:\n{np.round(A.numpy(), 2)}")

    # Áp mask: giả sử cổ phiếu cuối không eligible
    mask = torch.ones(N, dtype=torch.bool); mask[-1] = False
    A_masked = mask_adjacency(A, mask)
    print(f"\nSau mask (cắt {symbols[-1]}): tổng cạnh tới {symbols[-1]} = "
          f"{A_masked[-1].sum().item():.1f} (kỳ vọng 0)")

    # Chuẩn hoá GCN + edge_index cho PyG
    A_norm = normalize_adjacency(A_masked)
    ei, ew = adjacency_to_edge_index(A_masked)
    print(f"GCN-normalized : shape {tuple(A_norm.shape)}, ok")
    print(f"edge_index     : {tuple(ei.shape)} cạnh, edge_weight {tuple(ew.shape)}")
    print("\n✓ adjacency.py chạy đúng — sẵn sàng ghép vào GNN.")


def demo_with_real_data(df: pd.DataFrame):
    """
    Chạy với final_df thật: build A tại decision date đầu mỗi tập,
    in thống kê đồ thị (density, bậc) để kiểm tra hợp lý.
    Cần đã import split_dates, build_snapshot_tensor từ temporal_encoder.
    """
    from temporal_encoder import split_dates, build_snapshot_tensor

    print("=" * 60)
    print("DEMO: build_adjacency với final_df thật")
    print("=" * 60)

    df = df.copy(); df['date'] = pd.to_datetime(df['date'])
    splits = split_dates(df)

    for split, dates in splits.items():
        d_date = dates[0]
        # Lấy symbols + mask khớp với snapshot tensor
        _, mask, symbols = build_snapshot_tensor(df, d_date)
        A = build_adjacency(df, d_date, symbols)
        A_masked = mask_adjacency(A, mask)

        N        = len(symbols)
        n_elig   = int(mask.sum().item())
        deg      = (A_masked != 0).sum(dim=1).float()
        # bậc trung bình chỉ tính trên nút eligible (trừ self-loop)
        deg_elig = deg[mask].mean().item() - 1 if n_elig > 0 else 0.0

        print(f"\n── {split.upper()} — {d_date.date()} ──")
        print(f"  N / eligible   : {N} / {n_elig}")
        print(f"  A shape        : {tuple(A.shape)}")
        print(f"  Bậc TB (elig)  : {deg_elig:.2f} hàng xóm/cổ phiếu")

    print("\n✓ build_adjacency chạy với data thật OK.")


if __name__ == '__main__':
    demo_with_dummy_data()
