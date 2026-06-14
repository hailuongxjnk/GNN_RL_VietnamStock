"""
Temporal Encoder cho GNN + RL Portfolio Optimization
=====================================================
Input:  DataFrame long-format (final_df)
Output: H shape (N, D) — embedding mỗi cổ phiếu tại mỗi decision date

Hai encoder:
  - LSTMEncoder        : baseline, ổn định với data ít
  - TransformerEncoder : main model, mạnh hơn

Pipeline:
  final_df
    → split train/val/test theo thời gian (70/10/20)
    → tạo decision dates (ngày giao dịch đầu tiên mỗi tháng)
    → build_snapshot_tensor tại mỗi decision date
    → encoder(X) → H (N, D)
    → apply_mask → H_masked (N, D)  ← sẵn sàng cho GNN
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional


# ─────────────────────────────────────────────────────────────
# CẤU HÌNH
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [
    # Giá & kỹ thuật
    'log_return', 'log_volume',
    'volatility_20d', 'volatility_60d',
    'rsi_14', 'momentum_20d', 'momentum_60d',
    'ma_ratio', 'log_amihud_20d',
    # Tài chính doanh nghiệp
    'PE_RATIO', 'PX_TO_BOOK_RATIO',
    'Net income growth', 'ROE', 'A/E',
    # Vĩ mô
    'inflation', 'bond_yield', 'interbank_rate',
    'vnindex_return', 'usdvnd_fx_rate',
]

# Biến vĩ mô: giống nhau ở mọi mã trong cùng ngày → KHÔNG cross-sectional được
MACRO_COLS = [
    'inflation', 'bond_yield', 'interbank_rate',
    'vnindex_return', 'usdvnd_fx_rate',
]

LOOKBACK   = 60    # T — số phiên lịch sử mỗi snapshot
D_MODEL    = 64    # D — chiều embedding đầu ra


# ─────────────────────────────────────────────────────────────
# BƯỚC 1: CHIA TRAIN / VAL / TEST
# ─────────────────────────────────────────────────────────────

def split_dates(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.10,
    freq:        str   = 'M',   # 'M' = tháng, 'W' = tuần (nhiều decision point hơn cho RL)
    # test_ratio tự động = phần còn lại (0.20)
) -> dict:
    """
    Chia toàn bộ timeline thành 3 giai đoạn THEO THỜI GIAN.
    Không shuffle — tuyệt đối không shuffle dữ liệu time series.

    Trả về dict chứa decision dates của từng tập:
      {
        'train': [date1, date2, ...],   # 70% đầu
        'val':   [date1, date2, ...],   # 10% giữa
        'test':  [date1, date2, ...],   # 20% cuối
      }

    Decision date = ngày giao dịch ĐẦU TIÊN của mỗi tháng.
    (không nhất thiết là ngày 1 — có thể là ngày 2, 3, 4...
     tuỳ lịch nghỉ lễ và cuối tuần)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Tất cả ngày giao dịch thực có trong data, sắp xếp tăng dần
    all_trading_days = sorted(df['date'].unique())

    # Tạo decision dates = ngày giao dịch đầu tiên của mỗi kỳ (tháng hoặc tuần)
    trading_day_series = pd.Series(all_trading_days)
    decision_dates = (
        trading_day_series
        .groupby(trading_day_series.dt.to_period(freq))
        .first()
        .tolist()
    )
    # Bỏ các tháng không đủ lookback phiên lịch sử
    # (tháng đầu tiên cần ít nhất LOOKBACK phiên trước nó)
    min_date = all_trading_days[LOOKBACK]   # phiên thứ 60 trở đi mới đủ lookback
    decision_dates = [d for d in decision_dates if d > min_date]

    # Chia theo tỷ lệ thời gian (không phải số lượng)
    # Cắt tại điểm % trên timeline ngày, không phải % số decision dates
    start = all_trading_days[0]
    end   = all_trading_days[-1]
    total_days = (end - start).days

    train_end = start + pd.Timedelta(days=int(total_days * train_ratio))
    val_end   = start + pd.Timedelta(days=int(total_days * (train_ratio + val_ratio)))

    train_dates = [d for d in decision_dates if d <= train_end]
    val_dates   = [d for d in decision_dates if train_end < d <= val_end]
    test_dates  = [d for d in decision_dates if d > val_end]

    print(f"Timeline   : {all_trading_days[0].date()} → {all_trading_days[-1].date()}")
    print(f"Train      : {train_dates[0].date()} → {train_dates[-1].date()}  "
          f"({len(train_dates)} decision dates)")
    print(f"Validation : {val_dates[0].date()} → {val_dates[-1].date()}  "
          f"({len(val_dates)} decision dates)")
    print(f"Test       : {test_dates[0].date()} → {test_dates[-1].date()}  "
          f"({len(test_dates)} decision dates)")

    return {'train': train_dates, 'val': val_dates, 'test': test_dates}


# ─────────────────────────────────────────────────────────────
# BƯỚC 2: BUILD SNAPSHOT TENSOR TẠI MỘT DECISION DATE
# ─────────────────────────────────────────────────────────────

def build_snapshot_tensor(
    df:            pd.DataFrame,
    decision_date: pd.Timestamp,
    feature_cols:  list  = FEATURE_COLS,
    lookback:      int   = LOOKBACK,
    standardize:   bool  = True,    # z-score feature trong cửa sổ (khuyên bật)
    clip:          float = 10.0,    # chặn outlier sau chuẩn hoá
) -> tuple:
    """
    Tại decision_date, cắt cửa sổ lookback phiên TRƯỚC đó.
    Mỗi cổ phiếu được xử lý độc lập (groupby symbol) — không leakage.

    Returns
    -------
    X      : (N, T, F)  float32 — features, NaN+inf fill 0, (tuỳ chọn) đã z-score
    mask   : (N,)       bool    — True nếu eligible tại decision_date
    symbols: list[str]          — tên cổ phiếu theo thứ tự axis N
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    decision_date = pd.Timestamp(decision_date)

    # ── Tìm T=lookback phiên giao dịch ngay trước decision_date ──
    all_trading_days = sorted(df['date'].unique())
    past_days = [d for d in all_trading_days if d < decision_date]

    if len(past_days) < lookback:
        raise ValueError(
            f"Không đủ phiên lịch sử tại {decision_date.date()}: "
            f"cần {lookback}, có {len(past_days)}"
        )
    window_dates = past_days[-lookback:]   # T ngày gần nhất trước decision_date
    window_set   = set(window_dates)

    # ── Lọc data trong cửa sổ ────────────────────────────────
    df_win = df[df['date'].isin(window_set)].copy()

    # ── Danh sách cổ phiếu & index map ───────────────────────
    symbols  = sorted(df_win['symbol'].unique())
    N        = len(symbols)
    T        = lookback
    F        = len(feature_cols)
    date2idx = {d: i for i, d in enumerate(window_dates)}   # giữ thứ tự thời gian

    # ── Xây tensor X: groupby symbol để tránh leakage ────────
    # Khởi tạo toàn NaN — cổ phiếu thiếu phiên nào sẽ giữ NaN
    X = np.full((N, T, F), np.nan, dtype=np.float32)

    for ni, sym in enumerate(symbols):
        # Lọc đúng cổ phiếu này — không chạm sang cổ phiếu khác
        df_sym = df_win[df_win['symbol'] == sym]

        for _, row in df_sym.iterrows():
            ti = date2idx.get(row['date'])
            if ti is None:
                continue
            for fi, col in enumerate(feature_cols):
                if col in row.index and not pd.isna(row[col]):
                    X[ni, ti, fi] = float(row[col])

    # ── Tính mask ─────────────────────────────────────────────
    # Mask gốc từ DataFrame: 1 nếu eligible (vốn hoá + niêm yết >= 6 tháng)
    # Lấy giá trị mask tại ngày giao dịch gần nhất trong cửa sổ
    last_window_day = window_dates[-1]
    df_last = df_win[df_win['date'] == last_window_day].set_index('symbol')

    mask_from_df = (
        df_last['mask']
        .reindex(symbols)
        .fillna(0)
        .astype(bool)
        .values
    )

    # Loại thêm cổ phiếu không đủ data trong cửa sổ (> 80% phiên là NaN)
    # Tính theo từng cổ phiếu (axis=1,2 → T,F), không trộn với cổ phiếu khác
    nan_frac = np.isnan(X).all(axis=2).mean(axis=1)   # (N,) — tỷ lệ phiên toàn NaN
    has_enough_data = nan_frac <= 0.80                 # True nếu đủ data

    mask_final = mask_from_df & has_enough_data        # kết hợp hai điều kiện

    # ── Fill NaN + inf, rồi (tuỳ chọn) chuẩn hoá ─────────────
    # QUAN TRỌNG: phải đặt posinf/neginf=0.0.
    # Nếu không, np.nan_to_num biến inf → ~3.4e38 (max float32) → encoder
    # tràn số (overflow) → LayerNorm(inf) → NaN. Đây là bug đã gặp.
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if standardize:
        # Chuẩn hoá theo LOẠI biến (look-ahead-safe: chỉ dùng dữ liệu trong cửa sổ):
        #   - Biến theo-mã → CÁCH 2 (cross-sectional): mỗi phiên t, z-score các mã
        #     so với nhau → bắt "mã nào rẻ/mạnh tương đối hôm nay".
        #   - Biến vĩ mô   → CÁCH 1 (theo thời gian): giống nhau mọi mã nên
        #     không cross-sectional được; z-score theo chiều thời gian trong cửa sổ.
        # Mean/std chỉ tính trên các mã eligible (tránh mã rác chi phối).
        macro_idx = [i for i, c in enumerate(feature_cols) if c in MACRO_COLS]
        stock_idx = [i for i, c in enumerate(feature_cols) if c not in MACRO_COLS]
        elig = mask_final

        if elig.any():
            # CÁCH 2 — cross-sectional theo từng phiên (trục N), cho biến theo-mã
            for fi in stock_idx:
                sub = X[elig, :, fi]                 # (N_elig, T)
                mu  = sub.mean(axis=0)               # (T,) — trung bình các mã ở mỗi phiên
                sd  = sub.std(axis=0)                # (T,)
                sd[sd < 1e-8] = 1.0
                X[:, :, fi] = (X[:, :, fi] - mu[None, :]) / sd[None, :]

            # CÁCH 1 — theo thời gian, cho biến vĩ mô
            for fi in macro_idx:
                sub    = X[elig, :, fi]              # (N_elig, T) — các hàng giống nhau
                mu, sd = float(sub.mean()), float(sub.std())
                if sd < 1e-8:
                    sd = 1.0
                X[:, :, fi] = (X[:, :, fi] - mu) / sd

            X = np.clip(X, -clip, clip)              # chặn outlier cực đoan

    X_tensor    = torch.tensor(X, dtype=torch.float32)       # (N, T, F)
    mask_tensor = torch.tensor(mask_final, dtype=torch.bool) # (N,)

    return X_tensor, mask_tensor, symbols


# ─────────────────────────────────────────────────────────────
# BƯỚC 3a: LSTM ENCODER
# ─────────────────────────────────────────────────────────────

class LSTMEncoder(nn.Module):
    """
    (N, T, F) → (N, D)

    Linear(F→d) → LSTM(d, hidden, layers) → lấy hidden state cuối → Linear(hidden→D) → LayerNorm
    """

    def __init__(
        self,
        n_features:  int   = len(FEATURE_COLS),
        d_model:     int   = D_MODEL,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, d_model)
        self.lstm        = nn.LSTM(
            input_size  = d_model,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_size, d_model)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, T, F) → H: (N, D)"""
        x = self.input_proj(X)          # (N, T, F) → (N, T, d_model)
        _, (hn, _) = self.lstm(x)       # hn: (num_layers, N, hidden)
        h = hn[-1]                      # layer cuối: (N, hidden)
        return self.norm(self.output_proj(h))   # (N, D)


# ─────────────────────────────────────────────────────────────
# BƯỚC 3b: TRANSFORMER ENCODER
# ─────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Mã hoá vị trí thời gian bằng sin/cos — cố định, không train."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, T, d_model)"""
        return self.dropout(x + self.pe[:x.size(1)].unsqueeze(0))


class TransformerEncoder(nn.Module):
    """
    (N, T, F) → (N, D)

    Linear(F→d) → PositionalEncoding → TransformerEncoder (causal) → mean pool → Linear → LayerNorm
    """

    def __init__(
        self,
        n_features:  int   = len(FEATURE_COLS),
        d_model:     int   = D_MODEL,
        nhead:       int   = 4,
        num_layers:  int   = 2,
        dim_feedfwd: int   = 256,
        dropout:     float = 0.1,
        pooling:     str   = 'mean',
    ):
        super().__init__()
        assert d_model % nhead == 0, \
            f"d_model ({d_model}) phải chia hết cho nhead ({nhead})"

        self.pooling    = pooling
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedfwd,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: (N, T, F) → H: (N, D)"""
        x = self.pos_enc(self.input_proj(X))    # (N, T, d_model)

        T = x.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device
        )
        x = self.transformer(x, mask=causal_mask, is_causal=True)  # (N, T, d_model)

        h = x.mean(dim=1) if self.pooling == 'mean' else x[:, -1, :]
        return self.norm(self.output_proj(h))   # (N, D)


# ─────────────────────────────────────────────────────────────
# BƯỚC 4: ÁP MASK LÊN EMBEDDING
# ─────────────────────────────────────────────────────────────

def apply_mask_to_embedding(
    H:    torch.Tensor,   # (N, D)
    mask: torch.Tensor,   # (N,) bool
) -> torch.Tensor:
    """
    Zero-out cổ phiếu không eligible.
    Các nút này sẽ không lan truyền thông tin trong GNN.
    """
    return H * mask.float().unsqueeze(-1)   # (N, D)


# ─────────────────────────────────────────────────────────────
# BƯỚC 5: DEMO / SANITY CHECK
# ─────────────────────────────────────────────────────────────

def demo_with_dummy_data():
    """
    Test toàn bộ pipeline với data giả — không cần final_df.
    Chạy để xác nhận shape đúng trước khi dùng data thật.
    """
    print("=" * 60)
    print("DEMO: Pipeline với dummy data")
    print("=" * 60)

    N, T, F = 50, LOOKBACK, len(FEATURE_COLS)
    D = D_MODEL
    torch.manual_seed(42)

    X    = torch.randn(N, T, F)
    mask = torch.ones(N, dtype=torch.bool)
    mask[45:] = False   # 5 cổ phiếu cuối không eligible

    print(f"\nInput X  : {tuple(X.shape)}  (N={N}, T={T}, F={F})")
    print(f"Mask     : {mask.sum().item()} eligible / {N}")

    for name, enc in [
        ('LSTM',        LSTMEncoder(n_features=F, d_model=D)),
        ('Transformer', TransformerEncoder(n_features=F, d_model=D)),
    ]:
        enc.eval()
        with torch.no_grad():
            H        = enc(X)
            H_masked = apply_mask_to_embedding(H, mask)

        n_params = sum(p.numel() for p in enc.parameters())
        ok_shape    = H.shape == (N, D)
        ok_eligible = (H_masked[:45].abs().sum(dim=-1) > 0).all().item()
        ok_masked   = (H_masked[45:].abs().sum() == 0).item()

        print(f"\n[{name}]")
        print(f"  Params   : {n_params:,}")
        print(f"  H shape  : {tuple(H.shape)}  {'✓' if ok_shape else 'SHAPE SAI'}")
        print(f"  Eligible != 0 : {ok_eligible}")
        print(f"  Masked   == 0 : {ok_masked}")

    print(f"\n✓ Shapes đúng — sẵn sàng nhận data thật.")


def demo_with_real_data(df: pd.DataFrame):
    """
    Chạy full pipeline với final_df thật:
      1. Split train/val/test
      2. Build snapshot tại decision date đầu tiên của mỗi tập
      3. Chạy cả hai encoder
      4. In shape và thống kê mask
    """
    print("=" * 60)
    print("DEMO: Pipeline với final_df thật")
    print("=" * 60)

    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 1. Split
    print("\n── Split ──────────────────────────────")
    splits = split_dates(df)

    # 2. Lấy decision date đầu của mỗi tập để test
    sample_dates = {
        split: dates[0]
        for split, dates in splits.items()
    }

    F = len(FEATURE_COLS)
    D = D_MODEL

    lstm_enc = LSTMEncoder(n_features=F, d_model=D)
    tf_enc   = TransformerEncoder(n_features=F, d_model=D)

    for split, d_date in sample_dates.items():
        print(f"\n── {split.upper()} — decision date: {d_date.date()} ──")
        X, mask, symbols = build_snapshot_tensor(df, d_date)

        N = len(symbols)
        n_eligible = mask.sum().item()
        print(f"  N cổ phiếu     : {N}")
        print(f"  Eligible       : {n_eligible} / {N}  "
              f"({100*n_eligible/N:.1f}%)")
        print(f"  X shape        : {tuple(X.shape)}")

        for name, enc in [('LSTM', lstm_enc), ('Transformer', tf_enc)]:
            enc.eval()
            with torch.no_grad():
                H        = enc(X)
                H_masked = apply_mask_to_embedding(H, mask)
            print(f"  [{name}] H: {tuple(H.shape)}  H_masked: {tuple(H_masked.shape)}")

    print("\n✓ Pipeline chạy thành công với data thật.")
    return splits


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    demo_with_dummy_data()
    print("\n" + "─"*60)
    print("Để chạy với data thật:")
    print("  from temporal_encoder import demo_with_real_data")
    print("  splits = demo_with_real_data(final_df)")
