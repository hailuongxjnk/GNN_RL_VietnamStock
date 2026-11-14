"""
Tiện ích dùng chung cho các thí nghiệm của luận văn.

Nguyên tắc: KHÔNG sửa code gốc (temporal_encoder.py, adjacency.py, gnn.py, rl_agent.py,
train.py, evaluate.py). File này chỉ:
  1) cache snapshot (X, mask, symbols) và ma trận kề A theo từng decision date để không
     phải dựng lại ở mỗi lần train (kết quả giống hệt hàm gốc, chỉ nhanh hơn);
  2) định nghĩa các biến thể dùng cho ablation / baseline (GRU, không đồ thị, đồ thị cửa sổ dài);
  3) các chỉ số đánh giá mở rộng (Calmar, độ biến động, turnover).
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
CACHE = os.path.join(ROOT, "experiments", "cache")
RESULTS = os.path.join(ROOT, "experiments", "results")
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)

import temporal_encoder as te          # noqa: E402
import adjacency as adj                # noqa: E402
import gnn as gnn_mod                  # noqa: E402
import train as train_orig             # noqa: E402  (bản gốc, giữ để đối chiếu)
import train_v2 as train_mod           # noqa: E402  (bản sửa dùng cho thí nghiệm luận văn)
import evaluate as eval_mod            # noqa: E402
from temporal_encoder import FEATURE_COLS, D_MODEL, LOOKBACK  # noqa: E402

_ORIG_SNAPSHOT = te.build_snapshot_tensor
_ORIG_ADJ = adj.build_adjacency


def load_df(raw=False):
    """raw=False: dữ liệu đã điều chỉnh sự kiện doanh nghiệp (mặc định); raw=True: dữ liệu gốc như notebook."""
    df = pd.read_pickle(os.path.join(CACHE, "final_df_raw.pkl" if raw else "final_df.pkl"))
    df['date'] = pd.to_datetime(df['date'])
    return df


def quiet_split(df, freq):
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return te.split_dates(df, freq=freq)


# ───────────────────────── cache snapshot + adjacency ─────────────────────────
def snapshot_cache_path():
    return os.path.join(CACHE, "snapshots.pkl")


def build_adjacency_window(df, decision_date, symbols, window):
    """Như build_adjacency gốc nhưng cho phép cửa sổ dài; nếu lịch sử chưa đủ thì dùng hết phần có."""
    n_past = int((pd.to_datetime(df['date'].unique()) < pd.Timestamp(decision_date)).sum())
    w = min(window, n_past)
    return _ORIG_ADJ(df, decision_date, symbols, corr_window=w,
                     min_obs=min(30, w))


def compute_one(args):
    df, d = args
    X, mask, syms = _ORIG_SNAPSHOT(df, d)
    A60 = _ORIG_ADJ(df, d, syms)
    A250 = build_adjacency_window(df, d, syms, 250)
    return pd.Timestamp(d), {'X': X, 'mask': mask, 'symbols': syms, 'A60': A60, 'A250': A250}


class CachedData:
    """Thay build_snapshot_tensor / build_adjacency trong train.py & evaluate.py bằng bản tra cache."""

    _mem = {}

    def __init__(self, graph='A60', freq='M'):
        # bản cache gọn theo (tần suất, loại đồ thị) — tiết kiệm bộ nhớ khi chạy song song
        key = (freq, graph)
        if key not in CachedData._mem:
            with open(os.path.join(CACHE, f"snap_{freq}_{graph}.pkl"), 'rb') as f:
                CachedData._mem[key] = pickle.load(f)
        self.snap = CachedData._mem[key]
        self.graph = graph

    # Hàng của mã KHÔNG đủ điều kiện được đặt về 0: embedding của chúng vốn bị zero-out ngay sau encoder
    # nên kết quả không đổi, nhưng cho phép SkipEmpty bỏ qua chúng để chạy nhanh hơn.
    def snapshot(self, df, decision_date, *a, **k):
        s = self.snap[pd.Timestamp(decision_date)]
        X = s['X'].clone(); X[~s['mask']] = 0.0
        return X, s['mask'].clone(), list(s['symbols'])

    def adjacency(self, df, decision_date, symbols, *a, **k):
        return self.snap[pd.Timestamp(decision_date)][self.graph].clone()

    def install(self):
        for mod in (train_mod, train_orig, eval_mod):
            mod.build_snapshot_tensor = self.snapshot
            mod.build_adjacency = self.adjacency
        return self


# ───────────────────────── biến thể mô hình ─────────────────────────
class GRUEncoder(nn.Module):
    """Baseline GRU: cùng cấu trúc với LSTMEncoder gốc, chỉ thay LSTM bằng GRU."""

    def __init__(self, n_features=len(FEATURE_COLS), d_model=D_MODEL,
                 hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.gru = nn.GRU(d_model, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.output_proj = nn.Linear(hidden_size, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X):
        _, hn = self.gru(self.input_proj(X))
        return self.norm(self.output_proj(hn[-1]))


class NoGraph(gnn_mod.GAT):
    """
    Ablation 'không đồ thị': giữ nguyên kiến trúc/số tham số của GAT nhưng chỉ để lại self-loop,
    tức mỗi cổ phiếu KHÔNG nhận thông tin từ cổ phiếu khác.
    """

    def forward(self, H, A):
        active = (A != 0).any(dim=1).float()
        return super().forward(H, torch.diag(active))


class SkipEmpty(nn.Module):
    """Bọc một encoder: chỉ chạy trên các hàng (cổ phiếu) có dữ liệu khác 0, hàng còn lại trả về 0."""

    def __init__(self, enc):
        super().__init__(); self.enc = enc

    def forward(self, X):
        keep = X.abs().sum(dim=(1, 2)) > 0
        H = X.new_zeros(X.size(0), D_MODEL)
        if keep.any():
            H[keep] = self.enc(X[keep])
        return H


def make_models(encoder='transformer', graph='gat'):
    from rl_agent import PortfolioAgent
    enc = SkipEmpty({'transformer': te.TransformerEncoder, 'lstm': te.LSTMEncoder, 'gru': GRUEncoder}[encoder]())
    g = {'gat': gnn_mod.GAT, 'gcn': gnn_mod.GCN, 'none': NoGraph}[graph]()
    return {'encoder': enc, 'gnn': g, 'agent': PortfolioAgent()}


# ───────────────────────── chỉ số đánh giá ─────────────────────────
def perf_metrics(returns, ppy):
    r = np.asarray(returns, dtype=float)
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    peak = np.maximum(peak, 1.0)
    mdd = float(((eq - peak) / peak).min())
    cum = float(eq[-1] - 1)
    ann = float(eq[-1] ** (ppy / len(r)) - 1)
    vol = float(r.std() * np.sqrt(ppy))
    sharpe = float(r.mean() / (r.std() + 1e-12) * np.sqrt(ppy))
    calmar = float(ann / abs(mdd)) if mdd < 0 else float('nan')
    return {'cum_return': cum, 'ann_return': ann, 'ann_vol': vol, 'sharpe': sharpe,
            'max_drawdown': mdd, 'calmar': calmar}


# ───────────────────────── hiệu chỉnh sự kiện doanh nghiệp (kiểm tra độ vững) ─────────────────────────
def _legacy_clean_corporate_actions(df, thr=-0.17, market_floor=-4.0, pb_gap=0.10):
    """
    Giá trong stock_value CHƯA điều chỉnh theo cổ tức cổ phiếu / cổ phiếu thưởng / chia tách, nên tại ngày
    giao dịch không hưởng quyền xuất hiện các lợi suất âm rất lớn (vd. TCB 20/06/2024: -49%) không phải
    thua lỗ thật. Quy tắc nhận diện một sự kiện doanh nghiệp:
      (1) lợi suất logarit một ngày < thr (vượt cả biên độ -15% của UPCoM);
      (2) VN-Index hôm đó không giảm quá |market_floor|% (loại các phiên sụp đổ thật như 08/04/2025);
      (3) nếu có P/B (lấy từ nguồn đã điều chỉnh): P/B KHÔNG giảm tương ứng, tức
          dlog(P/B) - log_return > pb_gap. Thiếu P/B thì chỉ dùng (1) và (2).
    Lợi suất tại các sự kiện được đặt về 0. Đây chỉ là xấp xỉ: không bắt được các đợt chia nhỏ (<17%)
    và cổ tức tiền mặt. Trả về (df đã hiệu chỉnh, bảng các sự kiện).
    """
    out = df.sort_values(['symbol', 'date']).copy()
    dlpb = np.log(out['PX_TO_BOOK_RATIO'].where(out['PX_TO_BOOK_RATIO'] > 0)).groupby(out['symbol']).diff()
    ev = (out['log_return'] < thr) & (out['vnindex_return'] > market_floor) & \
         (dlpb.isna() | ((dlpb - out['log_return']) > pb_gap))
    events = out.loc[ev, ['date', 'symbol', 'log_return', 'mask']].assign(dlog_pb=dlpb[ev])
    out.loc[ev, 'log_return'] = 0.0
    return out, events
