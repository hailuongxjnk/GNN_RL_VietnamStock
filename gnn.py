"""
GNN cho GNN + RL Portfolio Optimization
=======================================
Nhận:  H (N, D)  — embedding mỗi cổ phiếu (từ encoder, đã mask)
       A (N, N)  — ma trận kề (từ build_adjacency, đã mask)
Trả:   Z (N, D') — embedding MỚI, đã tích hợp thông tin từ các cổ phiếu liên quan

Hai biến thể (song song LSTM vs Transformer bên encoder):
  - GCN : baseline — trọng số tổng hợp cố định theo cấu trúc đồ thị
  - GAT : main model — học trọng số attention cho từng cạnh

Vị trí trong pipeline:
    build_snapshot_tensor → X (N,T,F), mask, symbols
    encoder(X)            → H (N, D)
    apply_mask_to_embedding(H, mask)        → H_masked
    build_adjacency(...) → A ; mask_adjacency(A, mask) → A_masked
    gnn(H_masked, A_masked)                 → Z (N, D')   ← state cho RL

Quy ước mask: A_masked đã zero hàng/cột của cổ phiếu KHÔNG eligible.
  → module tự nhận diện nút eligible = hàng có ít nhất 1 cạnh (kể cả self-loop),
    và zero-out output của nút không eligible. Không cần truyền mask riêng.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from temporal_encoder import D_MODEL
except Exception:
    D_MODEL = 64

try:
    from adjacency import normalize_adjacency
except Exception:
    def normalize_adjacency(A, eps=1e-8):
        deg = A.sum(dim=1)
        d = (deg + eps).pow(-0.5)
        return d.unsqueeze(1) * A * d.unsqueeze(0)


# ─────────────────────────────────────────────────────────────
# Tiện ích: nút nào "active" (eligible) dựa trên A đã mask
# ─────────────────────────────────────────────────────────────

def _active_nodes(A: torch.Tensor) -> torch.Tensor:
    """Nút eligible = hàng có ít nhất một cạnh khác 0 trong A_masked. (N,) bool"""
    return (A != 0).any(dim=1)


# ═════════════════════════════════════════════════════════════
# GCN  (baseline)
# ═════════════════════════════════════════════════════════════

class GCNLayer(nn.Module):
    """Một tầng GCN dense:  H' = Â · H · W   (σ áp ở ngoài)."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, H: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        return A_norm @ self.lin(H)     # (N, out_dim)


class GCN(nn.Module):
    """
    (H (N,D), A (N,N)) → Z (N, out_dim)

    Chuẩn hoá đối xứng Â = D^-1/2 (A) D^-1/2 một lần, rồi 2 tầng GCN.
    """

    def __init__(
        self,
        in_dim:   int = D_MODEL,
        hidden:   int = 64,
        out_dim:  int = D_MODEL,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.gc1     = GCNLayer(in_dim, hidden)
        self.gc2     = GCNLayer(hidden, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(out_dim)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        active = _active_nodes(A)               # (N,) bool
        A_norm = normalize_adjacency(A)         # Â (N, N)

        h = F.relu(self.gc1(H, A_norm))         # (N, hidden)
        h = self.dropout(h)
        z = self.gc2(h, A_norm)                 # (N, out_dim)
        z = self.norm(z)

        return z * active.float().unsqueeze(-1)  # zero-out nút không eligible


# ═════════════════════════════════════════════════════════════
# GAT  (main model)
# ═════════════════════════════════════════════════════════════

class GATLayer(nn.Module):
    """
    Một tầng Graph Attention (dense, multi-head).
    Chỉ tính attention trên các cạnh tồn tại (edge_mask) → tự học trọng số.
    """

    def __init__(
        self,
        in_dim:  int,
        out_dim: int,
        n_heads: int = 4,
        concat:  bool = True,
        dropout: float = 0.1,
        leaky:   float = 0.2,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat  = concat

        self.W     = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        self.leaky   = nn.LeakyReLU(leaky)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, H: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        """H: (N, in_dim) ; edge_mask: (N, N) bool — True nếu i nhận tin từ j."""
        N  = H.size(0)
        Wh = self.W(H).view(N, self.n_heads, self.out_dim)    # (N, heads, out)

        # e[i,j,h] = LeakyReLU( a_src·Wh[i] + a_dst·Wh[j] )
        src = (Wh * self.a_src).sum(-1)                       # (N, heads) — phần của i
        dst = (Wh * self.a_dst).sum(-1)                       # (N, heads) — phần của j
        e   = self.leaky(src.unsqueeze(1) + dst.unsqueeze(0)) # (N, N, heads)

        # Chặn các cặp không có cạnh → -inf để softmax bỏ qua
        e = e.masked_fill(~edge_mask.unsqueeze(-1), float('-inf'))

        alpha = torch.softmax(e, dim=1)                       # chuẩn hoá theo hàng xóm j
        alpha = torch.nan_to_num(alpha, nan=0.0)              # phòng hàng rỗng
        alpha = self.dropout(alpha)

        # Tổng hợp: out[i,h] = Σ_j alpha[i,j,h] · Wh[j,h]
        out = torch.einsum('ijh,jhd->ihd', alpha, Wh)         # (N, heads, out)

        if self.concat:
            return out.reshape(N, self.n_heads * self.out_dim)
        return out.mean(dim=1)                                # (N, out)


class GAT(nn.Module):
    """
    (H (N,D), A (N,N)) → Z (N, out_dim)

    2 tầng GAT:
      tầng 1: multi-head, concat   → (N, heads*hidden)
      tầng 2: gộp head (average)   → (N, out_dim)
    A chỉ dùng để xác định CẠNH NÀO tồn tại; trọng số do attention tự học.
    """

    def __init__(
        self,
        in_dim:   int = D_MODEL,
        hidden:   int = 16,
        out_dim:  int = D_MODEL,
        n_heads:  int = 4,
        dropout:  float = 0.1,
    ):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden, n_heads=n_heads,
                             concat=True, dropout=dropout)
        self.gat2 = GATLayer(hidden * n_heads, out_dim, n_heads=1,
                             concat=False, dropout=dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        active = _active_nodes(A)                              # (N,) bool

        # Cạnh tồn tại = A khác 0; thêm self-loop để không có hàng rỗng (tránh NaN)
        N = A.size(0)
        eye = torch.eye(N, dtype=torch.bool, device=A.device)
        edge_mask = (A != 0) | eye

        h = F.elu(self.gat1(H, edge_mask))                    # (N, heads*hidden)
        z = self.gat2(h, edge_mask)                           # (N, out_dim)
        z = self.norm(z)

        return z * active.float().unsqueeze(-1)               # zero-out nút không eligible


# ═════════════════════════════════════════════════════════════
# DEMO / SANITY CHECK
# ═════════════════════════════════════════════════════════════

def demo_with_dummy_data():
    print("=" * 60)
    print("DEMO: GCN & GAT với dummy data")
    print("=" * 60)

    torch.manual_seed(0)
    N, D = 30, D_MODEL

    # H giả (như output encoder)
    H = torch.randn(N, D)

    # A giả: đối xứng, ngưỡng hoá, self-loop; 5 nút cuối "không eligible" (zero)
    C = torch.rand(N, N); C = (C + C.t()) / 2
    A = torch.where(C > 0.6, C, torch.zeros_like(C))
    A.fill_diagonal_(1.0)
    A[25:, :] = 0; A[:, 25:] = 0          # 5 nút cuối bị mask
    H[25:] = 0

    print(f"\nH: {tuple(H.shape)}   A: {tuple(A.shape)}   "
          f"eligible: {int(_active_nodes(A).sum())}/{N}")

    for name, Model in [('GCN', GCN), ('GAT', GAT)]:
        model = Model(in_dim=D, out_dim=D)
        Z = model(H, A)

        n_params  = sum(p.numel() for p in model.parameters())
        ok_shape  = Z.shape == (N, D)
        ok_nan    = not torch.isnan(Z).any().item()
        ok_masked = (Z[25:].abs().sum() == 0).item()
        ok_active = (Z[:25].abs().sum(-1) > 0).all().item()

        # kiểm tra gradient chảy được (cho train end-to-end sau này)
        loss = Z[:25].pow(2).mean(); loss.backward()
        ok_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)

        print(f"\n[{name}]")
        print(f"  Params         : {n_params:,}")
        print(f"  Z shape        : {tuple(Z.shape)}  {'✓' if ok_shape else 'SAI'}")
        print(f"  Không NaN      : {ok_nan}")
        print(f"  Nút eligible≠0 : {ok_active}")
        print(f"  Nút masked ==0 : {ok_masked}")
        print(f"  Gradient OK    : {ok_grad}")

    print("\n✓ GCN & GAT chạy đúng — sẵn sàng nối vào RL agent.")


def demo_with_real_data(df):
    """Chạy full encoder → adjacency → GNN trên final_df thật tại 1 decision date."""
    from temporal_encoder import (
        build_snapshot_tensor, apply_mask_to_embedding,
        split_dates, LSTMEncoder, TransformerEncoder,
    )
    from adjacency import build_adjacency, mask_adjacency

    print("=" * 60)
    print("DEMO: encoder → A → GNN với final_df thật")
    print("=" * 60)

    splits = split_dates(df)
    d_date = splits['test'][0]

    X, mask, symbols = build_snapshot_tensor(df, d_date)
    A_masked = mask_adjacency(build_adjacency(df, d_date, symbols), mask)
    N = len(symbols)

    encoder = TransformerEncoder()
    encoder.eval()
    with torch.no_grad():
        H        = encoder(X)
        H_masked = apply_mask_to_embedding(H, mask)

    print(f"\ndecision_date : {d_date.date()}")
    print(f"N / eligible  : {N} / {int(mask.sum())}")
    print(f"H_masked      : {tuple(H_masked.shape)}   A_masked: {tuple(A_masked.shape)}")

    for name, Model in [('GCN', GCN), ('GAT', GAT)]:
        model = Model()
        model.eval()
        with torch.no_grad():
            Z = model(H_masked, A_masked)
        print(f"  [{name}] Z: {tuple(Z.shape)}   NaN: {torch.isnan(Z).any().item()}")

    print("\n✓ Pipeline encoder → A → GNN chạy OK với data thật.")
    return splits


if __name__ == '__main__':
    demo_with_dummy_data()
