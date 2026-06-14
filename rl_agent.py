"""
RL Agent (PPO) cho GNN + RL Portfolio Optimization
==================================================
Nhận:  Z (N, D)  — state, embedding các cổ phiếu (output GNN)
       mask (N,)  — cổ phiếu nào eligible
Trả:   w (N,)     — action, trọng số phân bổ danh mục (Σw = 1, w ≥ 0)

Hai mạng (actor-critic):
  - actor  : mỗi cổ phiếu Z_i → một "độ ưu tiên" → phân phối Dirichlet trên các
             mã eligible → sample ra trọng số w (nằm trên simplex, tự động Σ=1).
  - critic : gộp state → một con số "value" (kỳ vọng reward) — PPO cần để tính
             advantage.

Vì sao Dirichlet: nó là phân phối TRÊN simplex (các vector w ≥ 0, Σw = 1) — đúng
dạng của trọng số danh mục long-only, không cần softmax thủ công.

Action masking: chỉ dựng Dirichlet trên các mã eligible; mã không eligible nhận
trọng số 0 (đúng ràng buộc "không đầu tư vào mã không eligible").

Phần train PPO (advantage, clipped loss) nằm ở training loop — file này chỉ định
nghĩa agent + hai method act / evaluate mà PPO cần.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Dirichlet

try:
    from temporal_encoder import D_MODEL
except Exception:
    D_MODEL = 64


class PortfolioAgent(nn.Module):
    def __init__(self, d_model: int = D_MODEL, hidden: int = 64):
        super().__init__()
        # actor: áp cho TỪNG cổ phiếu Z_i (D,) → 1 số (độ ưu tiên)
        self.actor = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # critic: nhận state đã gộp → 1 số value
        self.critic = nn.Sequential(
            nn.Linear(d_model, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    # ── concentration α cho Dirichlet (mỗi mã một số > 0) ──
    def _alpha(self, Z: torch.Tensor) -> torch.Tensor:
        scores = self.actor(Z).squeeze(-1)        # (N,)
        return F.softplus(scores) + 1e-3          # (N,) > 0

    # ── value: gộp embedding các mã eligible rồi cho qua critic ──
    def _value(self, Z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m      = mask.float().unsqueeze(-1)               # (N,1)
        pooled = (Z * m).sum(0) / m.sum().clamp(min=1.0)  # (D,) mean trên mã eligible
        return self.critic(pooled).squeeze(-1)            # scalar

    # ── phân phối Dirichlet CHỈ trên các mã eligible ──
    def _dist(self, Z: torch.Tensor, mask: torch.Tensor):
        alpha = self._alpha(Z)
        idx   = mask.nonzero(as_tuple=True)[0]    # chỉ số các mã eligible
        return Dirichlet(alpha[idx]), idx, Z.size(0)

    # ── ACT: dùng khi sinh dữ liệu (rollout) ──
    def act(self, Z, mask, deterministic: bool = False):
        """
        Returns
        -------
        w     : (N,)  trọng số phân bổ (mã không eligible = 0, Σ = 1)
        logp  : scalar log-prob của action (PPO cần)
        value : scalar ước lượng value (critic)
        """
        dist, idx, N = self._dist(Z, mask)
        w_elig = dist.mean if deterministic else dist.rsample()
        logp   = dist.log_prob(w_elig)

        w = torch.zeros(N, device=Z.device)
        w[idx] = w_elig
        return w, logp, self._value(Z, mask)

    # ── EVALUATE: dùng khi cập nhật PPO (chấm lại action cũ) ──
    def evaluate(self, Z, mask, w):
        """Trả log-prob của w dưới policy hiện tại, entropy, và value."""
        dist, idx, N = self._dist(Z, mask)
        logp    = dist.log_prob(w[idx])
        entropy = dist.entropy()
        return logp, entropy, self._value(Z, mask)


# ─────────────────────────────────────────────────────────────
# Ép trọng số về dạng THỰC TẾ (ràng buộc danh mục)
# ─────────────────────────────────────────────────────────────

def project_weights(w, top_k=None, w_max=1.0):
    """
    Biến w (N,) thành danh mục khả thi:
      - top_k : chỉ giữ K mã có trọng số lớn nhất (bỏ đuôi vụn) → ít mã, mỗi mã đủ lớn.
      - w_max : trần mỗi mã (vd 0.10 = tối đa 10%/mã), phần thừa chia lại cho mã khác.
    Mặc định không ràng buộc (top_k=None, w_max=1.0).
    """
    w = np.asarray(w, dtype=float).copy()

    if top_k is not None and top_k < int((w > 0).sum()):
        keep = np.argsort(w)[::-1][:top_k]
        m = np.zeros_like(w, dtype=bool); m[keep] = True
        w[~m] = 0.0

    s = w.sum()
    if s > 0:
        w = w / s

    if w_max < 1.0:
        for _ in range(100):                       # lặp đến khi không mã nào vượt trần
            over = w > w_max + 1e-9
            if not over.any():
                break
            excess = (w[over] - w_max).sum()
            w[over] = w_max
            under = (w < w_max - 1e-12) & (w > 0)
            if not under.any():
                break
            w[under] += excess * (w[under] / w[under].sum())
        s = w.sum()
        if s > 0:
            w = w / s

    return w


# ─────────────────────────────────────────────────────────────
# DEMO / SANITY CHECK
# ─────────────────────────────────────────────────────────────

def demo_with_dummy_data():
    print("=" * 60)
    print("DEMO: PortfolioAgent (PPO) với dummy state")
    print("=" * 60)

    torch.manual_seed(0)
    N, D = 30, D_MODEL
    Z    = torch.randn(N, D)
    mask = torch.ones(N, dtype=torch.bool); mask[25:] = False   # 5 mã không eligible

    agent = PortfolioAgent(d_model=D)

    w, logp, value = agent.act(Z, mask)
    print(f"\nw shape         : {tuple(w.shape)}")
    print(f"Σw              : {w.sum().item():.4f}  (kỳ vọng 1.0)")
    print(f"w ≥ 0           : {(w >= 0).all().item()}")
    print(f"w mã không elig : {w[25:].abs().sum().item():.4f}  (kỳ vọng 0)")
    print(f"log_prob        : {logp.item():.3f}   value: {value.item():.3f}")

    # action xác định (dùng lúc test)
    w_det, _, _ = agent.act(Z, mask, deterministic=True)
    print(f"\nΣw (deterministic): {w_det.sum().item():.4f}")

    # evaluate lại action cũ + kiểm tra gradient (PPO cần)
    logp2, ent, val2 = agent.evaluate(Z, mask, w.detach())
    loss = -(logp2 * 1.0) + 0.5 * (val2 - 0.0) ** 2 - 0.01 * ent
    loss.backward()
    grad_ok = all(p.grad is not None for p in agent.parameters())
    print(f"entropy         : {ent.item():.3f}")
    print(f"Gradient OK     : {grad_ok}")

    print("\n✓ PortfolioAgent chạy đúng — sẵn sàng cho training loop.")


def demo_with_real_data(df):
    """Chuỗi đầy đủ: encoder → A → GNN → agent, trên final_df thật tại 1 decision date."""
    from temporal_encoder import (build_snapshot_tensor, apply_mask_to_embedding,
                                  split_dates, TransformerEncoder)
    from adjacency import build_adjacency, mask_adjacency
    from gnn import GAT

    print("=" * 60)
    print("DEMO: encoder → A → GNN → agent với final_df thật")
    print("=" * 60)

    splits = split_dates(df)
    d_date = splits['test'][0]

    X, mask, symbols = build_snapshot_tensor(df, d_date)
    A_masked = mask_adjacency(build_adjacency(df, d_date, symbols), mask)

    enc, gnn, agent = TransformerEncoder(), GAT(), PortfolioAgent()
    enc.eval(); gnn.eval(); agent.eval()
    with torch.no_grad():
        H = apply_mask_to_embedding(enc(X), mask)
        Z = gnn(H, A_masked)
        w, logp, value = agent.act(Z, mask, deterministic=True)

    print(f"\ndecision_date : {d_date.date()}")
    print(f"N / eligible  : {len(symbols)} / {int(mask.sum())}")
    print(f"Σw            : {w.sum().item():.4f}   value: {value.item():.3f}")
    top = torch.topk(w, 5)
    print("Top 5 phân bổ :")
    for wi, ti in zip(top.values, top.indices):
        print(f"  {symbols[ti]:<7} {wi.item():.3f}")
    print("\n✓ Chuỗi đầy đủ chạy OK.")


if __name__ == '__main__':
    demo_with_dummy_data()
