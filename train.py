"""
Training Loop (PPO) — ghép tất cả lại, train end-to-end
=======================================================
Mỗi vòng lặp gồm 2 pha (như đã giải thích):

  PHA 1 — Rollout : cho agent chạy qua chuỗi decision date với tham số hiện tại,
                    lưu (action, log_prob, value) và tính reward (DSR trên lợi
                    nhuận ròng sau phí giao dịch).
  PHA 2 — Update  : tính advantage (GAE) → loss PPO (actor clip + critic + entropy)
                    → backward. Gradient chảy ngược qua agent → GNN → encoder.

Lưu ý: forward (encoder→GNN) được TÍNH LẠI ở pha update để gradient chảy tới cả
encoder và GNN (train end-to-end), không chỉ riêng agent.
"""

import numpy as np
import torch

from temporal_encoder import (build_snapshot_tensor, apply_mask_to_embedding,
                              split_dates, TransformerEncoder)
from adjacency import build_adjacency, mask_adjacency
from gnn import GAT
from rl_agent import PortfolioAgent, project_weights


# ─────────────────────────────────────────────────────────────
# Lợi nhuận nắm giữ từ d0 → d1 cho mỗi cổ phiếu (simple return)
# ─────────────────────────────────────────────────────────────

def compute_holding_return(df, d0, d1, symbols):
    sub = df[(df['date'] > d0) & (df['date'] <= d1)]
    cum = sub.groupby('symbol')['log_return'].sum().reindex(symbols).fillna(0.0)
    return np.exp(cum.values) - 1.0          # (N,) simple return


# ─────────────────────────────────────────────────────────────
# Differential Sharpe Ratio — reward từng bước
# ─────────────────────────────────────────────────────────────

class DSR:
    def __init__(self, eta: float = 0.01):
        self.A = 0.0; self.B = 0.0; self.eta = eta; self.init = False

    def step(self, r: float) -> float:
        if not self.init:                    # bước đầu chưa có Sharpe → 0
            self.A, self.B, self.init = r, r * r, True
            return 0.0
        dA, dB = r - self.A, r * r - self.B
        denom  = (self.B - self.A ** 2) ** 1.5
        d = 0.0 if denom < 1e-12 else (self.B * dA - 0.5 * self.A * dB) / denom
        self.A += self.eta * dA; self.B += self.eta * dB
        return float(d)


# ─────────────────────────────────────────────────────────────
# GAE — advantage "thực tế trừ kỳ vọng", gộp nhiều bước
# ─────────────────────────────────────────────────────────────

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    T = len(rewards); adv = [0.0] * T; gae = 0.0
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else 0.0
        delta  = rewards[t] + gamma * next_v - values[t]
        gae    = delta + gamma * lam * gae
        adv[t] = gae
    returns = [adv[t] + values[t] for t in range(T)]
    return adv, returns


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────

def train_ppo(
    df,
    n_iters:    int   = 20,
    ppo_epochs: int   = 4,
    lr:         float = 3e-4,
    clip:       float = 0.2,
    c_v:        float = 0.5,     # trọng số critic loss
    c_e:        float = 0.01,    # trọng số entropy
    cost_rate:  float = 0.001,   # phí giao dịch (10 bps)
    gamma:      float = 0.99,
    lam:        float = 0.95,
    reward_clip: float = 5.0,    # kẹp reward DSR để tránh nổ khi phương sai ~0
    top_k:      int   = None,    # chỉ giữ K mã lớn nhất (None = không giới hạn)
    w_max:      float = 1.0,     # trần tỷ trọng mỗi mã (1.0 = không trần)
    freq:       str   = 'M',     # 'M' tháng | 'W' tuần (nhiều decision point hơn)
    device:     str   = None,    # 'cuda'/'cpu'; None = tự dò GPU
    seed:       int   = 42,      # cố định kết quả; đặt None nếu muốn ngẫu nhiên
    init=None,                   # checkpoint để train tiếp (None = train từ đầu)
    verbose:    bool  = True,
):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    df = df.copy(); df['date'] = __import__('pandas').to_datetime(df['date'])
    _sp       = split_dates(df, freq=freq)
    dates     = _sp['train']
    val_dates = _sp['val']
    ppy       = 52 if freq == 'W' else 12     # số kỳ/năm để quy đổi Sharpe val

    if init is None:
        enc, gnn, agent = TransformerEncoder(), GAT(), PortfolioAgent()
    else:                                     # train tiếp từ checkpoint đã lưu
        enc, gnn, agent = init['encoder'], init['gnn'], init['agent']

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    enc.to(device); gnn.to(device); agent.to(device)
    if verbose:
        print(f"Device: {device}")

    params = list(enc.parameters()) + list(gnn.parameters()) + list(agent.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    # ── Tiền xử lý 1 LẦN: snapshot + lợi nhuận nắm giữ (không đổi qua các iter) ──
    def _build_episodes(decision_dates):
        eps = []
        for j in range(len(decision_dates) - 1):     # bỏ ngày cuối (không có lợi nhuận tương lai)
            d, d_next = decision_dates[j], decision_dates[j + 1]
            X, mask, syms = build_snapshot_tensor(df, d)
            if int(mask.sum()) < 2:                   # cần >= 2 mã eligible mới phân bổ được
                continue
            A = mask_adjacency(build_adjacency(df, d, syms), mask)
            R = compute_holding_return(df, d, d_next, syms)
            eps.append({'X': X, 'A': A, 'mask': mask, 'R': R})
        return eps

    episodes     = _build_episodes(dates)
    val_episodes = _build_episodes(val_dates)

    # ── tính Sharpe trên một tập episode (deterministic, không train) ──
    def _sharpe(eps):
        if len(eps) < 2:
            return float('nan')
        rets, w_prev = [], None
        with torch.no_grad():
            for ep in eps:
                X, A, mask = ep['X'].to(device), ep['A'].to(device), ep['mask'].to(device)
                Z = gnn(apply_mask_to_embedding(enc(X), mask), A)
                w, _, _ = agent.act(Z, mask, deterministic=True)
                wn    = project_weights(w.cpu().numpy(), top_k=top_k, w_max=w_max)
                gross = float((wn * ep['R']).sum())
                turn  = float(np.abs(wn - (w_prev if w_prev is not None else 0.0)).sum())
                rets.append(gross - cost_rate * turn); w_prev = wn
        r = np.asarray(rets)
        return float(r.mean() / (r.std() + 1e-12) * np.sqrt(ppy))

    # ── chụp / phục hồi trọng số (để giữ model val tốt nhất) ──
    def _snapshot():
        return {m: {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                for m, net in (('encoder', enc), ('gnn', gnn), ('agent', agent))}

    if verbose:
        print(f"Episodes train: {len(episodes)} | val: {len(val_episodes)}")

    best_val, best_state = -float('inf'), None
    history = []
    for it in range(n_iters):
        # ===== PHA 1: ROLLOUT =====
        dsr = DSR(); w_prev = None
        rewards, values, logp_old, actions = [], [], [], []
        with torch.no_grad():
            for ep in episodes:
                X, A, mask = ep['X'].to(device), ep['A'].to(device), ep['mask'].to(device)
                Z = gnn(apply_mask_to_embedding(enc(X), mask), A)
                w, lp, v = agent.act(Z, mask)
                wn    = project_weights(w.cpu().numpy(), top_k=top_k, w_max=w_max)
                gross = float((wn * ep['R']).sum())
                turn  = float(np.abs(wn - (w_prev if w_prev is not None else 0.0)).sum())
                net   = gross - cost_rate * turn        # lợi nhuận ròng sau phí
                r     = float(np.clip(dsr.step(net), -reward_clip, reward_clip))
                rewards.append(r)
                values.append(float(v)); logp_old.append(float(lp)); actions.append(w)
                w_prev = wn

        adv, rets = compute_gae(rewards, values, gamma, lam)
        adv_t = torch.tensor(adv, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)   # chuẩn hoá advantage
        ret_t = torch.tensor(rets, dtype=torch.float32, device=device)
        lpold = torch.tensor(logp_old, dtype=torch.float32, device=device)

        # ===== PHA 2: UPDATE (tính lại forward → gradient tới enc+gnn+agent) =====
        # backward TỪNG episode để giải phóng graph ngay → RAM/VRAM không nổ theo số episode
        for _ in range(ppo_epochs):
            opt.zero_grad(); loss_val = 0.0
            for k, ep in enumerate(episodes):
                X, A, mask = ep['X'].to(device), ep['A'].to(device), ep['mask'].to(device)
                Z = gnn(apply_mask_to_embedding(enc(X), mask), A)
                lp, ent, v = agent.evaluate(Z, mask, actions[k])
                ratio = torch.exp(lp - lpold[k])
                s1, s2 = ratio * adv_t[k], torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[k]
                actor_loss  = -torch.min(s1, s2)
                critic_loss = (v - ret_t[k]) ** 2
                loss_ep = (actor_loss + c_v * critic_loss - c_e * ent) / len(episodes)
                loss_ep.backward()                  # tích luỹ gradient, giải phóng graph episode
                loss_val += float(loss_ep)
            torch.nn.utils.clip_grad_norm_(params, 1.0)   # chặn gradient nổ
            opt.step()

        mean_rew = float(np.mean(rewards))
        val_sh   = _sharpe(val_episodes)                  # Sharpe trên validation
        history.append({'reward': mean_rew, 'val_sharpe': val_sh})

        if val_episodes and not np.isnan(val_sh) and val_sh > best_val:
            best_val, best_state = val_sh, _snapshot()    # giữ lại model val tốt nhất

        if verbose:
            star = ' *' if best_state is not None and val_sh == best_val else ''
            print(f"iter {it:02d} | reward {mean_rew:+.4f} | "
                  f"val Sharpe {val_sh:+.3f}{star} | loss {loss_val:+.4f}")

    # ── phục hồi model val tốt nhất (thay vì lấy model vòng cuối = dễ overfit nhất) ──
    if best_state is not None:
        enc.load_state_dict(best_state['encoder'])
        gnn.load_state_dict(best_state['gnn'])
        agent.load_state_dict(best_state['agent'])
        if verbose:
            print(f"\n→ Giữ model có val Sharpe tốt nhất: {best_val:+.3f}")

    return {'encoder': enc, 'gnn': gnn, 'agent': agent,
            'history': history, 'best_val_sharpe': best_val}


# ─────────────────────────────────────────────────────────────
# LƯU / NẠP MODEL — khỏi train lại từ đầu
# ─────────────────────────────────────────────────────────────

def save_checkpoint(result, path: str = 'checkpoint.pt'):
    """Lưu trọng số encoder + gnn + agent (và lịch sử reward) ra file."""
    torch.save({
        'encoder': result['encoder'].state_dict(),
        'gnn':     result['gnn'].state_dict(),
        'agent':   result['agent'].state_dict(),
        'history': result.get('history', []),
    }, path)
    print(f"Đã lưu model → {path}")


def load_checkpoint(path: str = 'checkpoint.pt'):
    """Nạp lại model đã lưu. Trả dict giống output của train_ppo."""
    ckpt = torch.load(path, map_location='cpu')
    enc, gnn, agent = TransformerEncoder(), GAT(), PortfolioAgent()
    enc.load_state_dict(ckpt['encoder'])
    gnn.load_state_dict(ckpt['gnn'])
    agent.load_state_dict(ckpt['agent'])
    print(f"Đã nạp model ← {path}")
    return {'encoder': enc, 'gnn': gnn, 'agent': agent,
            'history': ckpt.get('history', [])}


if __name__ == '__main__':
    import pandas as pd
    df = pd.read_pickle('fake_final_df.pkl')
    train_ppo(df, n_iters=5)
