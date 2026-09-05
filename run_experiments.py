"""
Harness thực nghiệm cho bản revision: chạy "Full model" + các ablation bằng
CHÍNH pipeline train.py/evaluate.py đã có (không sửa logic PPO), chỉ đổi
encoder/GNN qua tham số `init`, và chạy NHIỀU SEED để báo cáo mean±std thay vì
một lần chạy — trực tiếp giải quyết:
  - Review 3: Bảng 3 (so sánh) và Bảng 4 (ablation) hiện lệch số ("Full model"
    phải ra CÙNG một con số ở cả hai bảng vì cùng một cấu hình).
  - Vấn đề cá nhân của bạn: PPO có phương sai lớn giữa các lần train do sample
    ngẫu nhiên (Dirichlet) + ít episode — nên báo cáo mean±std, không chọn
    seed đẹp nhất.

Các ablation ĐÃ CÓ SẴN 100% trong code hiện tại, không cần viết thêm gì:
  - "w/o Temporal Attention" = LSTMEncoder thay TransformerEncoder
  - "w/o GAT"                = GCN thay GAT
Vẫn CHƯA implement (xem ghi chú cuối file):
  - "w/o Rolling Correlation Graph" (đồ thị tĩnh, xây 1 lần)
  - "w/o GNN Pre-training" / "w/o PPO" (baseline hồi quy giám sát, không RL)
  - Mean-Variance, LSTM/GRU-only, FinRL-PPO, DeepPocket, AlphaGAT, NGDRL,
    TC-MAC, WCG-RL

Lưu ý: đây là bản chạy Ở QUY MÔ THU NHỎ (freq='M', n_iters ít, 2 seed) để ra
được kết quả THẬT trong một phiên làm việc — không phải cấu hình cuối cùng
cho bài báo. Xem CONFIG bên dưới để chỉnh lại quy mô đầy đủ khi có nhiều thời
gian máy hơn (freq='W', n_iters=30, seeds nhiều hơn — giống notebook gốc).
"""

import json
import time
import numpy as np
import pandas as pd
import torch

from build_final_df import build_final_df
from temporal_encoder import TransformerEncoder, LSTMEncoder
from gnn import GAT, GCN
from rl_agent import PortfolioAgent
from train import train_ppo
from evaluate import evaluate

# ─────────────────────────────────────────────────────────────
# CONFIG — chỉnh ở đây khi chạy quy mô đầy đủ
# ─────────────────────────────────────────────────────────────
FREQ       = 'M'          # 'M' (nhanh, dùng cho bản rút gọn) | 'W' (giống notebook gốc, chậm hơn nhiều)
N_ITERS    = 8             # notebook gốc dùng 30
SEEDS      = [1, 2]        # notebook gốc chỉ dùng 1 seed (seed=1) -> đây là điểm cần cải thiện
TOP_K      = 20
W_MAX      = 0.10
OUT_JSON   = 'experiment_results.json'


CONFIGS = {
    'Full model (Proposed)': dict(encoder_cls=TransformerEncoder, gnn_cls=GAT),
    'w/o Temporal Attention (LSTM thay Transformer)': dict(encoder_cls=LSTMEncoder, gnn_cls=GAT),
    'w/o GAT (GCN thay GAT)': dict(encoder_cls=TransformerEncoder, gnn_cls=GCN),
}


def run_one(df, encoder_cls, gnn_cls, seed):
    # QUAN TRỌNG: phải seed TRƯỚC KHI khởi tạo model. train_ppo() chỉ seed
    # *bên trong* nó -- nếu init được dựng trước (như ở đây, để đổi
    # encoder/gnn cho ablation) thì seed truyền vào train_ppo không kiểm
    # soát được trọng số khởi tạo, chỉ kiểm soát phần rollout. Bug này khiến
    # lần chạy trước không thực sự tái lập được theo seed.
    torch.manual_seed(seed)
    np.random.seed(seed)
    init = {'encoder': encoder_cls(), 'gnn': gnn_cls(), 'agent': PortfolioAgent()}
    result = train_ppo(
        df, n_iters=N_ITERS, freq=FREQ, top_k=TOP_K, w_max=W_MAX,
        seed=seed, init=init, verbose=False,
    )
    out = evaluate(result, df, split='test', freq=FREQ, top_k=TOP_K, w_max=W_MAX, verbose=False)
    return out['metrics'], out['series']


def main():
    df = build_final_df(cache_path='final_df.pkl')

    all_runs = {}   # config_name -> seed -> metrics dict (bao gồm cả Equal-weight, VN-Index)
    first_full_series = None   # series của seed đầu tiên của Full model, để vẽ Hình 1
    for cfg_name, cfg in CONFIGS.items():
        all_runs[cfg_name] = {}
        for seed in SEEDS:
            t0 = time.time()
            print(f"\n=== {cfg_name} | seed={seed} ===", flush=True)
            metrics, series = run_one(df, cfg['encoder_cls'], cfg['gnn_cls'], seed)
            dt = time.time() - t0
            print(f"  xong sau {dt:.0f}s | Agent Sharpe={metrics['Agent (GNN+RL)']['sharpe']:.3f} "
                  f"Return={metrics['Agent (GNN+RL)']['cum_return']*100:.1f}%", flush=True)
            all_runs[cfg_name][seed] = metrics

            if cfg_name == 'Full model (Proposed)' and first_full_series is None:
                first_full_series = series

            with open(OUT_JSON, 'w', encoding='utf-8') as f:
                json.dump(all_runs, f, indent=2, ensure_ascii=False)

    if first_full_series is not None:
        with open('full_model_series_for_fig1.json', 'w', encoding='utf-8') as f:
            json.dump({
                'dates': [str(d) for d in first_full_series['dates']],
                'agent': first_full_series['agent'],
                'equal_weight': first_full_series['equal_weight'],
                'vnindex': first_full_series['vnindex'],
            }, f, indent=2)
        print("Đã lưu full_model_series_for_fig1.json (dùng để vẽ Hình 1 khớp đúng Bảng 3)")

    # ── tổng hợp mean ± std qua các seed cho Agent (GNN+RL) mỗi config ──
    print("\n\n" + "=" * 70)
    print(f"TỔNG HỢP — freq={FREQ}, n_iters={N_ITERS}, seeds={SEEDS}, top_k={TOP_K}, w_max={W_MAX}")
    print("=" * 70)
    header = f"{'Config':<48}{'Sharpe':>14}{'Return(%)':>14}{'MaxDD(%)':>12}"
    print(header)
    for cfg_name, per_seed in all_runs.items():
        sharpes = [m['Agent (GNN+RL)']['sharpe'] for m in per_seed.values()]
        rets    = [m['Agent (GNN+RL)']['cum_return'] * 100 for m in per_seed.values()]
        dds     = [m['Agent (GNN+RL)']['max_drawdown'] * 100 for m in per_seed.values()]
        print(f"{cfg_name:<48}"
              f"{np.mean(sharpes):>7.2f}±{np.std(sharpes):<5.2f}"
              f"{np.mean(rets):>9.1f}±{np.std(rets):<4.1f}"
              f"{np.mean(dds):>7.1f}±{np.std(dds):<4.1f}")

    # baseline (không phụ thuộc seed nhưng vẫn in ra 1 lần từ run cuối cùng)
    last = list(all_runs.values())[-1][SEEDS[-1]]
    print("-" * 70)
    for name in ['Equal-weight (1/N)', 'VN-Index']:
        m = last[name]
        print(f"{name:<48}{m['sharpe']:>7.2f}{'':<6}{m['cum_return']*100:>9.1f}{'':<5}"
              f"{m['max_drawdown']*100:>7.1f}")

    print(f"\nĐã lưu chi tiết từng seed → {OUT_JSON}")


if __name__ == '__main__':
    main()
