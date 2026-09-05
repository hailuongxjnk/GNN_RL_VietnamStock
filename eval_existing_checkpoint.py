import sys
from build_final_df import build_final_df
from train import load_checkpoint
from evaluate import evaluate

df = build_final_df(cache_path='final_df.pkl')

for ckpt_path, freq, top_k, w_max in [
    (r"D:\GNN_RL_VietnamStock\model_weekly.pt", 'W', 20, 0.10),
    (r"D:\GNN_RL_VietnamStock\model.pt", 'M', None, 1.0),
]:
    print("\n" + "=" * 70)
    print(f"Checkpoint: {ckpt_path}  (freq={freq}, top_k={top_k}, w_max={w_max})")
    print("=" * 70)
    try:
        result = load_checkpoint(ckpt_path)
        out = evaluate(result, df, split='test', freq=freq, top_k=top_k, w_max=w_max, verbose=True)
    except Exception as e:
        print(f"LỖI khi evaluate {ckpt_path}: {e}")
