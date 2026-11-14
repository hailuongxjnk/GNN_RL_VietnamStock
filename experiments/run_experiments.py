"""
Chạy bộ thí nghiệm của luận văn (tái cân bằng THÁNG, 8 vòng PPO — đúng thiết lập bài báo).

  python run_experiments.py            # chạy tất cả cấu hình × seed còn thiếu, 4 tiến trình
Kết quả mỗi lần chạy: results/<config>_seed<k>.pkl
"""
import io, os, sys, pickle, time, contextlib, warnings
from multiprocessing import Pool
import numpy as np
import torch

warnings.filterwarnings('ignore')
import common as C

FREQ, TOP_K, W_MAX, N_ITERS, COST = 'M', 20, 0.10, 8, 0.001
SEEDS = [1, 2, 3, 4, 5]
#            tên               encoder        graph   ma trận kề
CONFIGS = [('full',           'transformer', 'gat',  'A60'),
           ('lstm_gat',       'lstm',        'gat',  'A60'),
           ('tf_gcn',         'transformer', 'gcn',  'A60'),
           ('tf_nograph',     'transformer', 'none', 'A60'),
           ('tf_gat_long',    'transformer', 'gat',  'A250'),
           ('lstm_nograph',   'lstm',        'none', 'A60'),
           ('gru_nograph',    'gru',         'none', 'A60')]


def backtest(result, df, split, df_eval=None):
    """Giống evaluate.evaluate nhưng lưu thêm trọng số, turnover và tham số Dirichlet.
    df_eval (tuỳ chọn): bảng dữ liệu dùng để tính lãi/lỗ thực hiện (vd. đã hiệu chỉnh sự kiện doanh nghiệp)."""
    df_eval = df if df_eval is None else df_eval
    enc, gnn, agent = result['encoder'], result['gnn'], result['agent']
    enc.eval(); gnn.eval(); agent.eval()
    dates = C.quiet_split(df, FREQ)[split]
    rets, turns, W, SY, AL = [], [], [], [], []
    w_prev = None
    with torch.no_grad():
        for k in range(len(dates) - 1):
            d, d_next = dates[k], dates[k + 1]
            X, mask, syms = C.train_mod.build_snapshot_tensor(df, d)
            if int(mask.sum()) < 2:
                continue
            A = C.adj.mask_adjacency(C.train_mod.build_adjacency(df, d, syms), mask)
            R = C.train_mod.compute_holding_return(df_eval, d, d_next, syms)
            Z = gnn(C.te.apply_mask_to_embedding(enc(X), mask), A)
            w, _, _ = agent.act(Z, mask, deterministic=True)
            alpha = agent._alpha(Z)[mask].numpy()
            wn = C.train_mod.project_weights(w.numpy(), top_k=TOP_K, w_max=W_MAX)
            turn = float(np.abs(wn - (w_prev if w_prev is not None else 0.0)).sum())
            rets.append(float((wn * R).sum()) - COST * turn); turns.append(turn)
            W.append(wn); SY.append(syms); AL.append(alpha); w_prev = wn
    return {'returns': rets, 'turnover': turns, 'weights': W, 'symbols': SY, 'alpha': AL,
            'dates': list(dates[:len(rets)]), 'metrics': C.perf_metrics(rets, 12)}


def run_one(job):
    name, enc, graph, akey, seed = job
    out_path = os.path.join(C.RESULTS, f"{name}_seed{seed}.pkl")
    if os.path.exists(out_path):
        return name, seed, 'skip'
    torch.set_num_threads(1)
    df = C.load_df()
    C.CachedData(graph=akey, freq=FREQ).install()
    torch.manual_seed(seed); np.random.seed(seed)
    init = C.make_models(enc, graph)
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        res = C.train_mod.train_ppo(df, n_iters=N_ITERS, freq=FREQ, top_k=TOP_K, w_max=W_MAX,
                                    cost_rate=COST, seed=seed, device='cpu', init=init, verbose=False)
    out = {'config': name, 'seed': seed, 'history': res['history'],
           'best_val_sharpe': res['best_val_sharpe'],
           'test': backtest(res, df, 'test'), 'val': backtest(res, df, 'val'),
           'state': {k: res[k].state_dict() for k in ('encoder', 'gnn', 'agent')},
           'train_seconds': time.time() - t}
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    return name, seed, round(time.time() - t)


if __name__ == '__main__':
    only = sys.argv[1:]
    # 5 hạt giống cho mô hình chính và ba phép loại bỏ thành phần cốt lõi; 3 hạt giống cho các biến thể phụ
    n_seeds = {'full': 5, 'lstm_gat': 5, 'tf_gcn': 5, 'tf_nograph': 5}
    jobs = [(n, e, g, a, s) for (n, e, g, a) in CONFIGS for s in SEEDS[:n_seeds.get(n, 3)]
            if not only or n in only]
    with Pool(4) as p:
        for r in p.imap_unordered(run_one, jobs):
            print(time.strftime('%H:%M:%S'), r, flush=True)
