"""
Đối chứng 'mạng chưa huấn luyện': khởi tạo ngẫu nhiên Transformer + GAT + tác tử (KHÔNG huấn luyện),
chạy chính sách tất định trên tập kiểm tra. Nếu phân phối kết quả của mạng chưa huấn luyện trùng với
kết quả sau huấn luyện thì việc huấn luyện chưa tạo ra khác biệt đo được.
Kết quả: results/untrained.pkl
"""
import pickle, os, warnings
import numpy as np, torch
warnings.filterwarnings('ignore')
import common as C
import run_experiments as RE

if __name__ == '__main__':
    torch.set_num_threads(1)
    import sys
    graph = sys.argv[1] if len(sys.argv) > 1 else 'A250'      # mặc định: đồ thị dài hạn (cấu hình chính)
    df = C.load_df(); C.CachedData(graph=graph, freq='M').install()
    out = []
    for seed in range(1, 31):
        torch.manual_seed(seed); np.random.seed(seed)
        res = C.make_models('transformer', 'gat')
        bt = RE.backtest(res, df, 'test')
        held = [set(np.array(sy)[w > 0]) for w, sy in zip(bt['weights'], bt['symbols'])]
        out.append({'seed': seed, 'metrics': bt['metrics'], 'returns': bt['returns'],
                    'turnover': float(np.mean(bt['turnover'][1:])), 'held': held})
        print(seed, {k: round(v, 3) for k, v in bt['metrics'].items()}, flush=True)
    sh = np.array([o['metrics']['sharpe'] for o in out])
    print('untrained Sharpe: mean %.2f std %.2f min %.2f max %.2f' % (sh.mean(), sh.std(ddof=1), sh.min(), sh.max()))
    pickle.dump(out, open(os.path.join(C.RESULTS, f'untrained_{graph}.pkl'), 'wb'))
