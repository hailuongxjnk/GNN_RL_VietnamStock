"""
Đối chứng cho các thí nghiệm bổ sung:
  - walk-forward: các chiến lược không học + 30 mạng chưa huấn luyện trên toàn giai đoạn 01/2020–12/2025;
  - tập thanh khoản: các chiến lược không học + 30 mạng chưa huấn luyện trên 2024–2025.
Kết quả: results/baselines_wf.pkl, results/baselines_liq.pkl, results/untrained_wf_*.pkl, results/untrained_liq.pkl
"""
import os, pickle, warnings
import numpy as np, torch
warnings.filterwarnings('ignore')
import common as C
import baselines as B
import run_experiments as RE

torch.set_num_threads(1)
df = C.load_df()
allM = sorted(C.CachedData('A250').snap.keys())
wf_dates = [d for d in allM if d.year >= 2020]

def untrained(graph, dates, n=30):
    C.CachedData(graph=graph, freq='M').install()
    C.quiet_split = lambda df_, freq: {'test': dates}
    out = []
    for seed in range(1, n + 1):
        torch.manual_seed(seed); np.random.seed(seed)
        bt = RE.backtest(C.make_models('transformer', 'gat'), df, 'test')
        out.append({'seed': seed, 'returns': bt['returns'], 'dates': bt['dates'], 'metrics': bt['metrics'],
                    'turnover': float(np.mean(bt['turnover'][1:]))})
    return out

if __name__ == '__main__':
    import sys
    part = sys.argv[1]
    if part == 'wf':
        pickle.dump(B.run('M', dates=wf_dates, graph='A250'), open(os.path.join(C.RESULTS, 'baselines_wf.pkl'), 'wb'))
        for g in ('A250', 'A60'):
            pickle.dump(untrained(g, wf_dates), open(os.path.join(C.RESULTS, f'untrained_wf_{g}.pkl'), 'wb'))
    elif part == 'liq60':
        tdates = C.quiet_split(df, 'M')['test']
        pickle.dump(untrained('A60_liq', tdates), open(os.path.join(C.RESULTS, 'untrained_liq_A60.pkl'), 'wb'))
    else:
        tdates = C.quiet_split(df, 'M')['test']
        pickle.dump(B.run('M', dates=tdates, graph='A250_liq'), open(os.path.join(C.RESULTS, 'baselines_liq.pkl'), 'wb'))
        pickle.dump(untrained('A250_liq', tdates), open(os.path.join(C.RESULTS, 'untrained_liq.pkl'), 'wb'))
    print('done', part)
