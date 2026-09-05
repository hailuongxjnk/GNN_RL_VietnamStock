import json
import numpy as np

with open('experiment_results.json', encoding='utf-8') as f:
    data = json.load(f)

def calmar(ret, dd):
    return ret / abs(dd) if dd != 0 else float('inf')

print(f"{'Config':<45}{'Sharpe':>16}{'Return(%)':>16}{'MaxDD(%)':>14}{'Calmar':>12}")
for cfg, per_seed in data.items():
    sharpes, rets, dds, cals = [], [], [], []
    for seed, m in per_seed.items():
        a = m['Agent (GNN+RL)']
        sharpes.append(a['sharpe']); rets.append(a['cum_return']*100); dds.append(a['max_drawdown']*100)
        cals.append(calmar(a['cum_return'], a['max_drawdown']))
    print(f"{cfg:<45}"
          f"{np.mean(sharpes):>8.2f}+/-{np.std(sharpes):<5.2f}"
          f"{np.mean(rets):>9.1f}+/-{np.std(rets):<5.1f}"
          f"{np.mean(dds):>7.1f}+/-{np.std(dds):<5.1f}"
          f"{np.mean(cals):>6.2f}+/-{np.std(cals):<4.2f}")

# baseline (constant across configs/seeds, take from any entry)
last = list(list(data.values())[0].values())[0]
print("-"*100)
for name in ['Equal-weight (1/N)', 'VN-Index']:
    m = last[name]
    c = calmar(m['cum_return'], m['max_drawdown'])
    print(f"{name:<45}{m['sharpe']:>8.2f}      {m['cum_return']*100:>9.1f}      {m['max_drawdown']*100:>7.1f}      {c:>6.2f}")
