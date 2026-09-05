import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

with open('full_model_series_for_fig1.json', encoding='utf-8') as f:
    s = json.load(f)

dates = pd.to_datetime(s['dates'])

fig, ax = plt.subplots(figsize=(7, 4))
for key, label in [('agent', 'Proposed (GNN+RL)'),
                    ('equal_weight', 'Equal-weight (1/N)'),
                    ('vnindex', 'VN-Index')]:
    equity = (np.cumprod(1 + np.array(s[key])) - 1) * 100
    ax.plot(dates, equity, label=label, linewidth=1.8)
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.set_xlabel('Date'); ax.set_ylabel('Cumulative return (%)')
ax.legend(); ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig('cumulative_return.pdf')
print("saved cumulative_return.pdf")
