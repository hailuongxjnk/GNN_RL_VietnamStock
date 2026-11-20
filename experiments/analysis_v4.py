"""
Tổng hợp kết quả train_v4.py (A2C và tối ưu Sharpe trực tiếp thay cho PPO), so với PPO gốc.
Cùng cách chấm như analysis_v3.py: phí 10 bps và phí theo thanh khoản; PPO gốc dưới phí theo thanh khoản lấy từ
results/v3.json (tính lại từ trọng số đã lưu).
Kết quả: results/v4.json, ../thesis/tables/v4.tex
"""
import os, glob, json, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')
import common as C
R = C.RESULTS; TAB = os.path.join(C.ROOT, 'thesis', 'tables')
YEARS = list(range(2020, 2026))
sh = lambda r: C.perf_metrics(list(r), 12)['sharpe']
V3 = json.load(open(os.path.join(R, 'v3.json')))
CFGS = ('a2c', 'direct', 'direct_tier')


def load(pat):
    return [pickle.load(open(f, 'rb')) for f in sorted(glob.glob(os.path.join(R, pat)))]


def summ(o):
    t = [x['test'] for x in o]
    vals = [[h['val_sharpe'] for h in x['history'] if 'val_sharpe' in h] for x in o]
    out = {'seeds': [x['seed'] for x in o], 'flat': [x['metrics']['sharpe'] for x in t],
           'tier': [x['metrics_tier']['sharpe'] for x in t],
           'turn': float(np.mean([np.mean(x['turnover'][1:]) for x in t])),
           'best_val': float(np.mean([x['best_val_sharpe'] for x in o])),
           'mean_val': float(np.mean([np.mean(v) for v in vals]))}
    tr = [[h['train_sharpe_soft'] for h in x['history']] for x in o if 'train_sharpe_soft' in x['history'][0]]
    if tr:
        out['train_first'] = float(np.mean([v[0] for v in tr])); out['train_last'] = float(np.mean([v[-1] for v in tr]))
    return out


S = {}
for per, sfx in (('test', ''), ('liq', '_liq')):
    S[per] = {'orig': V3[per]['orig'], 'rule': V3[per]['rule']}
    for c in CFGS:
        o = load(f'v4_{c}{sfx}_seed*.pkl')
        if o: S[per][c] = summ(o)
S['wf'] = {'orig': V3['wf']['orig'], 'rule': V3['wf']['rule']}
for c in CFGS:
    by = {}
    for f in glob.glob(os.path.join(R, 'wf', f'v4_{c}_y*_seed*.pkl')):
        o = pickle.load(open(f, 'rb')); by.setdefault(o['seed'], {})[o['year']] = o['test']
    by = {s: ys for s, ys in by.items() if all(y in ys for y in YEARS)}
    if by:
        ss = sorted(by)
        S['wf'][c] = {'seeds': ss, 'flat': [sh(sum((by[s][y]['returns'] for y in YEARS), [])) for s in ss],
                      'tier': [sh(sum((by[s][y]['returns_tier'] for y in YEARS), [])) for s in ss]}
for per in S:
    o = dict(zip(S[per]['orig']['seeds'], S[per]['orig']['tier']))
    for c in CFGS:
        if c not in S[per]: continue
        pr = [(b, o[s]) for s, b in zip(S[per][c]['seeds'], S[per][c]['tier']) if s in o]
        S[per][c]['paired_tier'] = {'n': len(pr), 'better': int(sum(a > b for a, b in pr)),
                                    'mean_diff': float(np.mean([a - b for a, b in pr])),
                                    'orig_same_seeds': float(np.mean([b for a, b in pr]))}
    print(per, {k: (round(np.mean(v['flat']), 3), round(np.mean(v['tier']), 3), len(v['flat'])) for k, v in S[per].items()})
    for c in CFGS:
        if c in S[per]: print('  ', c, {k: v for k, v in S[per][c].items() if k not in ('seeds', 'flat', 'tier')})
json.dump(S, open(os.path.join(R, 'v4.json'), 'w'), indent=1)

f = lambda x: f"{x:.2f}".replace('.', '{,}').replace('-', '$-$')
ms = lambda v: f(np.mean(v)) + (f" $\\pm$ {f(np.std(v))}" if len(v) > 1 else '')
names = [('orig', 'PPO (mô hình đề xuất)'), ('a2c', 'A2C'), ('direct', 'Tối ưu Sharpe trực tiếp, phí 10 bps'),
         ('direct_tier', 'Tối ưu Sharpe trực tiếp, phí theo TK'), ('rule', 'Quy tắc bậc thấp (không học)')]
tb = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{Kiểm tra 2024--2025} & \\multicolumn{2}{c}{Cuốn chiếu 2020--2025}"
      " & \\multicolumn{2}{c}{Tập thanh khoản} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
      "Phương pháp học & 10 bps & Theo TK & 10 bps & Theo TK & 10 bps & Theo TK \\\\\n\\midrule\n")
for k, lab in names:
    row = [lab]
    for per in ('test', 'wf', 'liq'):
        v = S[per].get(k)
        row += ['--', '--'] if v is None else [ms(v['flat']), ms(v['tier'])]
    tb += " & ".join(row) + " \\\\\n"
tb += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(TAB, 'v4.tex'), 'w').write(tb)
