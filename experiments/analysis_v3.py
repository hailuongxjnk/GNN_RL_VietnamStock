"""
Tổng hợp kết quả train_v3.py (hai cải tiến cho thành phần học tăng cường) và so sánh với PPO gốc.

Mỗi cấu hình được chấm ở hai mức phí: 10 bps cố định và phí theo thanh khoản (10/25/50 bps theo Amihud).
PPO gốc (tf_gat_long) ở mức phí theo thanh khoản được tính lại từ tỷ trọng đã lưu, cùng hàm tier_vec của train_v3.
Kết quả: results/v3.json, ../thesis/tables/v3.tex
"""
import os, glob, json, pickle, warnings
import numpy as np
warnings.filterwarnings('ignore')
import common as C
import train_v3 as V
R = C.RESULTS; TAB = os.path.join(C.ROOT, 'thesis', 'tables')
YEARS = list(range(2020, 2026))
df = C.load_df()
sh = lambda r: C.perf_metrics(list(r), 12)['sharpe']


def recost(bt, graph):
    """lợi suất PPO gốc dưới phí theo thanh khoản, từ tỷ trọng đã lưu"""
    cd = C.CachedData(graph=graph); out, wp = [], None
    for d, w, r, t in zip(bt['dates'], bt['weights'], bt['returns'], bt['turnover']):
        _, mask, syms = cd.snapshot(df, d)
        c = V.tier_vec(df, d, syms, mask.numpy())
        dw = w - (wp if wp is not None else 0.0)
        out.append(r + V.FLAT * t - float((np.abs(dw) * c).sum())); wp = w
    return out


def load(pat):
    return [pickle.load(open(f, 'rb')) for f in sorted(glob.glob(os.path.join(R, pat)))]


def wf_load(cfg):
    by = {}
    for f in glob.glob(os.path.join(R, 'wf', f'{cfg}_y*_seed*.pkl')):
        o = pickle.load(open(f, 'rb')); by.setdefault(o['seed'], {})[o['year']] = o['test']
    return {s: ys for s, ys in by.items() if all(y in ys for y in YEARS)}


S = {}
for per, graph, orig_pat, sfx in (('test', 'A250', 'tf_gat_long_seed*.pkl', ''), ('liq', 'A250_liq', 'tf_gat_long_liq_seed*.pkl', '_liq')):
    S[per] = {}
    o = load(orig_pat)
    S[per]['orig'] = {'seeds': [x['seed'] for x in o], 'flat': [sh(x['test']['returns']) for x in o],
                      'tier': [sh(recost(x['test'], graph)) for x in o]}
    for cfg in ('tier', 'excess', 'long20'):
        o = load(f'v3_{cfg}{sfx}_seed*.pkl')
        if not o: continue
        S[per][cfg] = {'seeds': [x['seed'] for x in o], 'flat': [x['test']['metrics']['sharpe'] for x in o],
                       'tier': [x['test']['metrics_tier']['sharpe'] for x in o],
                       'turn': float(np.mean([np.mean(x['test']['turnover'][1:]) for x in o]))}
        rf, rt = o[0]['test']['rule_flat'], o[0]['test']['rule_tier']
    S[per]['rule'] = {'flat': [sh(rf)], 'tier': [sh(rt)]}
    S[per]['orig']['turn'] = float(np.mean([np.mean(x['test']['turnover'][1:]) for x in load(orig_pat)]))
    print(per, {k: (round(np.mean(v['flat']), 3), round(np.mean(v['tier']), 3), len(v['flat'])) for k, v in S[per].items()}, flush=True)

# cuốn chiếu 2020–2025: nối 6 năm cho mỗi hạt giống
S['wf'] = {}
wo = wf_load('tf_gat_long')
S['wf']['orig'] = {'seeds': sorted(wo), 'flat': [sh(sum((wo[s][y]['returns'] for y in YEARS), [])) for s in sorted(wo)],
                   'tier': [sh(sum((recost(wo[s][y], 'A250') for y in YEARS), [])) for s in sorted(wo)]}
for cfg in ('tier', 'excess'):
    w = wf_load(f'v3_{cfg}')
    if not w: continue
    S['wf'][cfg] = {'seeds': sorted(w), 'flat': [sh(sum((w[s][y]['returns'] for y in YEARS), [])) for s in sorted(w)],
                    'tier': [sh(sum((w[s][y]['returns_tier'] for y in YEARS), [])) for s in sorted(w)]}
    any_s = sorted(w)[0]
    S['wf']['rule'] = {'flat': [sh(sum((w[any_s][y]['rule_flat'] for y in YEARS), []))],
                       'tier': [sh(sum((w[any_s][y]['rule_tier'] for y in YEARS), []))]}
print('wf', {k: (round(np.mean(v['flat']), 3), round(np.mean(v['tier']), 3), len(v['flat'])) for k, v in S['wf'].items()}, flush=True)

# so sánh cặp với PPO gốc cùng hạt giống (phí theo thanh khoản)
for per in S:
    o = dict(zip(S[per]['orig']['seeds'], S[per]['orig']['tier']))
    for cfg in ('tier', 'excess', 'long20'):
        if cfg not in S[per]: continue
        pr = [(b, o[s]) for s, b in zip(S[per][cfg]['seeds'], S[per][cfg]['tier']) if s in o]
        S[per][cfg]['paired_tier'] = {'n': len(pr), 'better': int(sum(a > b for a, b in pr)),
                                      'mean_diff': float(np.mean([a - b for a, b in pr]))}
        print(per, cfg, S[per][cfg]['paired_tier'], flush=True)
json.dump(S, open(os.path.join(R, 'v3.json'), 'w'), indent=1)

f = lambda x: f"{x:.2f}".replace('.', '{,}').replace('-', '$-$')
ms = lambda v: f(np.mean(v)) + (f" $\\pm$ {f(np.std(v))}" if len(v) > 1 else '')
names = [('orig', 'PPO gốc (thưởng DSR, phí 10 bps, 12 vòng)'), ('tier', '(A) Học với phí theo thanh khoản'),
         ('excess', '(B) Thưởng trên lợi suất vượt quy tắc, 20 vòng'), ('long20', 'Đối chứng: thưởng DSR, 20 vòng'),
         ('rule', 'Quy tắc bậc thấp (không học)')]
tb = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{Kiểm tra 2024--2025} & \\multicolumn{2}{c}{Cuốn chiếu 2020--2025}"
      " & \\multicolumn{2}{c}{Tập thanh khoản} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
      "Cấu hình & 10 bps & Theo TK & 10 bps & Theo TK & 10 bps & Theo TK \\\\\n\\midrule\n")
for k, lab in names:
    row = [lab]
    for per in ('test', 'wf', 'liq'):
        v = S[per].get(k)
        row += ['--', '--'] if v is None else [ms(v['flat']), ms(v['tier'])]
    tb += " & ".join(row) + " \\\\\n"
tb += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(TAB, 'v3.tex'), 'w').write(tb)
