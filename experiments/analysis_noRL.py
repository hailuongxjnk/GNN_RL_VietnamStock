"""
Tổng hợp các thí nghiệm "không RL" và chỉ số bỏ họ Vingroup → thesis/tables/norl.tex, vingroup.tex,
results/norl.json
"""
import os, glob, json, pickle
import numpy as np
import common as C

R = C.RESULTS; TAB = os.path.join(C.ROOT, 'thesis', 'tables')
f = lambda x, d=2: (f"{x:.{d}f}".replace('.', '{,}').replace('-', '$-$'))
stats = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)
years = [2020, 2021, 2022, 2023, 2024, 2025]
S = {}

def seeds(pattern):
    return [pickle.load(open(fn, 'rb')) for fn in sorted(glob.glob(os.path.join(R, pattern)))]

def wf_chain(cfg):
    by = {}
    for fn in glob.glob(os.path.join(R, 'wf', f'{cfg}_y*_seed*.pkl')):
        o = pickle.load(open(fn, 'rb')); by.setdefault(o['seed'], {})[o['year']] = o['test']
    out = {}
    for s, ys in by.items():
        if all(y in ys for y in years):
            r = sum((ys[y]['returns'] for y in years), [])
            out[s] = {'returns': r, 'turn': float(np.mean(sum((ys[y]['turnover'][1:] for y in years), [])))}
    return out

def mrow(lab, mets, turn, n):
    g = lambda k, s=1, d=2: f"{f(np.mean([m[k] for m in mets])*s, d)}" + (f" $\\pm$ {f(np.std([m[k] for m in mets], ddof=1)*s, d)}" if n > 1 else "")
    return f"{lab} & {n} & {g('sharpe')} & {g('cum_return',100,1)} & {g('ann_vol',100,1)} & {g('max_drawdown',100,1)} & {f(turn)} \\\\\n"

head = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\nChiến lược & Số lần & Sharpe & Lợi suất (\\%) & Biến động (\\%) & MDD (\\%) & Luân chuyển \\\\\n\\midrule\n")
ld = pickle.load(open(os.path.join(R, 'lowdeg_variants.pkl'), 'rb'))
nr = pickle.load(open(os.path.join(R, 'baselines_noRL.pkl'), 'rb'))
bl = pickle.load(open(os.path.join(R, 'baselines.pkl'), 'rb'))['M']
bw = pickle.load(open(os.path.join(R, 'baselines_wf.pkl'), 'rb'))

# ── giai đoạn chính 2024–2025 ──
ppo = seeds('tf_gat_long_seed*.pkl'); ppo60 = seeds('full_seed*.pkl'); sup = seeds('sup_seed*.pkl')
un = pickle.load(open(os.path.join(R, 'untrained_A250.pkl'), 'rb'))
tb = head
tb += mrow('\\textbf{Transformer + GAT + PPO} (đề xuất)', [o['test']['metrics'] for o in ppo], np.mean([np.mean(o['test']['turnover'][1:]) for o in ppo]), len(ppo))
tb += mrow('Transformer + GAT + học có giám sát', [o['test']['metrics'] for o in sup], np.mean([np.mean(o['test']['turnover'][1:]) for o in sup]), len(sup))
tb += mrow('Transformer + GAT chưa huấn luyện', [o['metrics'] for o in un], np.mean([o['turnover'] for o in un]), len(un))
tb += mrow('Quy tắc bậc thấp, đồ thị dài hạn', [ld['test_A250']['metrics']], np.mean(ld['test_A250']['turnover'][1:]), 1)
tb += mrow('Quy tắc bậc thấp, đồ thị ngắn hạn', [ld['test_A60']['metrics']], np.mean(ld['test_A60']['turnover'][1:]), 1)
tb += mrow('Phân bổ đều (1/N)', [bl['Equal-weight']['metrics']], np.mean(bl['Equal-weight']['turnover'][1:]), 1)
tb += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(TAB, 'norl_test.tex'), 'w').write(tb)
S['test'] = {'ppo': stats([o['test']['metrics']['sharpe'] for o in ppo]),
             'sup': stats([o['test']['metrics']['sharpe'] for o in sup]),
             'sup_values': [o['test']['metrics']['sharpe'] for o in sup],
             'sup_cum': stats([o['test']['metrics']['cum_return'] for o in sup]),
             'sup_turn': float(np.mean([np.mean(o['test']['turnover'][1:]) for o in sup])),
             'sup_val': stats([o['best_val_sharpe'] for o in sup]),
             'sup_beat_ew': int(sum(o['test']['metrics']['sharpe'] > bl['Equal-weight']['metrics']['sharpe'] for o in sup)),
             'lowdeg_A250': ld['test_A250']['metrics'], 'lowdeg_A60': ld['test_A60']['metrics']}
# trùng lặp mã giữa mô hình PPO và quy tắc bậc thấp
ov = []
for o in ppo:
    for w, sy, h in zip(o['test']['weights'], o['test']['symbols'], ld['test_A250']['held']):
        ov.append(len(set(np.array(sy)[w > 0]) & set(h)) / 20)
S['test']['overlap_ppo_lowdeg'] = float(np.mean(ov))

# ── cuốn chiếu ──
wfp, wfs = wf_chain('tf_gat_long'), wf_chain('sup')
unw = pickle.load(open(os.path.join(R, 'untrained_wf_A250.pkl'), 'rb'))
pm = lambda ch: [C.perf_metrics(v['returns'], 12) for v in ch.values()]
tb = head
tb += mrow('\\textbf{Transformer + GAT + PPO} (đề xuất)', pm(wfp), np.mean([v['turn'] for v in wfp.values()]), len(wfp))
tb += mrow('Transformer + GAT + học có giám sát', pm(wfs), np.mean([v['turn'] for v in wfs.values()]), len(wfs))
tb += mrow('Transformer + GAT chưa huấn luyện', [o['metrics'] for o in unw], np.mean([o['turnover'] for o in unw]), len(unw))
tb += mrow('Quy tắc bậc thấp, đồ thị dài hạn', [ld['wf_A250']['metrics']], np.mean(ld['wf_A250']['turnover'][1:]), 1)
tb += mrow('Quy tắc bậc thấp, đồ thị ngắn hạn', [ld['wf_A60']['metrics']], np.mean(ld['wf_A60']['turnover'][1:]), 1)
tb += mrow('Phân bổ đều (1/N)', [bw['Equal-weight']['metrics']], np.mean(bw['Equal-weight']['turnover'][1:]), 1)
tb += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(TAB, 'norl_wf.tex'), 'w').write(tb)
S['wf'] = {'ppo': stats([m['sharpe'] for m in pm(wfp)]), 'sup': stats([m['sharpe'] for m in pm(wfs)]),
           'sup_values': {int(k): C.perf_metrics(v['returns'], 12)['sharpe'] for k, v in wfs.items()},
           'sup_cum': stats([m['cum_return'] for m in pm(wfs)]), 'sup_mdd': stats([m['max_drawdown'] for m in pm(wfs)]),
           'n_sup': len(wfs), 'lowdeg_A250': ld['wf_A250']['metrics'], 'lowdeg_A60': ld['wf_A60']['metrics']}
ens = np.mean([v['returns'] for v in wfs.values()], axis=0); S['wf']['sup_ensemble'] = C.perf_metrics(ens, 12)
S['liq'] = {'lowdeg_A250': ld['liq_A250']['metrics'], 'lowdeg_A60': ld['liq_A60']['metrics']}

# ── bảng chỉ số bỏ họ Vingroup ──
ppo_m = [o['test']['metrics'] for o in ppo]
tb = ("\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{2024--2025} & \\multicolumn{3}{c}{2020--2025} \\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nChiến lược & Sharpe & Lợi suất (\\%) & MDD (\\%) & Sharpe & Lợi suất (\\%) & MDD (\\%) \\\\\n\\midrule\n")
def two(lab, a, b):
    return f"{lab} & {f(a['sharpe'])} & {f(a['cum_return']*100,1)} & {f(a['max_drawdown']*100,1)} & {f(b['sharpe'])} & {f(b['cum_return']*100,1)} & {f(b['max_drawdown']*100,1)} \\\\\n"
mean_m = lambda ms: {k: float(np.mean([m[k] for m in ms])) for k in ('sharpe', 'cum_return', 'max_drawdown')}
tb += two('\\textbf{Mô hình đề xuất} (trung bình)', mean_m(ppo_m), mean_m(pm(wfp)))
tb += two('VN-Index', C.perf_metrics(nr['test']['_vnindex'], 12), C.perf_metrics(nr['wf']['_vnindex'], 12))
for nm, lab in (('Vốn hoá (tự dựng)', 'Chỉ số vốn hóa tự dựng'), ('Vốn hoá, bỏ họ Vingroup', 'Chỉ số vốn hóa, bỏ họ Vingroup'),
                ('Phân bổ đều, bỏ họ Vingroup', 'Phân bổ đều, bỏ họ Vingroup')):
    tb += two(lab, nr['test'][nm]['metrics'], nr['wf'][nm]['metrics'])
tb += two('Phân bổ đều (1/N)', bl['Equal-weight']['metrics'], bw['Equal-weight']['metrics'])
tb += "\\bottomrule\n\\end{tabular}\n"
open(os.path.join(TAB, 'vingroup.tex'), 'w').write(tb)
VIN = {'VIC', 'VHM', 'VRE', 'VPL'}
vw = [np.mean([sum(w[i] for i, s in enumerate(sy) if s in VIN) for w, sy in zip(o['test']['weights'], o['test']['symbols'])]) for o in ppo]
S['vin'] = {'share_first_last_test': [nr['test']['_vin_share'][0], nr['test']['_vin_share'][-1]],
            'corr_proxy_vni': [nr['test']['_corr_proxy_vni'], nr['wf']['_corr_proxy_vni']],
            'model_vin_weight_range': [float(min(vw)), float(max(vw)), float(np.mean(vw))],
            'by_year_vin_ret': {}}
d = nr['test']['_dates']; vr = nr['test']['_vin_return']
for y in (2024, 2025):
    S['vin']['by_year_vin_ret'][y] = float(np.prod([1 + r for r, dd in zip(vr, d) if dd.year == y]) - 1)
json.dump(S, open(os.path.join(R, 'norl.json'), 'w'), indent=1, default=float)
print(json.dumps(S, indent=1, default=float))
