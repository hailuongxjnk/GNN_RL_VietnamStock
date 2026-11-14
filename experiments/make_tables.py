"""Sinh các bảng LaTeX của Chương 4 từ results/summary.json → thesis/tables/*.tex"""
import json, os
import common as C
S = json.load(open(os.path.join(C.RESULTS, 'summary.json')))
OUT = os.path.join(C.ROOT, 'thesis', 'tables'); os.makedirs(OUT, exist_ok=True)
f = lambda x, d=2: (f"{x:.{d}f}").replace('.', '{,}').replace('-', '$-$')
pm = lambda a, k, s=1, d=2: f"{f(a[k]['mean']*s, d)} $\\pm$ {f(a[k]['std']*s, d)}"
def write(name, body):
    open(os.path.join(OUT, name + '.tex'), 'w').write(body); print('→', name)

NAMES = {'full': 'Transformer + GAT (đề xuất)', 'lstm_gat': 'LSTM + GAT', 'tf_gcn': 'Transformer + GCN',
         'tf_nograph': 'Transformer, không đồ thị', 'tf_gat_long': 'Transformer + GAT, đồ thị 250 phiên',
         'lstm_nograph': 'LSTM, không đồ thị', 'gru_nograph': 'GRU, không đồ thị'}
cfg = S['configs']
def row_cfg(c, label=None):
    a = cfg[c]
    return (f"{label or NAMES[c]} & {len(a['seeds'])} & {pm(a,'sharpe')} & {pm(a,'cum_return',100,1)} & "
            f"{pm(a,'ann_vol',100,1)} & {pm(a,'max_drawdown',100,1)} & {pm(a,'calmar')} & {f(a['turnover']['mean'])} \\\\")
def row_base(n, m, label=None):
    return (f"{label or n} & -- & {f(m['sharpe'])} & {f(m['cum_return']*100,1)} & {f(m['ann_vol']*100,1)} & "
            f"{f(m['max_drawdown']*100,1)} & {f(m['calmar'])} & {f(m['turnover'])} \\\\")
def row_rand(r, label):
    g = lambda k, s=1, d=2: f"{f(r[k]['p50']*s, d)} [{f(r[k]['p5']*s, d)}; {f(r[k]['p95']*s, d)}]"
    return f"{label} & 1000 & {g('sharpe')} & {g('cum_return',100,1)} & -- & {g('max_drawdown',100,1)} & {g('calmar')} & -- \\\\"
HEAD = ("\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\nChiến lược & Số lần & Sharpe & Lợi suất (\\%) & Biến động (\\%) & MDD (\\%) & Calmar & Luân chuyển \\\\\n\\midrule\n")
B = S['baselines_M']
body = HEAD
if 'full' in cfg: body += row_cfg('full', '\\textbf{Mô hình đề xuất}') + "\n\\midrule\n"
body += "\n".join(row_base(n, B[n], l) for n, l in [('Equal-weight', 'Phân bổ đều (1/N)'), ('VN-Index', 'VN-Index (mua và nắm giữ)'),
        ('Min-Variance', 'Phương sai nhỏ nhất'), ('Mean-Variance', 'Trung bình--phương sai'), ('Momentum top-20', 'Động lượng top-20')])
body += "\n\\midrule\n" + row_rand(S['random_random20_hold_M'], '20 mã ngẫu nhiên, giữ nguyên') + "\n" + row_rand(S['random_random20_M'], '20 mã ngẫu nhiên, đổi mỗi kỳ')
for c in ('lstm_nograph', 'gru_nograph'):
    if c in cfg: body += ("\n\\midrule\n" if c == 'lstm_nograph' else "\n") + row_cfg(c, NAMES[c].replace(', không đồ thị', ' thuần (không đồ thị) + PPO'))
body += "\n\\bottomrule\n\\end{tabular}\n"
write('overall', body)

body = HEAD.replace('Chiến lược', 'Cấu hình')
for c in ['full', 'lstm_gat', 'tf_gcn', 'tf_nograph', 'tf_gat_long']:
    if c in cfg: body += row_cfg(c) + "\n"
body += "\\bottomrule\n\\end{tabular}\n"
write('ablation', body)

if 'full_per_seed' in S:
    body = ("\\begin{tabular}{@{}cccccccccc@{}}\n\\toprule\nHạt giống & Vòng được chọn & Sharpe KĐ & Sharpe KT & SE (Lo) & Lợi suất (\\%) & MDD (\\%) & "
            "$t$ so với 1/N & Phân vị ngẫu nhiên \\\\\n\\midrule\n")
    for p in S['full_per_seed']:
        body += (f"{p['seed']} & {p['best_iter']+1} & {f(p['val_sharpe'])} & {f(p['sharpe'])} & {f(p['lo_se'])} & {f(p['cum']*100,1)} & "
                 f"{f(p['mdd']*100,1)} & {f(p['t_vs_ew'])} & {f(p['pct_random_hold']*100,0)}\\% \\\\\n")
    a = cfg['full']
    body += (f"\\midrule\nTrung bình & -- & {f(a['val_sharpe']['mean'])} & {f(a['sharpe']['mean'])} & -- & {f(a['cum_return']['mean']*100,1)} & "
             f"{f(a['max_drawdown']['mean']*100,1)} & -- & -- \\\\\n\\bottomrule\n\\end{{tabular}}\n")
    body = body.replace("\\begin{tabular}{@{}cccccccccc@{}}", "\\begin{tabular}{@{}ccccccccc@{}}")
    write('perseed', body)
    body = ("\\begin{tabular}{@{}cccccc@{}}\n\\toprule\nHạt giống & $\\bar\\alpha$ & Hệ số biến thiên của $\\alpha$ & Tỷ trọng nhỏ nhất--lớn nhất (\\%) & "
            "Tỷ lệ mã giữ lại & Luân chuyển \\\\\n\\midrule\n")
    for b, p in zip(S['behaviour']['per_seed'], S['full_per_seed']):
        body += (f"{b['seed']} & {f(b['alpha_mean'])} & {f(b['alpha_cv']*100,1)}\\% & {f(b['w_min']*100,1)}--{f(b['w_max']*100,1)} & "
                 f"{f(b['retention']*100,0)}\\% & {f(p['turnover'])} \\\\\n")
    body += "\\bottomrule\n\\end{tabular}\n"
    write('behaviour', body)

W = S['baselines_W']; a = S['weekly_checkpoint']['agent']
body = ("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\nChiến lược & Sharpe & Lợi suất (\\%) & Biến động (\\%) & MDD (\\%) & Calmar \\\\\n\\midrule\n"
        f"Mô hình đề xuất (1 hạt giống, 30 vòng) & {f(a['sharpe'])} & {f(a['cum_return']*100,1)} & {f(a['ann_vol']*100,1)} & {f(a['max_drawdown']*100,1)} & {f(a['calmar'])} \\\\\n\\midrule\n")
for n, l in [('Equal-weight', 'Phân bổ đều (1/N)'), ('VN-Index', 'VN-Index (mua và nắm giữ)'), ('Min-Variance', 'Phương sai nhỏ nhất'),
             ('Mean-Variance', 'Trung bình--phương sai'), ('Momentum top-20', 'Động lượng top-20')]:
    m = W[n]; body += f"{l} & {f(m['sharpe'])} & {f(m['cum_return']*100,1)} & {f(m['ann_vol']*100,1)} & {f(m['max_drawdown']*100,1)} & {f(m['calmar'])} \\\\\n"
r = S['random_random20_hold_W']
body += (f"20 mã ngẫu nhiên, giữ nguyên (trung vị) & {f(r['sharpe']['p50'])} & {f(r['cum_return']['p50']*100,1)} & -- & {f(r['max_drawdown']['p50']*100,1)} & {f(r['calmar']['p50'])} \\\\\n"
         "\\bottomrule\n\\end{tabular}\n")
write('weekly', body)

# ── Phụ lục: thống kê mô tả đặc trưng và kết quả từng lần chạy ──
D = json.load(open(os.path.join(C.RESULTS, 'data_stats.json')))
body = ("\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\nĐặc trưng & Trung bình & Độ lệch chuẩn & Nhỏ nhất & Trung vị & Lớn nhất & Thiếu (\\%) \\\\\n\\midrule\n")
def g(x):
    ax = abs(x)
    s = f"{x:.0f}" if ax >= 1000 else (f"{x:.1f}" if ax >= 100 else (f"{x:.2f}" if ax >= 1 else f"{x:.3f}"))
    return s.replace('.', '{,}').replace('-', '$-$')
for k, v in D['describe_eligible'].items():
    body += (f"\\texttt{{{k.replace('_', chr(92)+'_')}}} & {g(v['mean'])} & {g(v['std'])} & {g(v['min'])} & {g(v['50%'])} & {g(v['max'])} & "
             f"{f(D['missing_eligible'][k]*100,1)} \\\\\n")
body += "\\bottomrule\n\\end{tabular}\n"
write('features_desc', body)

import pickle, glob
body = ("\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\nCấu hình & Hạt giống & Vòng chọn & Sharpe KĐ & Sharpe KT & Lợi suất (\\%) & MDD (\\%) & Calmar \\\\\n\\midrule\n")
for c in ['full', 'lstm_gat', 'tf_gcn', 'tf_nograph', 'tf_gat_long', 'lstm_nograph', 'gru_nograph']:
    fs = sorted(glob.glob(os.path.join(C.RESULTS, f'{c}_seed*.pkl')))
    for i, fn in enumerate(fs):
        r = pickle.load(open(fn, 'rb')); m = r['test']['metrics']
        import numpy as np
        bi = int(np.nanargmax([h['val_sharpe'] for h in r['history']])) + 1
        body += (f"{NAMES[c] if i == 0 else ''} & {r['seed']} & {bi} & {f(r['val']['metrics']['sharpe'])} & {f(m['sharpe'])} & "
                 f"{f(m['cum_return']*100,1)} & {f(m['max_drawdown']*100,1)} & {f(m['calmar'])} \\\\\n")
    if fs: body += "\\midrule\n"
body = body.rstrip().rsplit("\\midrule", 1)[0] + "\\bottomrule\n\\end{tabular}\n"
write('allruns', body)
if 'untrained' in S:
    pass
