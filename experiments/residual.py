"""
Học tăng cường "học phần điều chỉnh" quanh quy tắc bậc thấp (Mục 4.10).

Mục 4.10 so sánh PPO học từ đầu với quy tắc không học "chọn các mã ít liên kết nhất trên đồ thị dài hạn"
(lowdeg_variants.py): PPO hơn quy tắc trên tập đầy đủ nhưng kém hẳn trên tập thanh khoản. Thí nghiệm này hỏi:
nếu cho PPO XUẤT PHÁT từ quy tắc đó và chỉ học phần điều chỉnh, nó có vượt được cả hai không?

Thiết kế (không sửa code gốc):
  - Mạng đồ thị PriorGAT: giống GAT gốc, nối thêm một cột "điểm ưu tiên" p_i = -(bậc_i chuẩn hoá z
    trên các mã đủ điều kiện), tính từ chính ma trận kề A250 tại ngày quyết định.
  - Tác tử ResidualAgent: điểm Dirichlet = beta * p_i + g * actor(Z_i), với beta = 3 cố định và hệ số g học được,
    khởi tạo g = 0,1. Lúc khởi tạo, 20 mã có tham số Dirichlet lớn nhất trùng với 20 mã bậc thấp nhất,
    tức chính sách xuất phát ≈ quy tắc (tỷ trọng tỉ lệ với tham số, trần 10%). PPO học actor và g.
  - Mọi thứ khác giống mô hình đề xuất: Transformer + GAT, A250, PPO 12 vòng, DSR, top-20, trần 10%, phí 10 bps,
    chọn mô hình theo Sharpe tập kiểm định. Tại vòng 0 (trước khi cập nhật) KHÔNG được chấm vào lựa chọn,
    giống mô hình đề xuất.
Chạy: cách chia chính (10 hạt giống), cuốn chiếu 2020–2025 (8 hạt giống), tập thanh khoản (5 hạt giống).
Kết quả: results/resid*_seed*.pkl, results/wf/resid_y*_seed*.pkl
"""
import io, os, sys, time, pickle, contextlib, warnings
from multiprocessing import Pool
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
warnings.filterwarnings('ignore')
import common as C
import run_experiments as RE
import run_extra as RX
from rl_agent import PortfolioAgent

BETA, G0 = 3.0, 0.1


class PriorGAT(C.gnn_mod.GAT):
    def forward(self, H, A):
        Z = super().forward(H, A)
        act = (A != 0).any(dim=1)
        deg = ((A != 0).sum(1) - (torch.diagonal(A) != 0).long()).float()
        p = torch.zeros(A.size(0))
        if act.sum() > 1:
            d = deg[act]; p[act] = -(d - d.mean()) / (d.std() + 1e-8)
        return torch.cat([Z, p.unsqueeze(-1)], dim=-1)


class ResidualAgent(PortfolioAgent):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(G0))

    def _alpha(self, Z):
        s = BETA * Z[:, -1] + self.g * self.actor(Z[:, :-1]).squeeze(-1)
        return F.softplus(s) + 1e-3

    def _value(self, Z, mask):
        return super()._value(Z[:, :-1], mask)


def models():
    m = C.make_models('transformer', 'gat')
    return {'encoder': m['encoder'], 'gnn': PriorGAT(), 'agent': ResidualAgent()}


def train(df, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    init = models()
    with contextlib.redirect_stdout(io.StringIO()):
        return C.train_mod.train_ppo(df, n_iters=RE.N_ITERS, freq='M', top_k=RE.TOP_K, w_max=RE.W_MAX,
                                     cost_rate=RE.COST, seed=seed, device='cpu', init=init, verbose=False)


def run_job(job):
    torch.set_num_threads(1)
    df = C.load_df(); t = time.time()
    kind = job[0]
    if kind == 'wf':
        _, year, seed = job; path = os.path.join(C.RESULTS, 'wf', f'resid_y{year}_seed{seed}.pkl')
        if os.path.exists(path): return job, 'skip'
        cd = C.CachedData(graph='A250', freq='M').install()
        sp = RX.wf_split(sorted(cd.snap.keys()), year)
        C.train_mod.split_dates = lambda df_, freq='M', **k: sp
        C.quiet_split = lambda df_, freq: sp
    else:
        graph = 'A250' if kind == 'main' else 'A250_liq'
        seed = job[1]; path = os.path.join(C.RESULTS, f"resid{'' if kind == 'main' else '_liq'}_seed{seed}.pkl")
        if os.path.exists(path): return job, 'skip'
        C.CachedData(graph=graph, freq='M').install()
    torch.manual_seed(seed); np.random.seed(seed)
    # chính sách xuất phát (trước mọi cập nhật), để biết PPO làm tốt hơn hay kém đi so với điểm xuất phát
    start = RE.backtest(models(), df, 'test')
    res = train(df, seed)
    out = {'config': 'resid', 'seed': seed, 'history': res['history'], 'best_val_sharpe': res['best_val_sharpe'],
           'test': RE.backtest(res, df, 'test'), 'start': start,
           'g': float(res['agent'].g.detach()), 'train_seconds': time.time() - t}
    if kind == 'wf': out['year'] = job[1]
    pickle.dump(out, open(path, 'wb'))
    return job, round(time.time() - t), round(out['start']['metrics']['sharpe'], 3), round(out['test']['metrics']['sharpe'], 3)


if __name__ == '__main__':
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    jobs = [('main', s) for s in range(1, 11)]
    jobs += [('wf', y, s) for y in sorted(RX.WF_YEARS, reverse=True) for s in range(1, 9)]
    jobs += [('liq', s) for s in range(1, 6)]
    print(len(jobs), 'jobs', flush=True)
    with Pool(n_proc, maxtasksperchild=1) as p:
        for r in p.imap_unordered(run_job, jobs):
            print(time.strftime('%H:%M:%S'), r, flush=True)
