# Bản sửa Methodology (Section 3) cho ICTA2026 submission #333

Ghi chú cho bạn (tiếng Việt) nằm trong khối `> NOTE:`. Phần còn lại là text
tiếng Anh, viết để copy thẳng vào file `.tex`, khớp với code thật trong repo
(không phải bản mô tả cũ trong PDF).

---

## Bảng ánh xạ: review comment → chỗ sửa

| # | Reviewer nói gì | Nguyên nhân thật | Sửa ở đâu |
|---|---|---|---|
| R1.1 | "Dynamic" nhưng đồ thị tả là fixed | Text sai — code build lại A mỗi decision date | §3.2 (bên dưới) |
| R1.2 | Temporal Attention thiếu công thức toán | Chưa từng viết ra — nhưng cơ chế đã tồn tại (self-attention trong TransformerEncoder) | §3.3b (mới, bên dưới) |
| R1.3 | Công thức là GCN, gọi là GAT | Text copy nhầm công thức GCN | §3.3a (bên dưới) |
| R1.4 | Algorithm 1 quá chung chung, thiếu PPO | Algorithm 1 cũ mô tả pretrain giám sát — không phải cái thật sự chạy | §3.4 (bên dưới) |
| R2.2 | N=25 chọn sao, có selection bias? | N thật KHÔNG cố định — universe ~200+ mã vốn hoá vừa/lớn, lọc bằng mask, mỗi kỳ giữ tối đa `top_k` mã | §3.1 + §3.2 (bên dưới) |
| R2.3 / R3.2 | Transaction cost: Intro nói có, Conclusion nói "future work" | Code ĐÃ trừ transaction cost vào reward — câu trong Conclusion là sai/lỗi thời | Xoá câu ở Conclusion, thêm công thức ở §3.4 |
| R3.3 | Bảng 3 (1.49/60.7) vs Bảng 4 "Full model" (1.54/58.2) lệch nhau | Rất có thể 2 bảng đến từ 2 lần chạy khác nhau, không cùng 1 checkpoint | Dùng lại đúng 1 bộ số từ `run_experiments.py` cho cả 2 bảng (xem file `experiment_results.json` sau khi chạy xong) |
| R3.4 | Hình 1 mâu thuẫn Bảng 3 | Cùng nguyên nhân R3.3 | Vẽ lại Hình 1 từ đúng run đã dùng cho Bảng 3 |
| — | (tôi tự phát hiện) Bảng 3 nói "tất cả 10 phương pháp chạy trên cùng data" nhưng code chỉ thật sự tính 3 (Agent/Equal-weight/VN-Index) | 7 baseline còn lại không có code trong repo | Xem mục "Baseline" cuối file — cần bạn quyết định trước khi nộp |

> **NOTE:** Tất cả các mục R1, R2.3, R3.2 ở trên chỉ cần **sửa câu chữ**, không
> cần sửa code, vì code hiện tại đã làm đúng những gì reviewer muốn thấy —
> nó chỉ chưa được MÔ TẢ đúng trong bài. R2.2, R3.3, R3.4 cần số liệu thật từ
> `run_experiments.py`.

---

## §3.1 Problem Formulation (thay thế đoạn N=25, T=30, F=7)

> We consider portfolio optimization as a sequential decision-making problem
> over a time-varying financial graph
> $$\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t, A_t),$$
> where each node $v_i \in \mathcal{V}_t$ represents one eligible stock at
> rebalancing date $t$, and $A_t \in \mathbb{R}^{N_t \times N_t}$ is the
> adjacency matrix re-estimated at every decision date (Section 3.2). The
> eligible universe is drawn from mid- and large-cap stocks listed on HOSE
> and HNX (market capitalization above VND 5,000 billion, updated
> semi-annually) that have been listed for at least 126 trading sessions
> (approx. 6 months); a binary eligibility mask $m_i(t) \in \{0,1\}$ enforces
> this filter, so $N_t = \sum_i m_i(t)$ varies over time (in our dataset,
> $N_t$ ranges from a handful of names early in the sample to over 200 in
> recent years). At each rebalancing date, the portfolio is restricted to at
> most $K=20$ positions (top-$K$ selection by policy weight) with a per-stock
> cap of $w_{\max}=10\%$, which keeps the realized portfolio size comparable
> across periods while avoiding a fixed, hand-picked stock list.
>
> For each eligible stock, we construct a multivariate historical feature
> sequence using a rolling window of $T=60$ trading days. At each time step,
> the feature vector contains $F=19$ variables drawn from three groups:
>
> - **Price/technical** (9): log return, log volume, 20-/60-day realized
>   volatility, 14-day RSI, 20-/60-day momentum, moving-average ratio,
>   20-day log Amihud illiquidity.
> - **Firm fundamentals** (5): P/E, P/B, trailing-4-quarter net income
>   growth, ROE, asset-to-equity ratio.
> - **Macroeconomic** (5, shared across all stocks on a given day): YoY
>   inflation, government bond yield, interbank rate, VN-Index return,
>   USD/VND rate.
>
> The input tensor is therefore $X_t \in \mathbb{R}^{N_t \times T \times F}$.
> Unlike a return-prediction formulation, our framework does not train a
> supervised next-day-return target; instead, the graph and temporal
> representations are optimized end-to-end together with the portfolio
> policy (Section 3.4), directly for risk-adjusted trading performance.

> **NOTE:** Đoạn "in our dataset, $N_t$ ranges..." — số chính xác (min/max/mean
> N_t) tôi sẽ điền từ `run_experiments.py` log (có in ra "Episodes train" và
> có thể tính thêm min/max N_t nếu bạn muốn — nói tôi thêm).

---

## §3.2 Graph Construction (thay thế đoạn "graph remains fixed... constructed once")

> To model inter-stock relationships, we construct a **rolling, look-ahead-safe**
> correlation graph. At every decision date $t$, let $r_i^{(t-T:t)}$ denote the
> log-return series of stock $i$ over the $T=60$ trading sessions strictly
> preceding $t$ (no information at or after $t$ is used). The pairwise Pearson
> correlation is
> $$\rho_{ij}(t) = \operatorname{corr}\big(r_i^{(t-T:t)},\, r_j^{(t-T:t)}\big),$$
> computed pairwise with a minimum-overlap requirement of 30 common
> observations (pairs with less overlap are treated as unconnected). The
> adjacency matrix is
> $$A_{ij}(t) = \begin{cases} |\rho_{ij}(t)|, & |\rho_{ij}(t)| > 0.30,\ i\neq j \\ 1, & i = j \\ 0, & \text{otherwise} \end{cases}$$
> Self-loops are added so that a stock's own representation always
> contributes to message passing. **Because $A(t)$ is re-estimated at every
> rebalancing date from the most recent window, the graph structure evolves
> over the sample period**, capturing structural breaks in the Vietnamese
> market (e.g., 2020 COVID shock, 2022 correction) rather than assuming a
> single time-invariant correlation structure. Edges touching an ineligible
> stock ($m_i(t)=0$) are masked out, so ineligible nodes neither send nor
> receive messages.

---

## §3.3a Graph Attention Network (thay công thức GCN bằng GAT thật)

> Node representations are refined by a two-layer Graph Attention Network
> (GAT). Let $h_i^{(l)}$ denote the representation of stock $i$ at layer $l$
> (with $h_i^{(0)}$ given by the temporal encoder, Section 3.3b). For each
> attention head $k=1,\dots,K_h$ ($K_h=4$), an unnormalized attention score is
> computed for every edge $(i,j)$ present in $A(t)$:
> $$e_{ij}^{(k)} = \mathrm{LeakyReLU}\big(\mathbf{a}^{(k)}_{\text{src}} \cdot W^{(k)} h_i^{(l)} + \mathbf{a}^{(k)}_{\text{dst}} \cdot W^{(k)} h_j^{(l)}\big),$$
> normalized over the neighborhood $\mathcal{N}(i)$ (including the self-loop)
> by softmax,
> $$\alpha_{ij}^{(k)} = \frac{\exp(e_{ij}^{(k)})}{\sum_{j' \in \mathcal{N}(i)} \exp(e_{ij'}^{(k)})},$$
> and aggregated as
> $$h_i^{(l+1),k} = \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)}\, W^{(k)} h_j^{(l)}.$$
> The first GAT layer concatenates all $K_h=4$ heads (hidden size 16 per
> head, 64 total) followed by an ELU nonlinearity; the second layer averages
> a single head to produce the final 64-dimensional stock embedding
> $z_i(t)$, followed by layer normalization. Unlike a graph convolution, the
> edge weights $\alpha_{ij}^{(k)}$ are **learned** rather than fixed by the
> (symmetric-normalized) degree of the nodes, allowing the model to
> emphasize the most informative neighbors at each decision date.

---

## §3.3b Temporal Attention (module mới — bổ sung công thức cho phần đã tồn tại nhưng chưa viết ra)

> Historical market information is encoded using a Transformer-style temporal
> attention module. Given the $T=60$-day feature sequence of stock $i$,
> $x_i^{(1:T)} \in \mathbb{R}^{T\times F}$, each step is linearly projected to
> $d_{\text{model}}=64$ dimensions and combined with a fixed sinusoidal
> positional encoding. A causally-masked multi-head self-attention layer
> (4 heads, 2 layers, feed-forward width 256) computes, for each head,
> $$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V,$$
> where $M$ is a causal mask preventing position $t$ from attending to future
> steps $t' > t$, and $Q,K,V$ are learned linear projections of the
> sequence. The resulting sequence representation is mean-pooled over time
> and passed through a linear layer and layer normalization to produce the
> initial stock embedding $h_i^{(0)} \in \mathbb{R}^{64}$ used as input to the
> GAT (Section 3.3a). This mechanism allows the model to weight historical
> observations non-uniformly rather than relying on a fixed recurrent state,
> directly addressing which historical time steps are most informative for
> the current decision date.

> **NOTE:** Nếu reviewer muốn nhìn thấy trọng số attention theo TỪNG timestep
> (không chỉ mean-pool), tôi có thể thêm một attention-pooling layer thật
> (thay mean-pool bằng weighted-pool có trọng số học được) — việc này CẦN
> sửa code (`temporal_encoder.py`), không chỉ sửa text. Nói tôi nếu muốn làm.

---

## §3.4 Training Procedure (thay Algorithm 1 pretrain-MSE bằng PPO thật)

> Unlike a two-stage pipeline that pretrains graph representations via
> return regression before policy optimization, our framework is trained
> **end-to-end**: the temporal encoder, GAT, and policy/value networks are
> optimized jointly to directly maximize risk-adjusted trading performance,
> with no auxiliary return-prediction objective.
>
> **Reward.** At each rebalancing step, given portfolio weights $w_t$ (from
> the policy, projected onto the eligible simplex with the top-$K$ and
> $w_{\max}$ constraints) and the realized holding-period return $R_t$ of
> each stock, the net portfolio return after transaction costs is
> $$r_t = w_t^\top R_t \;-\; c \cdot \lVert w_t - w_{t-1} \rVert_1,$$
> where $c=10\text{ bps}$ is a proportional transaction cost applied to
> portfolio turnover. The reward signal used by PPO is the Differential
> Sharpe Ratio (Moody & Saffell), an online, incrementally-updated estimate
> of the marginal contribution of $r_t$ to the running Sharpe ratio,
> which allows the agent to optimize risk-adjusted return step-by-step
> rather than only at the end of an episode.
>
> **Algorithm.** Training alternates between a rollout phase and a PPO update
> phase, as summarized below.

```
Algorithm 1: End-to-end training of the proposed framework
Require: Panel data D, transaction cost rate c, PPO clip ε, epochs per
         update E_ppo, learning rate η, iterations N_iter
Ensure: Trained encoder, GAT, and policy/value parameters θ*
 1: Initialize temporal encoder f_enc, GAT f_gat, actor-critic π_θ
 2: for iteration = 1 to N_iter do
 3:     // --- Rollout (no gradient) ---
 4:     for each decision date t in the training period do
 5:         Build snapshot X_t, mask m_t; build rolling adjacency A_t (§3.2)
 6:         h_t ← f_enc(X_t);  z_t ← f_gat(h_t, A_t)                 // §3.3
 7:         Sample w_t ~ Dirichlet(π_θ.actor(z_t)) over eligible stocks
 8:         Compute r_t (net return incl. transaction cost, above)
 9:         Compute DSR reward d_t from r_t; store (z_t, w_t, log p(w_t), V(z_t), d_t)
10:     end for
11:     Compute advantages Â_t via Generalized Advantage Estimation (γ=0.99, λ=0.95)
12:     // --- PPO update (E_ppo epochs, full end-to-end backward) ---
13:     for epoch = 1 to E_ppo do
14:         for each stored step t do
15:             Recompute z_t = f_gat(f_enc(X_t), A_t)               // gradient flows to encoder+GAT
16:             ratio_t ← exp(log π_θ(w_t|z_t) − log π_θ_old(w_t|z_t))
17:             L_actor ← −min(ratio_t Â_t, clip(ratio_t, 1−ε, 1+ε) Â_t)
18:             L_critic ← (V_θ(z_t) − (Â_t + V_old(z_t)))²
19:             L ← L_actor + c_v L_critic − c_e H[π_θ]
20:             Backpropagate L through π_θ, f_gat, f_enc; accumulate gradient
21:         end for
22:         Clip gradient norm; update θ with Adam(η)
23:     end for
24:     Evaluate validation Sharpe; keep θ* = best validation checkpoint
25: end for
26: return θ*
```

> **NOTE:** Algorithm 1 mới này khớp 1-1 với `train.py` thật (`train_ppo`,
> `DSR`, `compute_gae`). Không có `graph regularization loss` hay `temporal
> consistency loss` nữa — vì code không có 2 loss đó. Nếu bạn/thầy cô muốn
> GIỮ 2 loss này trong bài (để match với ablation "w/o GNN pre-training"),
> thì đây là việc cần sửa CODE thật (thêm 2 penalty term vào `loss_ep` trong
> `train.py`), không phải chỉ sửa text — nói tôi nếu muốn làm hướng này thay
> vì hướng "bỏ 2 loss, mô tả đúng code hiện tại" mà tôi đang khuyến nghị.

---

## Bảng 1 & 2 — giá trị cần sửa lại

| Tham số | Giá trị cũ (sai) | Giá trị đúng |
|---|---|---|
| Historical window | 30 trading days | **60** trading days |
| Number of stocks | 25 (cố định) | **biến động theo tháng**, xem §3.1; top-K giữ vị thế = **20** |
| Node features | 7 | **19** (9 kỹ thuật + 5 tài chính + 5 vĩ mô) |
| Graph construction | "constructed once" | **rolling, rebuilt every decision date** |
| Learning rate | $1\times10^{-4}$ | **$3\times10^{-4}$** (giá trị thật trong `train_ppo`) |
| Graph regularization $\lambda_{graph}$ | 0.01 | **bỏ dòng này** (không tồn tại trong code) |
| Temporal consistency $\lambda_{TC}$ | 0.01 | **bỏ dòng này** (không tồn tại trong code) |
| Train/Val/Test | 75% / 12.5% / 12.5% | theo mốc thời gian thật trong `split_dates` (freq đang dùng là tuần/tháng — xem log chạy thật) |

---

## Fix 1 câu ở Introduction / Conclusion (transaction cost contradiction — R2.3, R3.2)

- **Conclusion**, xoá câu: *"assumes fixed transaction costs... Future work will investigate dynamic transaction cost modeling"* → transaction cost (10 bps trên turnover) đã được trừ trực tiếp vào reward mỗi bước (công thức ở §3.4 trên), không phải "future work". Nếu muốn giữ ý "dynamic cost", có thể sửa thành: *"the current cost model assumes a constant proportional rate; modeling cost as a function of trade size or market liquidity is left for future work."* — vẫn trung thực và giữ được câu văn.

---

## Vấn đề CẦN BẠN QUYẾT ĐỊNH trước khi nộp: Bảng 3 (10 baseline)

Code hiện tại chỉ tính thật **3** dòng: Agent (GNN+RL), Equal-weight (1/N),
VN-Index — xem `evaluate.py`. 7 dòng còn lại (Mean-Variance, LSTM, GRU,
FinRL-PPO, DeepPocket, AlphaGAT, NGDRL, TC-MAC, WCG-RL) **không có code
tương ứng** trong repo hay notebook.

Hai lựa chọn thực tế:

1. **Rút gọn Bảng 3** chỉ còn 3-4 dòng chạy thật (thêm Mean-Variance thì tôi
   viết thêm được, không khó) + để riêng một câu kiểu: *"Numbers for
   DeepPocket, AlphaGAT, NGDRL, TC-MAC, and WCG-RL are as reported in their
   original publications on different datasets/periods and are included for
   context only; they are not directly comparable under identical
   conditions."* — đây là cách làm phổ biến, an toàn, KHÔNG vi phạm liêm
   chính khoa học miễn là ghi rõ.
2. **Xoá câu** "all competing methods are implemented using the same
   dataset... to ensure a fair comparison" nếu giữ nguyên bảng — nhưng nếu
   giữ câu này mà không thật sự chạy lại 7 phương pháp kia, đây là rủi ro
   lớn nhất trong toàn bộ bài, nặng hơn nhiều so với 3 ý reviewer đã nêu.

> **NOTE:** Tôi khuyến nghị phương án 1. Cho tôi biết để tôi viết thêm
> Mean-Variance baseline thật (dễ, dùng ngay `evaluate.py`/`train.py` đã
> có) và cập nhật bảng.
