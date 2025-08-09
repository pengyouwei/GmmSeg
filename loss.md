# 损失函数推导详解

本文推导项目中两种方案的目标函数：

1. 原始方案（GmmLoss）：期望形式的重建项 + 两个 KL 正则（`KL(π||d)` 与 `KL(Dir(d)||Dir(d0))`）。
2. Dirichlet 混合方案（DirichletGmmLoss）：真实混合模型 NLL 或其 warmup 上界 + `KL(Dir(d)||Dir(d0))` + 可选熵正则。

记号对照（代码 → 数学）：

| 代码 | 形状 | 数学符号 | 含义 |
|------|------|----------|------|
| `x` | (C,H,W) | x | 观测（特征空间像素向量）|
| `μ_k` (`mu`) | (K,C,H,W) | μ_k | 第 k 个高斯分量均值 |
| `σ_k^2` (`var`) | (K,C,H,W) | σ_k^2 | 第 k 个高斯分量对角方差 |
| `π_k` (`pi`) | (K,H,W) | π_k | gating / q(z=k|x) 近似后验（方案1）或由 d 期望得到的先验加权（方案2 warmup）|
| `d_k` (`d`) | (K,H,W) | α_k | Dirichlet 浓度参数（网络输出）|
| `d0_k` (`d0`) | (K,H,W) | α^0_k | 注册得到的先验 Dirichlet 浓度 |
| `r_k` (`post_prob` in DirichletGmmLoss) | (K,H,W) | r_k | 后验责任 p(z=k | x)（方案2 NLL 梯度内隐含，显式计算用于预测）|

以下推导均对单个空间位置 (H,W) 给出；整图为所有像素独立（条件于参数）之和 / 平均。

---

## 1. 原始方案（GmmLoss）

### 1.1 生成模型假设

给定隐变量：

1. 结构层（空间先验）Ω → 用 Dirichlet 分布参数化，其在代码中通过注册先验 `d0` 与网络输出 `d` 近似。
2. 像素类别 z ∈ {1..K}，条件于 Ω： p(z=k | Ω) = ψ_k(Ω)。在实现中用 `d` 诱导的 **期望概率** 近似，即：
   \[ p(z=k|Ω) ≈ \mathbb{E}_{Dir(d)}[π_k] = d_k / \sum_j d_j. \]
3. 像素观测 x | z=k, Ω ~ N( μ_k , diag(σ_k^2) ).

### 1.2 变分分布设定

采用分解： \( q(z, Ω | x) = q(z|x) q(Ω|x) \)。

代码中：
* `pi` 直接作为 \( q(z|x) \)（softmax 输出）。
* `d` 作为 \( q(Ω|x) \) 的 Dirichlet 浓度 α（即 q(Ω)=Dir(α)）。

### 1.3 ELBO 基本形式

参考附件 loss.png 中第二行分解（对 Ω 期望之后）：
\[
\mathcal{L} = - \mathbb{E}_{q(z,Ω|x)} \log p(x|z,Ω) - \mathbb{E}_{q(Ω|x)} KL\big(q(z|x) || p(z|Ω)\big) - KL\big(q(Ω|x) || p(Ω)\big)
\]
我们具体化：

1. 重建项（期望形式，上界于真实 NLL）：
   \[
   \mathcal{L}_{rec} = - \sum_{k} q(z=k|x) \; \log \mathcal{N}(x; μ_k, σ_k^2) = - \sum_k π_k \log N_k.
   \]
   代码中：`reconstruction_loss`：先计算 `log_gauss_k = log N_k`，再做 `sum_k π_k log_gauss_k` 取反与均值。

2. 分类 KL：
   \[
   \mathcal{L}_{kl\_pi} = \mathbb{E}_{q(Ω|x)} KL( q(z|x) || p(z|Ω) ).
   \]
   用 `d` 的期望近似 p(z|Ω)： \( p(z=k|Ω) ≈ d_k / \sum_j d_j \)。于是：
   \[
   KL(q||p) = \sum_k π_k ( \log π_k - \log \tilde{p}_k ), \quad \tilde{p}_k= d_k / \sum_j d_j.
   \]
   代码中用 digamma(α) - digamma(Σα) 的期望来更精确近似 `E_{Dir} [log p(z|Ω)]`：
   \[
     E_{Dir(d)}[\log p(z=k|Ω)] = ψ(d_k) - ψ(\sum_j d_j)
   \]
   故：
   \[
   \mathcal{L}_{kl\_pi} = \sum_k π_k ( \log π_k - (ψ(d_k) - ψ(\sum_j d_j)) ).
   \]

3. Dirichlet KL：
   \[
   \mathcal{L}_{kl\_dir} = KL( Dir(d) || Dir(d0) ).
   \]
   闭式：
   \[
   KL = \log B(d_0) - \log B(d) + \sum_k (d_k - d^0_k) ( ψ(d_k) - ψ(\sum_j d_j) )
   \]
   代码中函数 `kl_dirichlet_loss` 与此一致。

4. 轻微 MSE 正则（工程项）：`MSE(d, d0)*λ`，非标准 ELBO 一部分，但有助收敛。

5. 动态权重：仅对 `kl_dir` 乘随 epoch 递减因子 w₃，平衡早期先验影响。

### 1.4 总损失（实现）
\[
\mathcal{L}_{total} = \mathcal{L}_{rec} + \mathcal{L}_{kl\_pi} + w_3 \mathcal{L}_{kl\_dir} + \lambda_{mse} \| d-d0 \|_2^2.
\]

注意：使用期望形式 \( \sum π_k (-\log N_k) \) 是对真实混合 NLL 的 Jensen 上界：
\[
 -\log \sum_k π_k N_k(x) \le \sum_k π_k (-\log N_k(x)).
\]
因此该方案的重建项更“平滑”，但会弱化组件竞争。

---

## 2. Dirichlet 混合方案（DirichletGmmLoss）

### 2.1 设计动机
去掉显式 gating 网络（或忽略其输出），直接令 Dirichlet 浓度 `d` 同时承担：
1. 提供 **混合权重期望** \( π_k = d_k / \sum_j d_j \)。
2. 与先验浓度 `d0` 进行 KL 正则保持空间结构。

真实生成模型：
\[
π | Ω ~ Dir(d), \quad z|π ~ Cat(π), \quad x|z=k ~ N(μ_k, σ_k^2).
\]
我们使用点估计 π=E[π|d] 近似，或直接在 NLL 推导中使用它；后验责任：
\[
 r_k(x) = \frac{π_k N_k(x)}{\sum_j π_j N_j(x)}.
\]

### 2.2 真实混合的负对数似然 (NLL)
单像素：
\[
 \mathcal{L}_{nll} = -\log \sum_{k=1}^K π_k \mathcal{N}(x; μ_k, σ_k^2).
\]
实现中用 `logsumexp`：
```
log_mix = log π_k + log N_k
log_sum = logsumexp_k(log_mix)
recon = - mean(log_sum)
```

### 2.3 Warmup 上界（可选）
前 `T_w` 轮用期望形式：
\[
 \mathcal{L}_{rec}^{upper} = - \sum_k π_k \log N_k \ge \mathcal{L}_{nll}.
\]
平滑初期梯度，随后切换到真实 NLL 以增强分量竞争与区分度。

### 2.4 Dirichlet KL 保持空间先验
同方案1：
\[
 \mathcal{L}_{kl\_dir} = KL( Dir(d) || Dir(d0) ).
\]

### 2.5 熵正则（可选）
为避免组件塌缩到单一分量，引入：
\[
 \mathcal{L}_{entropy} = - \beta H(π) = -\beta ( - \sum_k π_k \log π_k ) = \beta \sum_k π_k \log π_k.
\]
代码中实现为 **奖励熵**（添加 `+ 0.001 * (-Σ π log π)` 等价于减小上式）。

### 2.6 总损失（实现）
\[
 \mathcal{L}_{total} = \mathcal{L}_{nll/upper} + \mathcal{L}_{kl\_dir} + \gamma (- H(π)).
\]
其中：
* warmup：使用上界 `upper`；否则使用 `nll`；
* `γ` 很小（如 0.001），仅激活闲置组件。

### 2.7 与原方案的结构对比
| 方面 | 方案1 | 方案2 |
|------|-------|-------|
| 重建 | Σ π_k (-log N_k) （上界） | -log Σ π_k N_k （真实） |
| gating 来源 | 专门 z_net 输出 softmax | 由 Dirichlet 浓度 d 期望得到 |
| KL(π||·) | 有：KL(π||E[p(z|Ω)]) | 无（隐式在 NLL 竞争） |
| KL(Dir) | KL(Dir(d)||Dir(d0)) | 同 |
| 熵/辅助 | 可选（未内置） | 轻微熵正则避免塌缩 |
| 责任 r | 未显式（π 即 q(z|x)） | 显式 r=posterior |

### 2.8 r 与 π 的关系
\[
 r_k = \frac{π_k e^{\log N_k}}{\sum_j π_j e^{\log N_j}} = \text{softmax}_k( \log π_k + \log N_k ).
\]
当各分量 N_k 相差不大时 r≈π；随着 μ,σ 分化，r 更“尖锐”，利于分割边界清晰。

---

## 3. 两种方案的 ELBO 视角总结

| 项目 | 方案1 对应 | 方案2 对应 |
|------|-----------|-----------|
| 数据项 | \( -E_{q(z|x)} \log p(x|z) \) | \( -\log \sum_k π_k p(x|z=k) \) （更紧） |
| z 的 KL | \( KL(q(z|x)||p(z|Ω)) \) 显式 | 隐式通过真实似然竞争 |
| Ω 的 KL | \( KL(q(Ω|x)||p(Ω)) \) → KL(Dir(d)||Dir(d0)) | 同 |
| Jensen 关系 | 使用上界 | 最终使用真实，对应更低界值 |

方案2 实际是在近似：最大化 \( \log p(x) - KL(Dir(d)||Dir(d0)) - γ(-H(π)) \) 的正则化形式，其中 π 由 d 决定，未显式对 z 的 KL 拆分。

---

## 4. 数值稳定与实现要点

1. `logsumexp` 计算 NLL：用减去最大值避免溢出。
2. 方差下界：`var.clamp_min(1e-8)` 防止除 0 / log 0。
3. Dirichlet 浓度下界：`d.clamp_min(1e-6)` 避免 digamma 奇异与 KL 爆炸。
4. 责任 r 的计算使用对数空间再 softmax：\( r = softmax(\log π + \log N) \)。
5. Warmup：期望形式 → 真实 NLL，缓解早期 μ,σ 未成形时的梯度噪声。

---

## 5. 何时选择哪种方案

| 场景 | 建议 |
|------|------|
| 需要与文献上标准 ELBO 对齐、保留可解释 q(z|x) | 方案1 |
| 更关注分量竞争力与精细边界 | 方案2 (NLL) |
| 训练初期不稳定 | 方案2 + warmup（或先用方案1 再切换） |
| 想减少网络模块 | 方案2（可移除 z_net） |

---

## 6. 代码字段与日志监控建议

| 字段 | 方案1 | 方案2 |
|------|-------|-------|
| `recon` | 期望重建 | NLL 或上界（warmup） |
| `kl_pi` | KL(π||p) | 空（缺省为 0） |
| `kl_dir` | KL(Dir(d)||Dir(d0)) | 同 |
| `post_prob` | π | r（后验责任） |
| `pi` (仅方案2返回) | — | gating 期望 π |
| `used_nll` | — | 是否已切换到真实 NLL |

监控：
* 组件利用率：`r` 的平均直方分布；
* 熵：`H(r)`；
* KL 曲线：`kl_dir` 随 epoch 收敛趋势；
* 方差统计：min / median 检查塌缩。

---

## 7. 小结

* 方案1 基于 Jensen 上界的经典分解：重建(期望) + 2 个 KL；结构清晰、解释性强。
* 方案2 直接优化真实混合似然（更紧），用 Dirichlet 浓度统一表示：更紧凑，梯度更聚焦，需注意组件塌缩风险，已通过熵/ warmup 缓解。
* 后验责任 r = softmax(log π + log N) 是最终更合理的像素分类依据；方案2 中已显式输出。

---

若后续需要再加入“显式 KL(r || E[p(z|Ω)])”或“分量方差正则”推导，可在本文基础上补充增量章节。
