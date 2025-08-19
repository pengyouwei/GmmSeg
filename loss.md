## GMM 变分推断损失（ELBO）说明

本文档整理高斯混合模型 (GMM) 在变分贝叶斯 (Variational Bayes, VB) 框架下的目标函数（负 ELBO）各组成部分，统一符号并格式化公式，便于阅读与实现。

---

## 1. 观测与隐变量
* 观测数据：$x = \{x_i\}_{i=1}^N$
* 隐含指派变量：$z_i = (z_{i1}, \dots, z_{iK})$, 其中 $z_{ik} \in \{0,1\}$ 且 $\sum_k z_{ik}=1$
* 组件参数（此处抽象聚合记为）$\Omega = \{ w_{ik} \}$ 或其先验超参数集合

---

## 2. 似然 (Complete-Data Likelihood)
假设条件独立与高斯成分：

$$
P(x \mid z, \Omega) = \prod_{i=1}^N \prod_{k=1}^K \mathcal{N}\bigl(x_i \mid \mu_{ik}, \sigma_{ik}^2\bigr)^{z_{ik}}
$$

目标真实后验：$P(z, \Omega \mid x)$（不可直接求解 → 采用变分近似）。

---

## 3. 变分后验假设 (Mean-Field Factorization)

$$
q(z, \Omega \mid x) = q(z \mid x)\, q(\Omega \mid x)
$$

其中：

1. 隐变量的变分分布（Categorical 独立）：

$$
q(z \mid x) = \prod_{i=1}^N \prod_{k=1}^K \pi_{ik}^{z_{ik}}, \qquad \pi_{ik} \ge 0,\; \sum_{k=1}^K \pi_{ik}=1
$$

2. 参数的变分分布（Dirichlet）：

$$
q(\Omega \mid x) = \prod_{i=1}^N \operatorname{Dir}(\hat{d}_i) \prod_{k=1}^K w_{ik}^{\hat{d}_{ik}-1}
$$

---

## 4. 先验 (Prior)
联合先验：

$$
P(z, \Omega) = P(z \mid \Omega)\, P(\Omega)
$$

其中：

1. 条件先验（类别权重）：
$$
P(z \mid \Omega) = \prod_{i=1}^N \prod_{k=1}^K w_{ik}^{z_{ik}}
$$

2. Dirichlet 先验：
$$
P(\Omega) = \prod_{i=1}^N \operatorname{Dir}(d_i) \prod_{k=1}^K w_{ik}^{d_{ik}-1}
$$

---

## 5. 目标函数：负 ELBO
ELBO 定义：

$$
	ext{ELBO} = \mathbb{E}_{q} [\log P(x \mid z, \Omega)] - \mathrm{KL}\big(q(z,\Omega \mid x)\;\Vert\; P(z,\Omega)\big)
$$

最小化负 ELBO：

$$
\mathcal{L} = -\text{ELBO} = -\mathbb{E}_{q} [\log P(x \mid z, \Omega)] + \mathrm{KL}\big(q(z,\Omega \mid x)\;\Vert\; P(z,\Omega)\big)
$$

拆分：

$$
\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_2 + \mathcal{L}_3
$$

---

## 6. 各组成项

### 6.1 数据似然项 $\mathcal{L}_1$
$$
\mathcal{L}_1 = - \mathbb{E}_{q(z, \Omega)} [ \log P(x \mid z, \Omega) ]
$$
利用指派期望 $\mathbb{E}[z_{ik}] = \pi_{ik}$，高斯对数似然：
$$
\mathcal{L}_1 = - \sum_{i=1}^N \sum_{k=1}^K \pi_{ik} \Big( - \tfrac{(x_i-\mu_{ik})^2}{2\sigma_{ik}^2} - \tfrac{1}{2}\log 2\pi - \log \sigma_{ik} \Big)
$$

### 6.2 隐变量变分差异 $\mathcal{L}_2$
$$
\mathcal{L}_2 = \mathbb{E}_{q(\Omega)} \Big[ \mathrm{KL}\big(q(z \mid x) \Vert P(z \mid \Omega)\big) \Big]
$$
结合 Dirichlet 期望：$\mathbb{E}[\log w_{ik}] = \psi(\hat{d}_{ik}) - \psi\Big(\sum_{k'} \hat{d}_{ik'}\Big)$ 得：
$$
\mathcal{L}_2 = \sum_{i=1}^N \sum_{k=1}^K \pi_{ik} \Big[ \log \pi_{ik} - \big(\psi(\hat{d}_{ik}) - \psi(\sum_{k'} \hat{d}_{ik'})\big) \Big]
$$

### 6.3 参数变分差异 $\mathcal{L}_3$
$$
\mathcal{L}_3 = \mathrm{KL}\big(q(\Omega \mid x) \Vert P(\Omega)\big)
$$
Dirichlet KL 形式化简：
$$
\mathcal{L}_3 = \sum_{i=1}^N \log \frac{\operatorname{Dir}(\hat{d}_i)}{\operatorname{Dir}(d_i)} + \sum_{i=1}^N \sum_{k=1}^K (\hat{d}_{ik}-d_{ik})\Big( \psi(\hat{d}_{ik}) - \psi(\sum_{k'} \hat{d}_{ik'}) \Big)
$$

---

## 7. 符号与函数说明
| 符号 | 含义 |
|------|------|
| $N$ | 数据点数量 |
| $K$ | 成分数 |
| $x_i$ | 第 $i$ 个观测 |
| $z_{ik}$ | 指派指示变量 |
| $\pi_{ik}$ | 变分后验中 $z_{ik}$ 的概率 |
| $w_{ik}$ | (局部) Dirichlet 权重参数（或混合权重） |
| $d_{ik}$ | 先验 Dirichlet 参数 |
| $\hat{d}_{ik}$ | 变分 Dirichlet 参数（后验） |
| $\psi(\cdot)$ | Digamma 函数 |
| $\mathcal{N}(\cdot)$ | 高斯密度 |
| $\operatorname{Dir}(\cdot)$ | Dirichlet 密度 |

---

## 8. 实现提示
1. 计算 $\mathcal{L}_2$ 与 $\mathcal{L}_3$ 时需稳定 digamma 调用。
2. 若使用批训练，可加上缩放系数（mini-batch reweighting）。
3. 优化时常交替更新 $\pi_{ik}$ 与 $\hat{d}_{ik}$。
4. 监控 $\mathcal{L}$ 单调下降以验证实现正确性。

---

## 9. 总结
负 ELBO 由三部分构成：数据拟合 ($\mathcal{L}_1$)、隐变量熵与先验匹配差异 ($\mathcal{L}_2$)、参数先验匹配差异 ($\mathcal{L}_3$)。通过最小化 $\mathcal{L}$ 等价于最大化 ELBO，从而得到对真实后验的近似。

> 注：如需进一步扩展到共享全局权重的标准 GMM，可将 $w_{ik}$ 简化为全局 $w_k$，并相应调整 Dirichlet 参数聚合形式。
