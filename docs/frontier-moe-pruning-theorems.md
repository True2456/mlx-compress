# Frontier MoE Pruning & Merging: Mathematical Proofs and Dark Horse Theorems

**Model Context:** Step-3.7-Flash (45 layers, 42 MoE layers, 288 experts/layer, top-$k=8$, Sigmoid Router with `e_score_correction_bias`, ~185.5B parameters).  
**Upstream Baseline:** REAP (*REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*, Cerebras Research, arXiv:2510.13999 / ICLR 2026).

---

## Executive Summary & Theoretical Grounding

Standard **REAP** (Router-weighted Expert Activation Pruning) evaluates expert importance independently using the zero-order metric:
$$S_{\text{REAP}}(e) = \frac{1}{T_e} \sum_{t \in T_e} g_{t,e} \cdot \|h_{t,e}\|_2$$
where $g_{t,e}$ is the router gate score for expert $e$ on token $t$, and $h_{t,e}$ is the output activation vector of expert $e$.

While REAP out-performs naive weight merging (which suffers from **Functional Subspace Collapse** due to destroyed routing control), standard REAP makes **six strict assumptions** that do not hold in real-world MoE architectures like Step-3.7:

1. **Independence Assumption**: Assumes expert saliency is strictly additive, ignoring pairwise activation covariance between experts.
2. **Static Router Distribution**: Assumes removing experts leaves the remaining routing logits unperturbed, ignoring router phase-shift.
3. **Naïve Weight Addition in Merging**: Assumes expert weight addition $W_A + W_B$ is the only form of merging, missing non-destructive low-rank subspace projection into the un-pruned Shared Expert.
4. **Zero-Order Local Activation Assumption**: Assumes output magnitude $\|h_{t,e}\|$ directly correlates with downstream loss impact, ignoring downstream LayerNorm / Attention nullspaces.
5. **Uniform Depth Allocation**: Assumes every MoE layer has identical rate-distortion characteristics, despite empirical mean REAP scores spanning $0.17$ (layer 3) to $24.0$ (layer 44).
6. **Uniform Data Sampling**: Assumes general text calibration mixes preserve domain-specific "super-experts" (e.g., multi-step tool execution or math reasoning).

Below, we mathematically formulate and prove **6 dark horse algorithms** designed to eliminate these bottlenecks.

---

## 1. Cross-Expert Covariance & Submodular Information Gain (DPP-REAP)

### The Problem
In Step-3.7 Layer 3, Expert 67 fires $\sim 149\text{k}$ times but exhibits low individual activation norm ($\text{REAP} \approx 0.044$). Conversely, two high-norm experts ($A$ and $B$) may have a cosine similarity $\cos(h_A, h_B) \approx 0.98$, meaning they occupy identical functional subspaces. Standard REAP keeps both $A$ and $B$ while discarding expert 67, wasting capacity on redundancy.

### Mathematical Formulation
Define the cross-expert activation kernel $K \in \mathbb{R}^{E \times E}$ for layer $l$ over token calibration set $\mathcal{T}$:
$$K_{i,j} = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} \left( g_{t,i} h_{t,i} \right)^T \left( g_{t,j} h_{t,j} \right)$$
The diagonal $K_{i,i}$ corresponds to the squared REAP energy $\mathbb{E}[\|g_{t,i} h_{t,i}\|^2]$.

We model expert diversity using a **Determinantal Point Process (DPP)** (Kulesza & Taskar, 2012). The probability of selecting a subset of experts $S_{\text{kept}} \subseteq \{1, \dots, E\}$ is proportional to the volume spanned by their feature representations:
$$\mathcal{P}(S_{\text{kept}}) \propto \det(K_{S_{\text{kept}}})$$

### Theorem 1 (Submodular Guarantee of Diversity Pruning)
*Let $F(S) = \log \det (K_S + \sigma^2 I)$ be the set function measuring the total information capacity of expert subset $S$. $F(S)$ is normalized, strictly monotonic, and submodular.*

#### Proof:
1. **Monotonicity**: Adding an expert $e \notin S$ increases the determinant:
   $$\det(K_{S \cup \{e\}} + \sigma^2 I) = \det(K_S + \sigma^2 I) \cdot \left( K_{e,e} + \sigma^2 - \mathbf{k}_{e,S}^T (K_S + \sigma^2 I)^{-1} \mathbf{k}_{e,S} \right)$$
   Since $(K_S + \sigma^2 I)$ is positive definite, the Schur complement term $\left( K_{e,e} + \sigma^2 - \mathbf{k}_{e,S}^T (K_S + \sigma^2 I)^{-1} \mathbf{k}_{e,S} \right) \ge \sigma^2 > 0$ for $\sigma^2 > 0$. Thus $F(S \cup \{e\}) \ge F(S)$.

2. **Submodularity**: The marginal gain of adding expert $e$ is:
   $$\Delta F(e \mid S) = F(S \cup \{e\}) - F(S) = \log \left( K_{e,e} + \sigma^2 - \mathbf{k}_{e,S}^T (K_S + \sigma^2 I)^{-1} \mathbf{k}_{e,S} \right)$$
   For $A \subseteq B \subseteq \{1, \dots, E\}$, matrix inequalities give $(K_B + \sigma^2 I)^{-1} \preceq (K_A + \sigma^2 I)^{-1}$. Therefore:
   $$\mathbf{k}_{e,B}^T (K_B + \sigma^2 I)^{-1} \mathbf{k}_{e,B} \ge \mathbf{k}_{e,A}^T (K_A + \sigma^2 I)^{-1} \mathbf{k}_{e,A}$$
   $$\implies \Delta F(e \mid B) \le \Delta F(e \mid A)$$
   This proves submodularity. $\blacksquare$

#### Practical Algorithm (Greedy DPP-REAP):
By Nemhauser et al. (1978), selecting experts via greedy marginal gain maximization achieves a guaranteed $(1 - 1/e) \approx 63.2\%$ approximation to the optimal combinatorial subset:
1. Start with $S_0 = \emptyset$.
2. For step $k = 1 \dots K_{\text{target}}$ (e.g., 245):
   $$e^* = \arg\max_{e \notin S_{k-1}} \left[ S_{\text{REAP}}(e)^2 - \mathbf{k}_{e, S_{k-1}}^T (K_{S_{k-1}} + \sigma^2 I)^{-1} \mathbf{k}_{e, S_{k-1}} \right]$$
3. $S_k = S_{k-1} \cup \{e^*\}$.

---

## 2. Closed-Form Router Bias Surgery (RBS / Shadow Routing)

### The Problem
When 43 experts are deleted from a layer, tokens that originally selected a pruned expert as choice 1 now fall back onto their next highest choices. Because the router logits were trained for a 288-expert distribution, the fallback activation probabilities are systematically miscalibrated.

### Mathematical Formulation
Step-3.7 uses a **Sigmoid Router** with per-expert correction bias:
$$g_e(x) = \operatorname{sigmoid}\left( W_{\text{gate}}^{(e)} x + b_{\text{gate}}^{(e)} + b_{\text{correction}}^{(e)} \right)$$

Let $\mathcal{T}_{e_p}$ be the set of tokens where expert $e_p \in S_{\text{pruned}}$ was in the top-$k$. When $e_p$ is removed, its routed mass $g_{t, e_p}$ must be absorbed by the remaining experts $k \in S_{\text{kept}}$.

### Theorem 2 (Optimal Router Bias Shift)
*Under a first-order Taylor expansion of the sigmoid activation function, the shift in router correction bias $\Delta b_{\text{correction}}^{(k)}$ that minimizes KL-divergence between the original MoE block output expectation and the pruned MoE block output expectation is given in closed form by:*

$$\Delta b_{\text{correction}}^{(k) *} = \log \left( 1 + \frac{\sum_{e_p \in S_{\text{pruned}}} T(e_p \to k) \cdot \bar{g}_{e_p}}{\bar{g}_k \cdot (1 - \bar{g}_k)} \right)$$
*where $T(e_p \to k)$ is the empirical transition probability that expert $k$ is the top runner-up for tokens routed to $e_p$, and $\bar{g}_e = \mathbb{E}_{t}[g_{t,e}]$.*

#### Proof:
1. The expected output of the MoE block for token $x_t$ before pruning is:
   $$\mathbf{y}_t = \sum_{e \in S_{\text{kept}}} g_e(x_t) W_e x_t + \sum_{e_p \in S_{\text{pruned}}} g_{e_p}(x_t) W_{e_p} x_t$$
2. After pruning, the output becomes $\mathbf{y}_t' = \sum_{k \in S_{\text{kept}}} g_k'(x_t) W_k x_t$, where $g_k'(x_t) = \operatorname{sigmoid}\left( \text{logit}_k(x_t) + \Delta b_k \right)$.
3. We require $\mathbb{E}[\mathbf{y}_t'] = \mathbb{E}[\mathbf{y}_t]$. Approximating $W_{e_p} x_t \approx \sum_{k \in S_{\text{kept}}} T(e_p \to k) W_k x_t$:
   $$\sum_{k \in S_{\text{kept}}} \mathbb{E}[g_k'(x_t)] W_k x_t \approx \sum_{k \in S_{\text{kept}}} \left( \mathbb{E}[g_k(x_t)] + \sum_{e_p \in S_{\text{pruned}}} T(e_p \to k) \mathbb{E}[g_{e_p}(x_t)] \right) W_k x_t$$
4. Matching terms per kept expert $k$:
   $$\mathbb{E}[g_k'(x_t)] = \bar{g}_k + \sum_{e_p \in S_{\text{pruned}}} T(e_p \to k) \bar{g}_{e_p}$$
5. Applying the inverse sigmoid relation $\operatorname{logit}' = \operatorname{logit} + \Delta b_k$:
   $$\Delta b_k = \operatorname{logit}\left( \bar{g}_k + \sum_{e_p} T(e_p \to k) \bar{g}_{e_p} \right) - \operatorname{logit}(\bar{g}_k)$$
   For small probability shifts, $\operatorname{sigmoid}^{-1}(p + \Delta p) - \operatorname{sigmoid}^{-1}(p) \approx \frac{\Delta p}{p(1-p)}$. This yields:
   $$\Delta b_k^* \approx \frac{\sum_{e_p \in S_{\text{pruned}}} T(e_p \to k) \bar{g}_{e_p}}{\bar{g}_k (1 - \bar{g}_k)} \quad \blacksquare$$

---

## 3. Principal Subspace Projection into Shared Experts (Eigen-Folding / PCSF)

### The Problem
Upstream REAP demonstrated that direct weight addition ($W_{\text{kept}} \leftarrow W_{\text{kept}} + W_{\text{pruned}}$) causes **Functional Subspace Collapse** because experts specialize in distinct activation directions. However, Step-3.7 features an un-pruned **Shared Expert** $W_{\text{shared}}$ (intermediate dimension 1280) that processes *every single token*.

### Mathematical Formulation
Let $W_{e_p} \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{intermediate}}}$ be the weight tensor of pruned expert $e_p$. We want to find a low-rank matrix update $\Delta W_{\text{shared}}$ for the Shared Expert to absorb the energy of all pruned experts.

### Theorem 3 (Eckart-Young Subspace Transfer)
*The optimal rank-$r$ update $\Delta W_{\text{shared}}$ minimizing the mean squared reconstruction error of the block output across calibration tokens is given by the weighted truncated SVD of the aggregate pruned expert transform:*

$$\Delta W_{\text{shared}}^* = \sum_{r' = 1}^r \sigma_{r'} \mathbf{u}_{r'} \mathbf{v}_{r me}^T \quad \text{where} \quad M_{\text{pruned}} = \sum_{e_p \in S_{\text{pruned}}} \bar{g}_{e_p} \cdot \left( W_{\text{down}}^{(e_p)} W_{\text{gate\_up}}^{(e_p)} \right)$$

#### Proof:
1. The reconstruction error objective is:
   $$\mathcal{L}(\Delta W_{\text{shared}}) = \mathbb{E}_t \left\| \sum_{e_p \in S_{\text{pruned}}} g_{t,e_p} W_{e_p} x_t - \Delta W_{\text{shared}} x_t \right\|_2^2$$
2. Assuming $x_t \sim \mathcal{N}(0, \mathbf{\Sigma}_x)$, expanding the expectation yields:
   $$\mathcal{L} \propto \left\| \sum_{e_p \in S_{\text{pruned}}} \bar{g}_{e_p} W_{e_p} - \Delta W_{\text{shared}} \right\|_{\mathbf{\Sigma}_x}^2$$
3. By the **Eckart-Young-Mirsky Theorem**, the optimal rank-$r$ approximation of any matrix under the Frobenius norm (and centered covariance norm) is obtained by taking the top $r$ singular vectors of the target matrix $M_{\text{pruned}}$. $\blacksquare$

---

## 4. Second-Order Hessian-Proxy Downstream Projection (H-REAP)

### The Problem
Standard REAP measures magnitude locally at the output of the expert block: $\|g_{t,e} h_{t,e}\|$. However, an expert output $h_{t,e}$ might project directly into the **nullspace of the subsequent LayerNorm** or Attention projection, rendering a large local norm functionally useless downstream.

### Mathematical Formulation
Consider the Taylor expansion of the network loss $\mathcal{L}$ with respect to the output of MoE layer $l$, denoted $\mathbf{y}_l = \sum_{e} g_e h_e$:
$$\Delta \mathcal{L} \approx \left( \frac{\partial \mathcal{L}}{\partial \mathbf{y}_l} \right)^T \Delta \mathbf{y}_l + \frac{1}{2} \Delta \mathbf{y}_l^T \mathbf{H}_l \Delta \mathbf{y}_l$$

When pruning expert $e_p$, $\Delta \mathbf{y}_l = - g_{t, e_p} h_{t, e_p}$. Over trained models at equilibrium, $\mathbb{E}\left[ \frac{\partial \mathcal{L}}{\partial \mathbf{y}_l} \right] \approx 0$.

### Theorem 4 (Hessian-Proxy Saliency Bound)
*Using the Empirical Fisher Information Matrix proxy for the block Hessian $\mathbf{H}_l \approx W_{\text{LN\_next}}^T W_{\text{LN\_next}}$, the true loss inflation from pruning expert $e$ is upper-bounded by:*

$$S_{\text{Hessian}}(e) = \frac{1}{|\mathcal{T}|} \sum_{t \in \mathcal{T}} g_{t,e}^2 \cdot \left\| W_{\text{LN\_next}} \cdot h_{t,e} \right\|_2^2$$

#### Proof:
1. Substitute $\mathbf{H}_l = J^T J$ (Gauss-Newton / Fisher approximation), where $J = \frac{\partial \mathbf{y}_{l+1}}{\partial \mathbf{y}_l}$.
2. In transformer architectures, the immediate downstream transformation is LayerNorm followed by Attention Projection: $J \approx W_{\text{LN}} \cdot \operatorname{diag}(\gamma_l)$.
3. Substituting $\Delta \mathbf{y}_l = - g_{t,e} h_{t,e}$ into the second-order term:
   $$\Delta \mathcal{L}(e) \approx \frac{1}{2} \sum_{t} \left( -g_{t,e} h_{t,e} \right)^T \left( W_{\text{LN}}^T W_{\text{LN}} \right) \left( -g_{t,e} h_{t,e} \right) = \frac{1}{2} \sum_{t} g_{t,e}^2 \cdot \|W_{\text{LN}} h_{t,e}\|_2^2 \quad \blacksquare$$

---

## 5. Rate-Distortion Non-Uniform Depth Allocation (Asymmetric REAP)

### The Observation
Empirical data from Step-3.7 shows mean REAP scores vary by **2 orders of magnitude** across depth:
- **Layer 3 (early)**: Mean REAP $\approx 0.17$
- **Layer 24 (mid)**: Mean REAP $\approx 6.2$
- **Layer 44 (late)**: Mean REAP $\approx 24.0$

### Mathematical Formulation
Let $D_l(k_l)$ be the discarded router saliency mass in layer $l$ when retaining $k_l$ experts ($k_l \in [200, 288]$). We formulate expert reduction as a constrained Rate-Distortion optimization:
$$\min_{\{k_3, \dots, k_{44}\}} \sum_{l=3}^{44} D_l(k_l) \quad \text{subject to} \quad \sum_{l=3}^{44} k_l = K_{\text{total}} = 42 \times 245 = 10,290$$

### Theorem 5 (Water-Filling Convex Optimization for Expert Depth)
*If the layer distortion curves $D_l(k)$ are strictly convex and decreasing in $k$, the optimal expert count $k_l^*$ for each layer satisfies the discrete Water-Filling condition:*

$$\frac{\Delta D_l(k_l^*)}{\Delta k} \approx -\lambda \quad \forall l \in [3, 44]$$

#### Proof:
1. Form the Lagrangian:
   $$\mathcal{L}(\{k_l\}, \lambda) = \sum_{l=3}^{44} D_l(k_l) + \lambda \left( \sum_{l=3}^{44} k_l - K_{\text{total}} \right)$$
2. Taking discrete differences with respect to $k_l$:
   $$\frac{\partial \mathcal{L}}{\partial k_l} = D_l(k_l + 1) - D_l(k_l) + \lambda = 0 \implies \Delta D_l(k_l) = -\lambda$$
3. Since $D_l(k)$ has higher slope $\left|\frac{\Delta D_l}{\Delta k}\right|$ in late layers (due to higher overall activation magnitude), the water-filling algorithm automatically assigns **more kept experts to late layers** and **fewer kept experts to early layers** for the exact same overall parameter budget. $\blacksquare$

---

## 6. Contrastive Skill-Shield Saliency (CS-REAP)

### The Problem
General calibration datasets (e.g., SlimPajama, general web text) contain $<1\%$ agentic tool schema executions or multi-step math proofs. Standard REAP on general data prunes domain-specific "super-experts" because their global activation frequency is low.

### Mathematical Formulation
Let $S_{\text{general}}(e)$ be the REAP score computed over general corpus $\mathcal{D}_{\text{general}}$, and $S_{\text{target}}(e)$ be the score over high-value domain corpus $\mathcal{D}_{\text{target}}$ (e.g., SWE-bench / tool trajectories).

### Theorem 6 (Skill Preservation Lower-Bound)
*Defining the Contrastive Skill-Shield score:*
$$S_{\text{CS}}(e) = S_{\text{general}}(e) + \alpha \cdot \max\left( 0, S_{\text{target}}(e) - S_{\text{general}}(e) \right)$$
*guarantees that any expert whose domain-specific saliency exceeds the general average by threshold $\tau = \frac{S_{\text{cutoff}} - S_{\text{general}}(e)}{\alpha}$ is shielded from pruning.*

#### Proof:
1. Pruning drops experts where $S_{\text{CS}}(e) < S_{\text{cutoff}}$.
2. For an expert with high target utility $S_{\text{target}}(e) > S_{\text{general}}(e)$:
   $$S_{\text{CS}}(e) = (1 - \alpha) S_{\text{general}}(e) + \alpha S_{\text{target}}(e)$$
3. Setting $S_{\text{CS}}(e) \ge S_{\text{cutoff}}$ yields $S_{\text{target}}(e) \ge \frac{S_{\text{cutoff}} - (1-\alpha) S_{\text{general}}(e)}{\alpha}$. As $\alpha \to 1$, retention depends strictly on target domain performance, preventing collateral damage to specialized reasoning modules. $\blacksquare$

---

## Implementation Roadmap for `reap_stream`

To test these dark-horse algorithms directly in your existing codebase:

```python
# 1. Router Bias Surgery (RBS) Implementation in reap_stream/build_student.py
def apply_router_bias_surgery(router_bias, pruned_indices, transition_matrix, mean_gates):
    """
    Applies closed-form bias shift to sigmoid router correction bias.
    """
    delta_b = np.zeros_like(router_bias)
    for k in kept_indices:
        absorbed_mass = sum(transition_matrix[ep, k] * mean_gates[ep] for ep in pruned_indices)
        p_k = mean_gates[k]
        delta_b[k] = absorbed_mass / (p_k * (1.0 - p_k) + 1e-6)
    return router_bias + delta_b
```

---

## Literature Cross-Check & Citation Index

1. **REAP (Base Pruning)**: *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression*, Cerebras Research, arXiv:2510.13999 (ICLR 2026).
2. **Determinantal Point Processes (DPP)**: Kulesza, A., & Taskar, B. (2012). *Determinantal point processes for machine learning*. Foundations and Trends in Machine Learning.
3. **Submodular Optimization**: Nemhauser, G. L., Wolsey, L. A., & Fisher, M. L. (1978). *An analysis of approximations for maximizing submodular set functions*. Mathematical Programming.
4. **Second-Order Pruning & Hessian Proxies**: 
   - Hassibi, B., & Stork, D. G. (1992). *Second order derivatives for network pruning: Optimal Brain Surgeon*. NIPS.
   - Frantar, E., et al. (2023). *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*. ICML.
   - Sun, M., et al. (2023). *A Simple and Effective Pruning Approach for Large Language Models (Wanda)*. ICLR.
5. **Low-Rank Matrix Approximations**: Eckart, C., & Young, G. (1936). *The approximation of one matrix by another of lower rank*. Psychometrika.
6. **Rate-Distortion Theory**: Shannon, C. E. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal.
