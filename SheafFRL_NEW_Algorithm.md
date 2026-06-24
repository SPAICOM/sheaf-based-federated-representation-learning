## 1. Problem Setup

Let $\mathcal{G} = (\mathcal{N}, \mathcal{E})$ be a finite undirected graph with $|\mathcal{N}| = N$ agents. Each agent $i \in \mathcal{N}$ holds a private dataset $\mathcal{T}_i$ drawn from a local distribution $P_i$, and trains a **local model** split into:

- a **neural encoder** $f_{\varphi_i} : \mathbb{R}^{p_i} \to \mathcal{F}(i) = \mathbb{R}^{d_i}$
- a **local decoder / task head** $h_{\psi_i} : \mathcal{F}(i) \to \mathcal{Y}_i$

The local empirical risk for agent $i$ is:

$$\mathcal{L}_i(\boldsymbol{\theta}_i) = \frac{1}{M_i} \sum_{n=1}^{M_i} \ell_i!\left(h_{\psi_i} \circ f_{\varphi_i}(\mathbf{x}_i^n)\right), \quad \boldsymbol{\theta}_i = (\varphi_i, \psi_i). \tag{1}$$

Agents may have **heterogeneous latent dimensions** $d_i \neq d_j$ and **heterogeneous data distributions**. No shared global latent space is assumed.

---

## 2. Network Sheaf and Restriction Maps

To model geometric relationships among local representation spaces, we endow $\mathcal{G}$ with a **network sheaf** $\mathcal{F}$:

- Each node $i$ has a **node stalk** $\mathcal{F}(i) = \mathbb{R}^{d_i}$ — its latent space.
- Each edge $(i,j) \in \mathcal{E}$ has an **edge stalk** $\mathcal{F}(e_{ij}) = \mathbb{R}^{d_i}$ (with $d_{ij} = \max(d_i, d_j)$, choosing $i$ as the head whenever $d_i > d_j$).
- For each incident relation $i \to e_{ij}$ (head node), $\mathcal{F}$ specifies a **linear restriction map** that aligns neighboring latent spaces. In our case the two maps on each edge are:

$$\mathbf{O}_{ji} \in \mathrm{O}(d_i), \qquad \mathbf{V}_{ij} \in \mathrm{St}(d_i, d_j), \tag{2}$$

where $\mathrm{O}(d) = {\mathbf{O} \in \mathbb{R}^{d\times d} \mid \mathbf{O}^\top\mathbf{O} = \mathbf{I}_d}$ is the orthogonal group and $\mathrm{St}(d,k) = {\mathbf{V} \in \mathbb{R}^{d\times k} \mid \mathbf{V}^\top\mathbf{V}=\mathbf{I}_k}$ is the Stiefel manifold ($k < d$).

> **Semantic embedding principle:** $\mathbf{V}_{ij}$ embeds the lower-dimensional space $\mathcal{F}(j)$ isometrically into $\mathcal{F}(i)$, preserving semantics without compression. The graph is oriented from lower-$d$ nodes to higher-$d$ nodes.

The reparameterization $\mathbf{V}_{ij} = \mathbf{O}_{ji}^\top \mathbf{V}_{ij}$ (following orthogonal group transitivity) simplifies the coboundary to:

$$(\delta \mathbf{z})_{e_{ij}} = \mathbf{O}_{ji}\mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j \xrightarrow{\text{reparametrize}} \mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j. \tag{3}$$

---

## 3. Sheaf Total Variation and Gluing Penalty

The **sheaf Laplacian** $\mathbf{L}_\mathcal{F} = \delta^\top\delta$ is symmetric positive semidefinite. The **total variation** of a collection of latents $\mathbf{z} = {\mathbf{z}_i}_{i \in \mathcal{N}}$ is:

$$\mathcal{TV}(\mathbf{z}) = |\delta\mathbf{z}|_2^2 = \sum_{e_{ij} \in \mathcal{E}} |\mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j|_2^2. \tag{7, 9}$$

A **global section** $\mathbf{z}^\star \in \ker(\mathbf{L}_\mathcal{F})$ satisfies $\mathbf{z}_i^\star = \mathbf{V}_{ij}\mathbf{z}_j^\star$ for all edges — perfect geometric alignment. $\mathcal{TV}$ relaxes this hard constraint into a soft penalty.

The local contribution of node $i$ decomposes into **incoming** and **outgoing** embedding terms:

$$\mathcal{TV}(\mathbf{z})|_i = \frac{1}{2}\underbrace{\sum_{j \in \mathcal{N}(i)^-} |\mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j|_2^2}_{\text{Incoming embedding}} + \frac{1}{2}\underbrace{\sum_{j \in \mathcal{N}(i)^+} |\mathbf{z}_j - \mathbf{V}_{ji}\mathbf{z}_i|_2^2}_{\text{Outgoing embedding}}, \tag{11}$$

where $\mathcal{N}(i)^-$ are neighbors with $d_j < d_i$ and $\mathcal{N}(i)^+$ are those with $d_j > d_i$. To also reflect the similarity with the few existing federated methods working on the representation level instead of on weight spaces, we can define the _local section_ condition for each edge incident to a particular node $i$ as:

$$\mathbf{z}_i = \sum_{j\in \mathcal{N}(i)^-} \mathbf{V}_{ij} \mathbf{z}_j + \sum_{j\in \mathcal{N}(i)^+} \mathbf{V}_{ji}^\top \mathbf{z}_j.$$

This formulation directly generalizes the definition of local consensus for graphs, where consensus is obtained when the signal on a node equals the average signal from its neighbors. In this case we have a local section condition on a neighborhood of the network sheaf if the signal on a node equals the after-alignment average of the signals from its neighbors:

$$\langle\mathbf{z}\rangle_i = \sum_{j\in \mathcal{N}(i)^-} \mathbf{V}_{ij} \mathbf{z}_j + \sum_{j\in \mathcal{N}(i)^+} \mathbf{V}_{ji}^\top \mathbf{z}_j.$$

The total variation penalty can then be written as a generalised distributed version of the penalisation term of _FedProto_ and similar approaches:

$$\mathcal{TV}(\mathbf{z})|_i = \frac{1}{2} |\mathbf{z}_i - \langle\mathbf{z}\rangle_i|_2^2.$$

> **Federated Representation Learning:** In _FedProto_, local class-prototype representations are compared with a global mean representation when building the penalization term. It differs from our setup since: (i) global mean representations of the prototypes are defined globally by averaging over all agents in the network, (ii) they are aggregated without using any alignment map, thus assuming directly comparable latent spaces. A distributed implementation of FedProto would be a subcase of our Sheaf-FRL setup where all stalks have the same dimensionality and all restriction maps are identity maps.

### Anchor-restricted (scalable) gluing penalty

Rather than evaluating $\mathcal{TV}$ over all samples, we restrict to a small **anchor set** $\mathcal{A} \subset {1, \ldots, M_{\min}}$ with $|\mathcal{A}| = K \ll M_{\min}$. The **anchor feature matrix** of agent $i$ at round $t$ is:

$$\mathbf{A}_i(\varphi_i) = \left[f_{\varphi_i}(\mathbf{x}_i^k)\right]_{k \in \mathcal{A}} \in \mathbb{R}^{K \times d_i}. \tag{12}$$

The local gluing penalty decouples across nodes:

$$\mathcal{R}_{\mathcal{A}}|_i(\varphi_i, {\mathbf{V}_{ij}}, {\mathbf{V}_{ji}}) = \frac{\lambda}{2K} \left[\sum_{j \in \mathcal{N}(i)^-} |\mathbf{A}_i(\varphi_i) - \mathbf{V}_{ij}\mathbf{A}_j(\varphi_j)|_F^2 + \sum_{j \in \mathcal{N}(i)^+} |\mathbf{A}_j(\varphi_j) - \mathbf{V}_{ji}\mathbf{A}_i(\varphi_i)|_F^2\right]. \tag{14}$$

The full **SFRL optimization problem** is:

$$\min_{\substack{{\boldsymbol{\theta}_i = (\varphi_i, \psi_i)} \ {\mathbf{V}_{ij} \in \mathrm{St}(d_i,d_j)} \ {\mathbf{V}_{ji} \in \mathrm{St}(d_j,d_i)}}} \sum_{i \in \mathcal{N}} \mathcal{L}_i(\boldsymbol{\theta}_i) + \mathcal{R}_\mathcal{A}({\varphi_i}, {\mathbf{V}_{ij}}, {\mathbf{V}_{ji}}). \tag{SFRL}$$

---

## 4. Sheaf-FRL First Algorithm (Alternating Minimization)

The algorithm alternates between two steps per communication round $t$:

### Step 1 — Isometric Embedding Update (closed form)

Each node $i$ broadcasts its anchor matrix $\mathbf{A}_i^t$ to all $j \in \mathcal{N}(i)$. Then, for each edge $e_{ij}$:

- If $j \in \mathcal{N}(i)^-$ (node $i$ is the head, incoming map): $$\mathbf{V}_{ij}^t = \mathbf{U}\mathbf{W}^\top, \quad \text{where } [\mathbf{U},\boldsymbol{\Sigma},\mathbf{W}^\top] = \operatorname{SVD}(\mathbf{A}_i^t\mathbf{A}_j^{t\top}). \tag{16}$$
    
- If $j \in \mathcal{N}(i)^+$ (node $i$ is the tail, outgoing map): $$\mathbf{V}_{ji}^t = \mathbf{U}\mathbf{W}^\top, \quad \text{where } [\mathbf{U},\boldsymbol{\Sigma},\mathbf{W}^\top] = \operatorname{SVD}(\mathbf{A}_j^t\mathbf{A}_i^{t\top}). \tag{16}$$
    

This is the **orthogonal Procrustes solution** to $\min_{\mathbf{V} \in \mathrm{St}} |\mathbf{A}_i - \mathbf{V}\mathbf{A}_j|_F^2$.

> **Remark (homogeneous case):** When $d_i = d$ for all $i$, the maps reduce to canonical orthogonal Procrustes with $\mathbf{O}_{ij}^t = \mathbf{U}\mathbf{W}^\top$ and $\mathbf{O}_{ji} = \mathbf{O}_{ij}^\top$, halving the number of SVD computations (Algorithm 2).

### Step 2 — Neural Parameter Gradient Update

Each agent $i$ updates its local parameters $\boldsymbol{\theta}_i$ via gradient descent, combining the task loss and the sheaf regularization:

$$\nabla_{\varphi_i}\mathcal{R}_\mathcal{A}|_i = \frac{\lambda}{K} \sum_{j \in \mathcal{N}(i)} \sum_{k \in \mathcal{A}} \left(\nabla_{\varphi_i} f_{\varphi_i}(\mathbf{x}_i^k)\right)^\top !\left(f_{\varphi_i}(\mathbf{x}_i^k) - \mathbf{V}_{ij} f_{\varphi_j}(\mathbf{x}_j^k)\right), \tag{17}$$

with the regularization contribution to the full parameter gradient:

$$\mathbf{r}_i = \begin{bmatrix} \nabla_{\varphi_i}\mathcal{R}_\mathcal{A}|_i \ \mathbf{0}_{|\psi_i|} \end{bmatrix}, \tag{18}$$

and the local update rule:

$$\boldsymbol{\theta}_i^{t+1} = \boldsymbol{\theta}_i^t - \eta_\theta \left(\nabla_{\boldsymbol{\theta}_i}\mathcal{L}_i(\boldsymbol{\theta}_i^t) + \mathbf{r}_i^t\right). \tag{19}$$

> **Communication cost (Remark 2):** Each agent broadcasts only $\mathbf{A}_i^t \in \mathbb{R}^{K \times d_i}$, transmitting $\mathcal{O}(d_i K)$ scalars per round. Model parameters are never communicated.

---

## 5. Semantic Communications Oriented New Algorithm

### 5.1 Global Pilot Set as the Anchor Set $\mathcal{A}$

In the implementation, the anchor set $\mathcal{A}$ corresponds to a **global pilot dataset** — a fixed set of $K$ samples shared by all agents in the network (same samples, possibly different modalities per agent). This is the `global_pilot` key in each training batch.

At each step, each agent encodes the pilot batch through its current encoder:

$$\mathbf{A}_i^t = f_{\varphi_i^t}({\mathbf{x}_i^k}_{k \in \mathcal{A}}) \in \mathbb{R}^{K \times d_i}.$$

The pilot latents are accumulated over the epoch and used at training epoch end to update the **Stiefel matrices** (restriction maps) on each node via SVD of the whitened cross-covariance.

> The epoch-level accumulation gives more stable cross-covariance estimates than any single mini-batch.

### 5.2 Pre-whitening Before the Sheaf Penalty

Before computing the sheaf penalty $|\mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j|^2$ and before updating the Stiefel matrices, the implementation applies **ZCA whitening** to the pilot latents. This is a key deviation from the base paper formulation that significantly stabilises training.

#### Parameterised whitening and colouring operators

For each agent $i$ we introduce a **whitening map** $g_{\phi_i} : \mathbb{R}^{d_i} \to \mathbb{R}^{d_i}$ and a **colouring map** $g^*_{\phi_i} : \mathbb{R}^{d_i} \to \mathbb{R}^{d_i}$. Both are fully determined by the single parameter tuple

$$\phi_i = (\bar{\mathbf{z}}_i,;\mathbf{v}_i,;\mathbf{W}_i,;\boldsymbol{\gamma}_i,;\boldsymbol{\beta}_i),$$

where $\bar{\mathbf{z}}_i \in \mathbb{R}^{d_i}$ is the running mean, $\mathbf{v}_i \in \mathbb{R}^{d_i}$ the running per-feature variance, $\mathbf{W}_i \in \mathbb{R}^{d_i \times d_i}$ the (symmetric) whitening matrix, and $(\boldsymbol{\gamma}_i, \boldsymbol{\beta}_i)$ the affine scale/shift.

The key design choice is that **the colouring map $g^*_{\phi_i}$ carries no independent parameters**: it is the closed-form inverse of $g_{\phi_i}$, computed analytically from $\phi_i$ alone. This avoids redundant learning, guarantees $g^*_{\phi_i} \circ g_{\phi_i} = \mathrm{id}$ exactly, and means that once the whitening layer is fit or updated, colouring is immediately consistent at no extra cost.

**Statistics** (buffer-and-fit variant; fit once per epoch on the full pilot buffer):

Given pilot latents $\mathbf{Z} \in \mathbb{R}^{K \times d_i}$, compute the SVD of the centred data matrix $\mathbf{Z}_c = \mathbf{Z} - \mathbf{1}\bar{\mathbf{z}}_i^\top$:

$$\mathbf{Z}_c = \mathbf{U}\mathbf{S}\mathbf{V}^\top, \qquad \lambda_r = \frac{s_r^2}{\max(K-1,,1)},$$

$$\phi_i ;=; \Bigl(\bar{\mathbf{z}}_i,;\mathbf{v}_i = \mathbf{1},;\mathbf{W}_i = \mathbf{V}\operatorname{diag}(\lambda_r^{-1/2})\mathbf{V}^\top,;\boldsymbol{\gamma}_i = \mathbf{1},;\boldsymbol{\beta}_i = \mathbf{0}\Bigr).$$

**Forward operations** (right-multiply convention, batch dimension first):

$$g_{\phi_i}(\mathbf{z}) = \bigl((\mathbf{z} - \bar{\mathbf{z}}_i) \odot \mathbf{v}_i^{-1/2}\bigr),\mathbf{W}_i \odot \boldsymbol{\gamma}_i + \boldsymbol{\beta}_i,$$

$$g^*_{\phi_i}(\mathbf{z}) = \bigl((\mathbf{z} - \boldsymbol{\beta}_i) \odot \boldsymbol{\gamma}_i^{-1}\bigr),\mathbf{W}_i^{-\top} \odot \mathbf{v}_i^{1/2} + \bar{\mathbf{z}}_i,$$

where $\mathbf{W}_i^{-\top}\mathbf{z}^\top$ is evaluated via `torch.linalg.solve`$(\mathbf{W}_i^\top, \mathbf{z}^\top)$ without forming the inverse explicitly. Since $\mathbf{W}_i$ is symmetric at convergence, $\mathbf{W}_i^{-\top} = \mathbf{W}_i^{-1}$. By construction, $g^*_{\phi_i} \circ g_{\phi_i} = \mathrm{id}$ exactly.

The **forward-pass sheaf penalty** operates on whitened representations $g_{\phi_i}(\mathbf{z}_i)$, while the task loss is evaluated on the original encoder output $\mathbf{z}_i = f_{\varphi_i}(\mathbf{x}_i)$ fed directly to the decoder $h_{\psi_i}$. This decoupling avoids any information loss for the downstream task (normalising the latent space can hinder classification).

The **Stiefel matrix update** (at epoch end) likewise operates on whitened accumulated pilots $g_{\phi_i}(\mathbf{A}_i)$.

**Why whitening?** The operators $g_{\phi_i}$ remove the scale and correlation structure of each agent's latent space. This is beneficial because:

1. It makes the Procrustes problem (SVD of the cross-covariance) well-conditioned and invariant to the particular coordinate system each encoder happens to learn.
2. It absorbs the anisotropic scaling that may arise from the isometric restriction maps connecting latent spaces of models solving the same task. Analogous normalisation is used in the semantic alignment literature to make relative-representation zero-shot methods more robust.
3. Whitening has been shown to facilitate interpretability and concept-representation alignment (and potentially steering), as in Barton et al. 2026.
4. With the learnable whitening variant we also avoid explicit SVD on each node's representations.

#### Acquiring whitening statistics: buffer-and-fit vs. SWBN

Two approaches are supported for estimating $\phi_i$ in a distributed setting. In both cases the colouring map $g^*_{\phi_i}$ follows for free as the closed-form inverse.

1. **Buffer-and-fit** — accumulate all pilot representations seen during a training epoch, then fit $\phi_i$ once at epoch end via the SVD procedure above. The resulting statistics are frozen and used both for the subsequent Stiefel matrix update and for the regularisation term in the following epoch.
    
2. **Learnable SWBN-style whitening layer** — parameterise $g_{\phi_i}$ as a differentiable layer whose internal whitening matrix $\mathbf{W}_i$ is updated _online_ without expensive SVD or eigendecomposition at each step, following the Stochastic Whitening Batch Normalisation (SWBN) design of Zhang et al. (CVPR 2021). Because the colouring layer shares $\phi_i$ and carries no independent parameters, it does not require a separate learning mechanism: it simply reads the current $\mathbf{W}_i$ from the whitening layer and inverts it on the fly.
    

#### SWBN-based design for $g_{\phi_i}$

The key idea of SWBN is to **decouple** the update of the whitening matrix from the backpropagation of the task loss. The layer maintains two sets of parameters with separate update rules:

- **Whitening matrix** $\mathbf{W}_i \in \mathbb{R}^{d_i \times d_i}$ (initialised to $\mathbf{I}_{d_i}$), updated only in the **forward pass** via a stochastic gradient step on a whitening criterion $\mathcal{C}$, detached from the task backward graph.
- **Task parameters** $(\boldsymbol{\gamma}_i, \boldsymbol{\beta}_i) \in \mathbb{R}^{d_i} \times \mathbb{R}^{d_i}$ (initialised to $\mathbf{1}$ and $\mathbf{0}$), updated only in the **backward pass** by the task/alignment loss gradient.
- **Running statistics** $(\boldsymbol{\mu}_i^E, \mathbf{v}_i^E)$, updated as exponential moving averages of the batch mean and variance during training, and frozen at inference time (analogously to BN).

**Forward pass** (training, input $\mathbf{Z} \in \mathbb{R}^{K \times d_i}$):

1. Compute batch mean and variance per feature dimension: $\boldsymbol{\mu} = \tfrac{1}{K}\mathbf{1}^\top\mathbf{Z}$, $;\mathbf{v} = \tfrac{1}{K-1}\sum_k (\mathbf{z}_k - \boldsymbol{\mu})^2$.
2. Update running stats: $\boldsymbol{\mu}_i^E \leftarrow \eta,\boldsymbol{\mu}_i^E + (1-\eta),\boldsymbol{\mu}$, and similarly for $\mathbf{v}_i^E$.
3. Standardise: $\mathbf{Z}^S = (\mathbf{Z} - \mathbf{1}\boldsymbol{\mu}),\operatorname{diag}(\mathbf{v} + \varepsilon)^{-1/2}$.
4. Compute sample correlation matrix: $\hat{\boldsymbol{\Sigma}} = \tfrac{1}{K}(\mathbf{Z}^S)^\top \mathbf{Z}^S$ (entries in $[-1,1]$ by construction).
5. **Update** $\mathbf{W}_i$ (detached from the task backward graph) by one stochastic step on the chosen criterion: $$\mathbf{W}_i \leftarrow \mathbf{W}_i - \alpha,\Delta\mathbf{W}_i, \qquad \Delta\mathbf{W}_i = \begin{cases} (\mathbf{W}_i\hat{\boldsymbol{\Sigma}}\mathbf{W}_i^\top - \mathbf{I}),\mathbf{W}_i & \mathcal{C} = \mathcal{C}_{\mathrm{KL}}, \[4pt] \dfrac{(\mathbf{W}_i\hat{\boldsymbol{\Sigma}}\mathbf{W}_i^\top - \mathbf{I}),\mathbf{W}_i\hat{\boldsymbol{\Sigma}}}{|\mathbf{I} - \mathbf{W}_i\hat{\boldsymbol{\Sigma}}\mathbf{W}_i^\top|_F} & \mathcal{C} = \mathcal{C}_{\mathrm{Fro}}, \end{cases}$$ with step size $\alpha \ll 1$ (e.g.\ $10^{-5}$).
6. Enforce symmetry (ZCA): $\mathbf{W}_i \leftarrow \tfrac{1}{2}(\mathbf{W}_i + \mathbf{W}_i^\top)$.
7. Whiten and affine-rescale: $\mathbf{Z}^W = \mathbf{Z}^S,\mathbf{W}_i^\top$, output $\hat{\mathbf{Z}} = \mathbf{Z}^W \odot \boldsymbol{\gamma}_i + \boldsymbol{\beta}_i$.

The whitened output $\hat{\mathbf{Z}}$ is $g_{\phi_i}(\mathbf{Z})$ with $\phi_i = (\boldsymbol{\mu}_i^E, \mathbf{v}_i^E, \mathbf{W}_i, \boldsymbol{\gamma}_i, \boldsymbol{\beta}_i)$.

**Backward pass:** gradients flow through steps 3 and 7 to update $(\boldsymbol{\gamma}_i, \boldsymbol{\beta}_i)$ and the encoder parameters $\varphi_i$. $\mathbf{W}_i$ is treated as a constant (detached from the computational graph), exactly as in SWBN.

**Inference forward pass:** replace batch stats with frozen running stats $(\boldsymbol{\mu}_i^E, \mathbf{v}_i^E)$; apply the final $\mathbf{W}_i$, $\boldsymbol{\gamma}_i$, $\boldsymbol{\beta}_i$.

#### Colouring as the closed-form inverse of $g_{\phi_i}$

Since $\mathbf{W}_i$ is symmetric and invertible at convergence, $g^*_{\phi_i}$ is obtained by analytically reversing steps 3–7:

$$g^*_{\phi_i}(\mathbf{z}) = \Bigl[\bigl((\mathbf{z} - \boldsymbol{\beta}_i) \odot \boldsymbol{\gamma}_i^{-1}\bigr),\mathbf{W}_i^{-1}\Bigr] \odot (\mathbf{v}_i^E + \varepsilon)^{1/2} + \boldsymbol{\mu}_i^E,$$

where $\mathbf{W}_i^{-1}\mathbf{z}^\top$ is evaluated via `torch.linalg.solve`$(\mathbf{W}_i, \mathbf{z}^\top)$. The implementation is:

```python
class SWBNWhiteningLayer(nn.Module):
    """Learnable ZCA-whitening layer (SWBN, Zhang et al. CVPR 2021).
    Owns all parameters phi_i = (mu_E, v_E, W_i, gamma, beta).
    """
    def __init__(self, d: int, criterion: str = 'fro',
                 alpha: float = 1e-5, momentum: float = 0.95, eps: float = 1e-8):
        super().__init__()
        self.criterion, self.alpha, self.momentum, self.eps = criterion, alpha, momentum, eps
        self.gamma = nn.Parameter(torch.ones(d))       # task parameters (trained by backprop)
        self.beta  = nn.Parameter(torch.zeros(d))
        self.register_buffer('W',            torch.eye(d))    # whitening matrix (detached)
        self.register_buffer('running_mean', torch.zeros(d))  # running statistics
        self.register_buffer('running_var',  torch.ones(d))

    def forward(self, Z: Tensor) -> Tensor:            # Z: [K, d]
        if self.training:
            mu = Z.mean(0);  v = Z.var(0, unbiased=True).clamp(min=0)
            self.running_mean.mul_(self.momentum).add_((1 - self.momentum) * mu.detach())
            self.running_var .mul_(self.momentum).add_((1 - self.momentum) *  v.detach())
        else:
            mu, v = self.running_mean, self.running_var
        Z_s = (Z - mu) / (v + self.eps).sqrt()                  # standardise
        if self.training:
            Sigma = (Z_s.T @ Z_s) / Z_s.shape[0]
            with torch.no_grad():                                # W update detached
                WSWt = self.W @ Sigma @ self.W.T
                if self.criterion == 'kl':
                    dW = (WSWt - torch.eye_like(WSWt)) @ self.W
                else:  # 'fro'
                    dW = ((WSWt - torch.eye_like(WSWt)) @ self.W @ Sigma) \
                         / (WSWt - torch.eye_like(WSWt)).norm().clamp(min=self.eps)
                self.W -= self.alpha * dW
                self.W  = 0.5 * (self.W + self.W.T)             # enforce symmetry (ZCA)
        return Z_s @ self.W.T * self.gamma + self.beta          # whiten + affine rescale


class SWBNColouringLayer(nn.Module):
    """Closed-form inverse of a paired SWBNWhiteningLayer.
    Owns NO parameters; reads phi_i directly from the whitening layer.
    """
    def __init__(self, whitening_layer: SWBNWhiteningLayer):
        super().__init__()
        self.W_layer = whitening_layer                           # shared reference, no new params

    def forward(self, z: Tensor) -> Tensor:            # z: [K, d]
        wl = self.W_layer
        z = (z - wl.beta) / wl.gamma.clamp(min=wl.eps)         # invert affine rescale
        z = torch.linalg.solve(wl.W, z.T).T                    # invert whitening (W symmetric)
        z = z * (wl.running_var + wl.eps).sqrt() + wl.running_mean  # invert standardisation
        return z
```

> _TODO:_ Integrate `SWBNWhiteningLayer` and `SWBNColouringLayer` into `src/communication/whitening.py`. Add a `learnable_whitening: bool = True` flag to the `SheafFRL` orchestrator: when `True`, pass encoded pilots through `SWBNWhiteningLayer` before computing the sheaf penalty, and use the paired `SWBNColouringLayer` (sharing the same $\phi_i$, zero extra parameters) for re-colouring in the after-communication task loss. Validate that `W_i` updates are correctly detached from the computational graph before integration into Phase A and C training loops.

### 5.3 After-Communication Task Loss

To encourage representations that are more amenable to semantic communication, we augment the training objective with an **after-communication task loss** that directly penalises errors incurred when pilot representations are transported across the network and decoded by a remote agent.

For each edge $(i, j) \in \mathcal{E}$, let $(\mathbf{A}_i, \mathbf{y}_i)$ and $(\mathbf{A}_j, \mathbf{y}_j)$ denote the pilot anchor matrices and their corresponding labels. All transport operations are defined in the **whitened space**: denote by $\tilde{\mathbf{A}}_i = g_{\phi_i}(\mathbf{A}_i)$ and $\tilde{\mathbf{A}}_j = g_{\phi_j}(\mathbf{A}_j)$ the whitened pilot matrices for agents $i$ and $j$ respectively.

**Direction $j \to i$** (agent $j$ transmits to agent $i$):

1. **Apply the restriction map** to transport $\tilde{\mathbf{A}}_j$ into $i$'s whitened space: $$\hat{\mathbf{A}}_{j \to i} = \tilde{\mathbf{A}}_j,\mathbf{V}_{ij}^\top \in \mathbb{R}^{K \times d_i}.$$
2. **Re-colour** back to $i$'s native representation space using the closed-form inverse $g^*_{\phi_i}$: $$\hat{\mathbf{A}}_{j \to i}^{\text{col}} = g^*_{\phi_i}!\bigl(\hat{\mathbf{A}}_{j \to i}\bigr).$$
3. **Decode and evaluate** with agent $i$'s task head against $j$'s labels: $$\mathcal{L}_{\text{comm}}^{j \to i} = \ell_i!\left(h_{\psi_i}!\left(\hat{\mathbf{A}}_{j \to i}^{\text{col}}\right),; \mathbf{y}_j\right).$$

**Direction $i \to j$** (symmetric, using $g^*_{\phi_j}$ and $\mathbf{V}_{ji}$):

$$\hat{\mathbf{A}}_{i \to j} = \tilde{\mathbf{A}}_i,\mathbf{V}_{ji}^\top, \qquad \mathcal{L}_{\text{comm}}^{i \to j} = \ell_j!\left(h_{\psi_j}!\left(g^*_{\phi_j}!\bigl(\hat{\mathbf{A}}_{i \to j}\bigr)\right),; \mathbf{y}_i\right).$$

Note that in the $i \to j$ direction the receiver $j$ applies its own colouring $g^*_{\phi_j}$ (not $g^*_{\phi_i}$), since the transported representation must be decoded by $j$'s task head in $j$'s native space.

The **total communication loss** aggregated over all edges is:

$$\mathcal{L}_{\text{comm}} = \sum_{(i,j) \in \mathcal{E}} \left(\mathcal{L}_{\text{comm}}^{j \to i} + \mathcal{L}_{\text{comm}}^{i \to j}\right).$$

The **total training objective** is therefore:

$$\mathcal{L}_{\text{total}} = \underbrace{\sum_{i \in \mathcal{N}} \mathcal{L}_i(\boldsymbol{\theta}_i)}_{\text{private task loss}} + \lambda,\mathcal{R}_{\mathcal{A}} + \mu,\underbrace{\mathcal{L}_{\text{comm}}}_{\text{after-comm task loss}}.$$

> **Key remark:** The after-communication loss is only meaningful when agents share the same label space (homogeneous task setting). In strictly heterogeneous settings, only the sheaf penalty $\mathcal{R}_{\mathcal{A}}$ applies. The coefficient $\mu$ gates this term.

### 5.4 Optimisation Decoupling for Efficient During-Training Communication

The main challenge in optimising $\mathcal{L}_{\text{total}}$ is jointly managing the update of the neural network parameters ${\boldsymbol{\theta}_i}$ and the restriction maps ${\mathbf{V}_{ij}}$, while controlling the trade-off between the different loss terms through the regularisation coefficients $\lambda$ and $\mu$ and deciding how frequently agents need to communicate.

In classical Federated Averaging, the optimisation problem is simpler: each agent minimises its private loss independently, and at the end of each training epoch model parameters are exchanged with neighbours and replaced by a weighted average. Communication therefore occurs only once per epoch, at the parameter level.

In Sheaf-FRL, even though restriction maps are updated only at epoch end, agents still need to exchange pilot representations continuously during training to evaluate the sheaf penalty and the after-communication task loss. This continuous sharing significantly increases the number of communication rounds, and although the transmitted signals are lower-dimensional than full model parameters (only $d_i \times K$ scalars per node), the overhead can become prohibitive in collaborative training with large models.

Two complementary strategies reduce this burden:

- **Reduce signal dimensionality** — transmit fewer pilots. For instance, ComFed and FedProto send only class-wise mean latent representations (prototypes) rather than full pilot matrices.
- **Reduce communication frequency** — communicate less often during training. This is the strategy already used in FedAvg (epoch-end parameter exchange) and in representation-based approaches such as FedMuscle, which alternates between independent local optimisation and a centralised contrastive alignment step.

Taking inspiration from FedMuscle (adapted to the decentralised setting), we split the per-outer-iteration training into three phases:

**Phase A — Private task training** ($T_A$ epochs)

The neural network parameters are optimised solely w.r.t. the private task losses, with an additional sheaf penalty to prevent representations from drifting too far from those of neighbouring nodes. The Phase A objective is:

$$\mathcal{L}_A = \sum_{i \in \mathcal{N}} \Bigl[\underbrace{\mathcal{L}_i(\boldsymbol{\theta}_i)}_{\text{private task loss}} + \lambda,\underbrace{\mathcal{TV}(\mathbf{A}_i)}_{\text{local disagreement}}\Bigr],$$

where the local disagreement is the squared Frobenius distance between the whitened pilot matrix of node $i$ and the alignment-averaged pilot matrices received from its neighbours at the end of the previous epoch:

$$\mathcal{TV}(\mathbf{A}_i) = \bigl|\tilde{\mathbf{A}}_i - \langle\mathbf{A}\rangle_i\bigr|_F^2,$$

$$\tilde{\mathbf{A}}_i = g_{\phi_i}!\bigl(f_{\varphi_i}({\mathbf{x}_i^k}_{k \in \mathcal{A}})\bigr) \in \mathbb{R}^{K \times d_i},$$

$$\langle\mathbf{A}\rangle_i = \frac{1}{N_i}!\left(\sum_{j\in \mathcal{N}(i)^-} \tilde{\mathbf{A}}_j,\mathbf{V}_{ij}^\top + \sum_{j\in \mathcal{N}(i)^+} \tilde{\mathbf{A}}_j,\mathbf{V}_{ji}\right),$$

where $N_i = |\mathcal{N}(i)^- \cup \mathcal{N}(i)^+|$ is the degree of node $i$. The restriction maps ${\mathbf{V}_{ij}}$ are held fixed during this phase.

**Phase B — Whitening refit + Restriction map update** (closed form, one pass)

The whitening statistics $\phi_i$ are refitted from the accumulated pilot buffer (or have already been updated online in the SWBN variant). The Stiefel maps $\mathbf{V}_{ij}$ are then updated via the orthogonal Procrustes closed-form solution applied to the freshly whitened pilot matrices. The encoder parameters ${\boldsymbol{\theta}_i}$ are frozen:

$$\mathbf{V}_{ij}^{\text{new}} = \mathbf{U}\mathbf{W}^\top, \quad [\mathbf{U}, \boldsymbol{\Sigma}, \mathbf{W}^\top] = \operatorname{SVD}!\bigl(\tilde{\mathbf{A}}_i^\top,\tilde{\mathbf{A}}_j\bigr).$$

**Phase C — Semantic communication fine-tuning** ($T_C$ epochs)

Finally, the encoder and decoder parameters are fine-tuned for the semantic communication task. Each node shares a batch of global pilots with its neighbours at every step; the restriction maps are again held fixed. The Phase C objective for node $i$ is:

$$\mathcal{L}_C^i = \lambda,\mathcal{TV}(\mathbf{A}_i) + \mu,\left(\mathcal{L}_{\text{comm}}^{j \to i} + \mathcal{L}_{\text{comm}}^{i \to j}\right),$$

where $\mathcal{L}_{\text{comm}}^{j \to i}$ and $\mathcal{L}_{\text{comm}}^{i \to j}$ are defined in Section 5.3. Both whitening $g_{\phi_i}$ and colouring $g^*_{\phi_i}$ (its closed-form inverse from the same $\phi_i$) are used here, with no additional parameters.

The full pipeline (Phases A $\to$ B $\to$ C) is repeated for $E$ outer iterations.

---

## 6. Evaluation Pipeline: Private vs. Post-Communication Performance

After training, all orchestrators are evaluated on two metrics:

### 6.1 Private Task Performance

Each agent evaluates its own test data through its own encoder and decoder, with no communication:

$$\text{PrivAcc}_i = \frac{1}{|\mathcal{T}_i^{\text{test}}|} \sum_{(\mathbf{x},y) \in \mathcal{T}_i^{\text{test}}} \mathbf{1}\left[h_{\psi_i}(f_{\varphi_i}(\mathbf{x})) = y\right].$$

### 6.2 Post-Communication (Cross-agent) Task Performance

For each directed edge $(i \to j)$: agent $i$ sends its **test latents** to agent $j$ via the `send_message` pipeline, and agent $j$'s decoder classifies them against agent $i$'s ground-truth labels:

$$\text{CommAcc}_{j \leftarrow i} = \frac{1}{|\mathcal{T}_i^{\text{test}}|} \sum_{(\mathbf{x},y) \in \mathcal{T}_i^{\text{test}}} \mathbf{1}!\left[h_{\psi_j}(\operatorname{send_message}(i, j, f_{\varphi_i}(\mathbf{x}))) = y\right].$$

The `send_message` pipeline for **Sheaf-FRL** at test time:

1. **Whiten** sender $i$'s test latents: $\tilde{\mathbf{Z}}_i = g_{\phi_i}(\mathbf{Z}_i)$.
2. **Apply the Stiefel map**: $\hat{\mathbf{Z}} = \tilde{\mathbf{Z}}_i,\mathbf{V}_{ij}^\top$ (or $\tilde{\mathbf{Z}}_i,\mathbf{V}_{ji}$ depending on edge orientation).
3. **Re-colour** with receiver $j$'s colouring map $g^*_{\phi_j}(\hat{\mathbf{Z}})$ — the closed-form inverse of $g_{\phi_j}$, no extra parameters.

Also computed: **task fidelity** $= \text{CommAcc}_{j \leftarrow i} / \text{PrivAcc}_j$, measuring how well cross-agent communication preserves the receiver's task accuracy.

---

## 7. Comparison Baselines and Post-Training Alignment

For methods that **do not apply collaboration on representations during training** (e.g., `NonCooperativeLearning`, `FederatedLearning`, `dPSGD`), a **post-hoc alignment step** is applied at test time before evaluating communication accuracy. This mirrors the single-update version of the Sheaf-FRL Step 1.

### Post-training alignment pipeline

1. **Fit whitening** on each agent's full training latents: obtain $g_{\phi_i}$ and $g^*_{\phi_i}$ via the SVD procedure of Section 5.2.
2. **Fit pairwise alignment maps** on the shared global pilot set:
    - **Procrustes**: solve $\min_{\mathbf{V} \in \mathrm{St}} |\mathbf{A}_i^{\text{white}} - \mathbf{V}\mathbf{A}_j^{\text{white}}|_F^2$ via SVD: $$\mathbf{V}_{j \leftarrow i} = \mathbf{U}\mathbf{W}^\top, \quad [\mathbf{U},\boldsymbol{\Sigma},\mathbf{W}^\top] = \operatorname{SVD}(\mathbf{A}_i^{\text{white}\top} \mathbf{A}_j^{\text{white}}).$$
    - **General least-squares**: solve $\min_\mathbf{A} |\mathbf{X}_i\mathbf{A}^\top - \mathbf{X}_j|_F^2$ in closed form: $$\mathbf{A}_{j \leftarrow i} = \mathbf{X}_j^\top \mathbf{X}_i (\mathbf{X}_i^\top\mathbf{X}_i + \varepsilon\mathbf{I})^{-1}.$$
3. **`send_message`** at test time: $g_{\phi_i}(\cdot)$ → align ($\mathbf{V}_{j \leftarrow i}$) → $g^*_{\phi_j}(\cdot)$.

> This post-hoc Procrustes alignment corresponds exactly to a single Step 1 update of Sheaf-FRL, applied **once after training** rather than at every round. It is a one-shot oracle estimate of the best achievable alignment given frozen encoders.

### Summary of orchestrators and their alignment strategies

|Orchestrator|Collaboration during training|Alignment at test time|
|---|---|---|
|**Sheaf-FRL**|Sheaf penalty + Stiefel maps, every round|Learned Stiefel maps + whitening (online)|
|Sheaf-FMTL|Sheaf TV on model params (compression maps $\mathbf{P}_{ij}$)|None (not representation-level)|
|d-FedU|Graph Laplacian on model params|None|
|d-PSGD|Gossip averaging of model params|Post-hoc Procrustes|
|FedMuscle|Contrastive loss, single shared latent space|Post-hoc Procrustes|
|CoMFed|Prototype alignment, shared compressed space|Post-hoc Procrustes|
|**Non-cooperative**|None (independent local training)|Post-hoc Procrustes (or general LS)|

The key distinction is that **Sheaf-FRL continuously shapes the encoder geometry** toward geometric consistency during training, so no post-hoc alignment step is needed and the learned Stiefel maps already capture the optimal transport between neighbouring latent spaces.

---

## 8. Algorithm 1 Summary (Heterogeneous Sheaf-FRL with Three-Phase Pipeline)

```
Require: Oriented graph G = (N, E), anchor set A (|A| = K), hyperparameters
         λ > 0, μ ≥ 0, η_θ > 0, phase lengths T_A, T_C, outer iterations E

Initialize:  {θ_i^0 = (φ_i^0, ψ_i^0)}_{i ∈ N}               // encoder + decoder params

             // Stiefel restriction maps — truncated identity initialisation:
             // for each edge (i,j) with d_j ≤ d_i (i is head):
             //   V_ij^0  =  I_{d_i}[:, :d_j]  ∈ R^{d_i × d_j}  (first d_j cols of I_{d_i})
             //   V_ji^0  =  I_{d_i}[:d_j, :]  ∈ R^{d_j × d_i}  (first d_j rows of I_{d_i})
             // When d_i = d_j: V_ij^0 = V_ji^0 = I_{d_i}  (full identity, homogeneous case)
             {V_ij^0}_{(i,j) ∈ E}  ←  truncated identities  (see above)

             // Whitening — identity initialisation (no transform at start):
             //   φ_i^0 : mean = 0,  v = 1,  W_i = I_{d_i},  γ = 1,  β = 0
             // Colouring g*_{φ_i} is the closed-form inverse of g_{φ_i}: no extra parameters.
             {φ_i^0}_{i ∈ N}  ←  identity stats

// ── Outer loop ────────────────────────────────────────────────────────────────
for e = 0, ..., E-1 do

  // ══════════════════════════════════════════════════════════════════════════
  // PHASE A — Private task training  (T_A gradient steps, V_ij frozen)
  // ══════════════════════════════════════════════════════════════════════════
  for t = 0, ..., T_A - 1 do

    for all i ∈ N in parallel do
      // Encode pilot batch and whiten
      A_i   ←  f_{φ_i}({x_i^k}_{k ∈ A})                      // raw pilot matrix  [K × d_i]
      Ã_i   ←  g_{φ_i}(A_i)                                   // whiten            [K × d_i]
      //   (SWBN variant: W_i updated in-place during this forward pass, detached)

      // Receive whitened pilots from neighbours (sent end of previous step)
      // and compute alignment-averaged target (V_ij frozen)
      <A>_i  ←  (1/N_i) [ Σ_{j ∈ N(i)^-}  Ã_j  V_ij^⊤
                         + Σ_{j ∈ N(i)^+}  Ã_j  V_ji    ]

      // Local disagreement (sheaf total variation on whitened pilots)
      TV(A_i)  ←  ‖Ã_i − <A>_i‖_F²

      // Phase A loss and gradient step
      // (task loss on non-whitened representations to preserve task information)
      L_A^i  ←  L_i(θ_i) + λ · TV(A_i)
      θ_i    ←  θ_i − η_θ · ∇_{θ_i} L_A^i
      //   (γ_i, β_i updated here; W_i is not touched by this backward pass)

      // Broadcast Ã_i to neighbours for next step's <A>_i computation
    end for

  end for  // Phase A

  // ══════════════════════════════════════════════════════════════════════════
  // PHASE B — Whitening refit + Stiefel map update  (θ_i frozen)
  // ══════════════════════════════════════════════════════════════════════════

  // Step B.1 — refit φ_i on full pilot buffer
  //   buffer-and-fit variant: SVD → update (mean, v, W_i, γ, β)
  //   SWBN variant: φ_i already updated online during Phase A; skip or reset running stats
  for all i ∈ N in parallel do
    A_i   ←  f_{φ_i}({x_i^k}_{k ∈ A})                        // encode with current θ_i
    φ_i   ←  FitWhitening(A_i)                                // updates φ_i in place
    Ã_i   ←  g_{φ_i}(A_i)                                     // whitened pilots [K × d_i]
    //   g*_{φ_i} is now also updated (same φ_i, zero extra cost)
    Broadcast Ã_i to all j ∈ N(i)
  end for

  // Step B.2 — update Stiefel maps via orthogonal Procrustes (closed form)
  for all (i,j) ∈ E in parallel do
    if j ∈ N(i)^-  (i is head, d_j < d_i):
      [U, Σ, W^⊤]  ←  thinSVD(Ã_i^⊤  Ã_j)
      V_ij          ←  U W^⊤                                   // incoming embedding map
    else  (j ∈ N(i)^+, i is tail):
      [U, Σ, W^⊤]  ←  thinSVD(Ã_j^⊤  Ã_i)
      V_ji          ←  U W^⊤                                   // outgoing embedding map
    end if
  end for

  // ══════════════════════════════════════════════════════════════════════════
  // PHASE C — Semantic communication fine-tuning  (T_C steps, V_ij frozen)
  // ══════════════════════════════════════════════════════════════════════════
  for t = 0, ..., T_C - 1 do

    for all i ∈ N in parallel do
      // Encode and whiten pilot batch
      Ã_i  ←  g_{φ_i}( f_{φ_i}({x_i^k}_{k ∈ A}) )           // whitened pilots [K × d_i]

      // Receive Ã_j from all neighbours j ∈ N(i)

      // Recompute local disagreement (V_ij frozen)
      <A>_i    ←  (1/N_i) [ Σ_{j ∈ N(i)^-}  Ã_j  V_ij^⊤
                           + Σ_{j ∈ N(i)^+}  Ã_j  V_ji    ]
      TV(A_i)  ←  ‖Ã_i − <A>_i‖_F²

      // After-communication task loss for each incident edge
      L_comm^i  ←  0
      for each j ∈ N(i) do
        // Direction j → i  (receive from j, decode with i's head)
        Â_{j→i}      ←  Ã_j  V_ij^⊤                          // transport into i's white space
        Â_{j→i}^col  ←  g*_{φ_i}( Â_{j→i} )                 // re-colour: closed-form inverse of g_{φ_i}
        L_comm^i  ←  L_comm^i  +  ℓ_i( h_{ψ_i}(Â_{j→i}^col), y_j )

        // Direction i → j  (send to j, decode with j's head)
        Â_{i→j}      ←  Ã_i  V_ji^⊤                          // transport into j's white space
        Â_{i→j}^col  ←  g*_{φ_j}( Â_{i→j} )                 // re-colour: closed-form inverse of g_{φ_j}
        L_comm^i  ←  L_comm^i  +  ℓ_j( h_{ψ_j}(Â_{i→j}^col), y_i )
      end for

      // Phase C loss and gradient step
      L_C^i  ←  λ · TV(A_i)  +  μ · L_comm^i
      θ_i    ←  θ_i − η_θ · ∇_{θ_i} L_C^i

    end for

  end for  // Phase C

end for  // outer loop
```

> **Notes:**
> 
> - $N_i = |\mathcal{N}(i)^- \cup \mathcal{N}(i)^+|$ is the degree of node $i$; $\mathbf{y}_j$ are the pilot labels of agent $j$.
> - The colouring maps $g^*_{\phi_i}$ and $g^*_{\phi_j}$ carry **no independent parameters**: they are the closed-form inverses of $g_{\phi_i}$ and $g_{\phi_j}$ respectively, sharing the same $\phi_i$, $\phi_j$ and computed via `torch.linalg.solve`.
> - Setting $\mu = 0$ disables Phase C (and the after-communication loss), recovering the two-phase version (A → B only).
> - In Phase A, `<A>_i` is computed from whitened pilots broadcast by neighbours at the **end of the previous gradient step** (held fixed within the step), keeping the update fully local and synchronous.
> - In the **SWBN variant**, `FitWhitening` in Phase B is skipped (or used only to reset running statistics); $\mathbf{W}_i$ is already updated online during Phase A forward passes. In the **buffer-and-fit variant**, `FitWhitening` runs the SVD procedure of Section 5.2 and $\mathbf{W}_i$ is constant within each epoch.
> - The Stiefel maps $\mathbf{V}_{ij}$ are **never** updated in Phases A or C; they remain fixed until the next Phase B.

---