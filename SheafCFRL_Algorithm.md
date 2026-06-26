## 9. Alternative Algorithm: Semantic Compression
 
### 9.1 Compressed Edge Stalks
 
In the main algorithm, the edge stalk dimension is set to $\operatorname{dim}(\mathcal{F}(e_{ij})) = \max(d_i, d_j)$, which allows one restriction map to embed the smaller space isometrically into the larger one. This underpins the reparameterisation trick that simplifies the coboundary to $\mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j$.
 
An alternative is to deliberately **compress** the representation over each edge, setting:
 
$$\operatorname{dim}(\mathcal{F}(e_{ij})) = c_{ij} < \min(d_i, d_j).$$
 
Both restriction maps now project down to a strictly lower-dimensional shared space. The coboundary on edge $e_{ij}$ becomes:
 
$$(\delta\mathbf{z})_{e_{ij}} = \mathbf{V}_{ji}\mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j \in \mathbb{R}^{c_{ij}},$$
 
with **fat semi-orthogonal** restriction maps:
 
$$\mathbf{V}_{ij} \in \mathbb{R}^{c_{ij} \times d_j}, \quad \mathbf{V}_{ij}^\top \in \mathrm{St}(d_j, c_{ij}), \quad \mathbf{V}_{ij}\mathbf{V}_{ij}^\top = \mathbf{I}_{c_{ij}},$$
 
and symmetrically $\mathbf{V}_{ji} \in \mathbb{R}^{c_{ij} \times d_i}$ with $\mathbf{V}_{ji}\mathbf{V}_{ji}^\top = \mathbf{I}_{c_{ij}}$. The matrix $\mathbf{V}_{ij}^\top\mathbf{V}_{ij}$ is the orthogonal projector from $\mathcal{F}(j) = \mathbb{R}^{d_j}$ onto the $c_{ij}$-dimensional subspace mapped to the edge stalk. The local section condition is $\mathbf{V}_{ji}\mathbf{z}_i = \mathbf{V}_{ij}\mathbf{z}_j$.
 
> **Remark (no natural orientation):** In the main algorithm the graph is oriented from lower-$d$ to higher-$d$ nodes, and the asymmetry of the maps reflects this. Here, with $c_{ij} < \min(d_i, d_j)$, both maps are of the same type and there is no orientation induced by the latent dimensions. The graph orientation must be chosen by other criteria, e.g. communication cost or task hierarchy.
 
### 9.2 Local Loss and Correct Problem Formulation
 
With compression, the local objective for edge $(i,j)$ combines **semantic misalignment** — agreement in the edge stalk — with **communication loss** — how well each agent can reconstruct the other's representation after the round-trip through the edge stalk:
 
$$\mathcal{L}^\beta_{ij}(\mathbf{V}_{ij}, \mathbf{V}_{ji}) = (1-\beta)\underbrace{\|\mathbf{V}_{ij}\tilde{\mathbf{A}}_j - \mathbf{V}_{ji}\tilde{\mathbf{A}}_i\|_F^2}_{\text{semantic misalignment}} + \beta\underbrace{\bigl(\|\tilde{\mathbf{A}}_j - \mathbf{V}_{ij}^\top\mathbf{V}_{ji}\tilde{\mathbf{A}}_i\|_F^2 + \|\tilde{\mathbf{A}}_i - \mathbf{V}_{ji}^\top\mathbf{V}_{ij}\tilde{\mathbf{A}}_j\|_F^2\bigr)}_{\text{communication loss}},$$
 
where $0 < \beta < 1$ controls the trade-off and $\tilde{\mathbf{A}}_i, \tilde{\mathbf{A}}_j \in \mathbb{R}^{K \times d}$ are the whitened pilot matrices (Section 5.2). The local optimisation problem is:
 
$$\min_{\substack{\mathbf{V}_{ij}^\top \in \mathrm{St}(d_j,\, c) \\ \mathbf{V}_{ji}^\top \in \mathrm{St}(d_i,\, c)}} \mathcal{L}^\beta_{ij}(\mathbf{V}_{ij}, \mathbf{V}_{ji}). \tag{P2}$$
 
> **Why the alternating Procrustes solution is wrong here.** When $\beta = 0$, fixing $\mathbf{V}_{ji}$ and minimising over $\mathbf{V}_{ij}$ alone would indeed reduce to a one-sided Procrustes problem. But for $\beta > 0$, the communication loss terms $\|\tilde{\mathbf{A}}_j - \mathbf{V}_{ij}^\top\mathbf{V}_{ji}\tilde{\mathbf{A}}_i\|_F^2$ and $\|\tilde{\mathbf{A}}_i - \mathbf{V}_{ji}^\top\mathbf{V}_{ij}\tilde{\mathbf{A}}_j\|_F^2$ couple $\mathbf{V}_{ij}$ and $\mathbf{V}_{ji}$ non-linearly — one appears transposed inside a product with the other. Fixing $\mathbf{V}_{ji}$ and optimising over $\mathbf{V}_{ij}$ does **not** give a Procrustes problem; it gives a Sylvester equation. The Stiefel constraints then cannot be enforced directly and must be handled via an auxiliary variable splitting (ADMM).
 
### 9.3 Phase B: ADMM Updates
 
Following the Splitting of Orthogonality Constraints (SOC) method, we introduce auxiliary variables $\mathbf{Y}_i \in \mathrm{St}(d_i, c)$ and $\mathbf{Y}_j \in \mathrm{St}(d_j, c)$ via the equality constraints $\mathbf{V}_{ji}^\top = \mathbf{Y}_i$ and $\mathbf{V}_{ij}^\top = \mathbf{Y}_j$, separating the Stiefel constraints from the smooth quadratic sub-problems. Let $\mathbf{U}_i, \mathbf{U}_j$ be the corresponding scaled dual variables. The scaled augmented Lagrangian is:
 
$$\mathcal{L}_\alpha = \frac{1-\beta}{2}\|\mathbf{V}_{ij}\tilde{\mathbf{A}}_j - \mathbf{V}_{ji}\tilde{\mathbf{A}}_i\|_F^2 + \frac{\beta}{2}\|\tilde{\mathbf{A}}_j - \mathbf{V}_{ij}^\top\mathbf{V}_{ji}\tilde{\mathbf{A}}_i\|_F^2 + \frac{\beta}{2}\|\tilde{\mathbf{A}}_i - \mathbf{V}_{ji}^\top\mathbf{V}_{ij}\tilde{\mathbf{A}}_j\|_F^2 + \frac{\alpha}{2}\|\mathbf{V}_{ji}^\top - \mathbf{Y}_i + \mathbf{U}_i\|_F^2 + \frac{\alpha}{2}\|\mathbf{V}_{ij}^\top - \mathbf{Y}_j + \mathbf{U}_j\|_F^2.$$
 
Denote the pilot covariances $\boldsymbol{\Sigma}_i = \tilde{\mathbf{A}}_i^\top\tilde{\mathbf{A}}_i \in \mathbb{R}^{d_i \times d_i}$, $\boldsymbol{\Sigma}_j = \tilde{\mathbf{A}}_j^\top\tilde{\mathbf{A}}_j \in \mathbb{R}^{d_j \times d_j}$, and the cross-covariances $\boldsymbol{\Sigma}_{ji} = \tilde{\mathbf{A}}_j^\top\tilde{\mathbf{A}}_i \in \mathbb{R}^{d_j \times d_i}$, $\boldsymbol{\Sigma}_{ij} = \tilde{\mathbf{A}}_i^\top\tilde{\mathbf{A}}_j \in \mathbb{R}^{d_i \times d_j}$.
 
The ADMM recursion at iteration $k$ consists of four closed-form steps.
 
**Step 1 — Update $\mathbf{V}_{ji}$** (Sylvester equation).
 
Setting $\partial\mathcal{L}_\alpha/\partial\mathbf{V}_{ji} = 0$ with $\mathbf{V}_{ij}$ fixed, all three loss terms contribute:
 
- **Term 1** ($\frac{1-\beta}{2}\|\mathbf{V}_{ij}\tilde{\mathbf{A}}_j - \mathbf{V}_{ji}\tilde{\mathbf{A}}_i\|_F^2$): $\quad-(1-\beta)\mathbf{V}_{ij}\boldsymbol{\Sigma}_{ji} + (1-\beta)\mathbf{V}_{ji}\boldsymbol{\Sigma}_i$
- **Term 2** ($\frac{\beta}{2}\|\tilde{\mathbf{A}}_j - \mathbf{V}_{ij}^\top\mathbf{V}_{ji}\tilde{\mathbf{A}}_i\|_F^2$): $\quad-\beta\mathbf{V}_{ij}\boldsymbol{\Sigma}_{ji} + \beta\mathbf{V}_{ij}\mathbf{V}_{ij}^\top\mathbf{V}_{ji}\boldsymbol{\Sigma}_i$
- **Term 3** ($\frac{\beta}{2}\|\tilde{\mathbf{A}}_i - \mathbf{V}_{ji}^\top\mathbf{V}_{ij}\tilde{\mathbf{A}}_j\|_F^2$, differentiating through the transposed $\mathbf{V}_{ji}^\top$ using $\frac{\partial}{\partial\mathbf{V}_{ji}}\operatorname{tr}[\mathbf{V}_{ji}\mathbf{V}_{ji}^\top\mathbf{M}\mathbf{M}^\top] = 2\mathbf{M}\mathbf{M}^\top\mathbf{V}_{ji}$ with $\mathbf{M}=\mathbf{V}_{ij}\tilde{\mathbf{A}}_j$): $\quad-\beta\mathbf{V}_{ij}\boldsymbol{\Sigma}_{ji} + \beta\mathbf{V}_{ij}\boldsymbol{\Sigma}_j\mathbf{V}_{ij}^\top\mathbf{V}_{ji}$
- **Penalty** ($\frac{\alpha}{2}\|\mathbf{V}_{ji}^\top - \mathbf{Y}_i + \mathbf{U}_i\|_F^2$): $\quad\alpha\mathbf{V}_{ji} - \alpha(\mathbf{Y}_i - \mathbf{U}_i)^\top$
Collecting terms in $\mathbf{V}_{ji}$ on the left — note Term 2 multiplies $\mathbf{V}_{ji}$ on the right by $\boldsymbol{\Sigma}_i$, Term 3 multiplies on the left by $\mathbf{V}_{ij}\boldsymbol{\Sigma}_j\mathbf{V}_{ij}^\top$, and the penalty multiplies by $\mathbf{I}_{d_i}$:
 
$$\underbrace{\bigl[(1-\beta)\mathbf{I}_c + \beta\mathbf{V}_{ij}^k\mathbf{V}_{ij}^{k\top}\bigr]}_{\mathbf{A}_j}\mathbf{V}_{ji}\,\boldsymbol{\Sigma}_i + \underbrace{\bigl[\alpha\mathbf{I}_c + \beta\mathbf{V}_{ij}^k\boldsymbol{\Sigma}_j\mathbf{V}_{ij}^{k\top}\bigr]}_{\mathbf{B}_j}\mathbf{V}_{ji} = \underbrace{(1+\beta)\mathbf{V}_{ij}^k\boldsymbol{\Sigma}_{ji} + \alpha(\mathbf{Y}_i^k-\mathbf{U}_i^k)^\top}_{\mathbf{C}_j}. \tag{S1}$$
 
This is a Sylvester equation $\mathbf{A}_j\mathbf{V}_{ji}\boldsymbol{\Sigma}_i + \mathbf{B}_j\mathbf{V}_{ji} = \mathbf{C}_j$. Left-multiplying by $\mathbf{B}_j^{-1}$ gives the standard $PX + XQ = R$ form:
 
$$(\mathbf{B}_j^{-1}\mathbf{A}_j)\,\mathbf{V}_{ji}\,\boldsymbol{\Sigma}_i + \mathbf{V}_{ji} = \mathbf{B}_j^{-1}\mathbf{C}_j,$$
 
solved via the Bartels–Stewart algorithm. Note that $\mathbf{B}_j \succ 0$ since $\alpha > 0$ and $\beta\mathbf{V}_{ij}\boldsymbol{\Sigma}_j\mathbf{V}_{ij}^\top \succeq 0$.
 
**Step 2 — Update $\mathbf{V}_{ij}$** (symmetric to Step 1, using $\mathbf{V}_{ji}^{k+1}$ just computed):
 
$$\mathbf{A}_i\,\mathbf{V}_{ij}\,\boldsymbol{\Sigma}_j + \mathbf{B}_i\,\mathbf{V}_{ij} = \mathbf{C}_i, \tag{S2}$$
 
$$\mathbf{A}_i = (1-\beta)\mathbf{I}_c + \beta\mathbf{V}_{ji}^{k+1}\mathbf{V}_{ji}^{k+1\top}, \quad \mathbf{B}_i = \alpha\mathbf{I}_c + \beta\mathbf{V}_{ji}^{k+1}\boldsymbol{\Sigma}_i\mathbf{V}_{ji}^{k+1\top},$$
$$\mathbf{C}_i = (1+\beta)\mathbf{V}_{ji}^{k+1}\boldsymbol{\Sigma}_{ij} + \alpha(\mathbf{Y}_j^k - \mathbf{U}_j^k)^\top.$$
 
**Step 3 — Update $\mathbf{Y}_i$ and $\mathbf{Y}_j$** (Stiefel projection via polar decomposition):
 
$$\mathbf{Y}_i^{k+1} = \operatorname{prox}_{\mathrm{St}(d_i,c)}\!\bigl(\mathbf{V}_{ji}^{k+1\top} + \mathbf{U}_i^k\bigr), \qquad \mathbf{Y}_j^{k+1} = \operatorname{prox}_{\mathrm{St}(d_j,c)}\!\bigl(\mathbf{V}_{ij}^{k+1\top} + \mathbf{U}_j^k\bigr).$$
 
The proximal step equals the $\mathbf{U}_p$ factor of the polar decomposition of the argument. For $\widetilde{\mathbf{Y}} \in \mathbb{R}^{d \times c}$: compute $[\mathbf{U}, \cdot, \mathbf{W}^\top] = \operatorname{thinSVD}(\widetilde{\mathbf{Y}})$, then $\operatorname{prox}_{\mathrm{St}}(\widetilde{\mathbf{Y}}) = \mathbf{U}\mathbf{W}^\top \in \mathrm{St}(d, c)$.
 
**Step 4 — Dual variable updates:**
 
$$\mathbf{U}_i^{k+1} \mathrel{+}= \mathbf{V}_{ji}^{k+1\top} - \mathbf{Y}_i^{k+1}, \qquad \mathbf{U}_j^{k+1} \mathrel{+}= \mathbf{V}_{ij}^{k+1\top} - \mathbf{Y}_j^{k+1}.$$
 
The full Phase B pseudocode for a single edge $(i,j)$:
 
```
// Phase B (compressed), per edge (i,j)
// Requires: Σ_i, Σ_j, Σ_ij, Σ_ji precomputed from whitened pilots Ã_i, Ã_j
// Initialise: Y_i, Y_j, U_i, U_j  (see Section 9.4)
 
for k = 0, ..., T_B - 1 do
 
  // Step 1: update V_ji via Sylvester equation (S1)
  A_j  ←  (1-β) I_c  +  β V_ij^k V_ij^k⊤                    // c × c
  B_j  ←  α I_c      +  β V_ij^k Σ_j V_ij^k⊤                // c × c
  C_j  ←  (1+β) V_ij^k Σ_{ji}  +  α (Y_i^k - U_i^k)⊤        // c × d_i
  // Solve:  (B_j^{-1} A_j) V_ji Σ_i  +  V_ji  =  B_j^{-1} C_j
  V_ji^{k+1}  ←  BartelsStewart( B_j^{-1} A_j,  Σ_i,  B_j^{-1} C_j )
 
  // Step 2: update V_ij via Sylvester equation (S2)
  A_i  ←  (1-β) I_c  +  β V_ji^{k+1} V_ji^{k+1⊤}            // c × c
  B_i  ←  α I_c      +  β V_ji^{k+1} Σ_i V_ji^{k+1⊤}        // c × c
  C_i  ←  (1+β) V_ji^{k+1} Σ_{ij}  +  α (Y_j^k - U_j^k)⊤   // c × d_j
  // Solve:  (B_i^{-1} A_i) V_ij Σ_j  +  V_ij  =  B_i^{-1} C_i
  V_ij^{k+1}  ←  BartelsStewart( B_i^{-1} A_i,  Σ_j,  B_i^{-1} C_i )
 
  // Step 3: project onto Stiefel via polar decomposition
  [U, _, W⊤]  ←  thinSVD( V_ji^{k+1⊤} + U_i^k )              // d_i × c input
  Y_i^{k+1}   ←  U W⊤                                         // ∈ St(d_i, c)
 
  [U, _, W⊤]  ←  thinSVD( V_ij^{k+1⊤} + U_j^k )              // d_j × c input
  Y_j^{k+1}   ←  U W⊤                                         // ∈ St(d_j, c)
 
  // Step 4: dual variable updates
  U_i^{k+1}  +=  V_ji^{k+1⊤} - Y_i^{k+1}                     // d_i × c
  U_j^{k+1}  +=  V_ij^{k+1⊤} - Y_j^{k+1}                     // d_j × c
 
end for
 
// After convergence, read off Stiefel-feasible maps:
V_ji  ←  Y_i^⊤    // c × d_i,  V_ji V_ji⊤ = I_c  ✓
V_ij  ←  Y_j^⊤    // c × d_j,  V_ij V_ij⊤ = I_c  ✓
```
 
> **Remark (Phase B now iterative with $T_B$ inner steps):** In the main algorithm Phase B is a single closed-form SVD pass. Here it requires $T_B$ ADMM iterations per edge, each involving two Sylvester solves (Bartels–Stewart, $\mathcal{O}(c^3 + c \cdot d)$ per solve since $c \ll d$) and two thin SVDs of $d \times c$ matrices. In practice $T_B = 10$–$30$ iterations is sufficient for convergence of the primal residuals $\|\mathbf{V}_{ji}^\top - \mathbf{Y}_i\|_F$ and $\|\mathbf{V}_{ij}^\top - \mathbf{Y}_j\|_F$.
 
### 9.4 Initialisation of the Compressed Restriction Maps
 
Initialisation matters more here than in the main algorithm because the ADMM sub-problems are non-convex and the Sylvester equations are solved with fixed other-variable estimates.
 
**Restriction maps** $\mathbf{V}_{ij}^0$, $\mathbf{V}_{ji}^0$: initialise with the **top-$c$ left singular vectors** of the respective whitened pilot matrix:
 
$$\mathbf{V}_{ij}^0 = \mathbf{U}_j^{(c)\top}, \quad [\mathbf{U}_j^{(c)}, \cdot, \cdot] = \operatorname{thinSVD}_c(\tilde{\mathbf{A}}_j),$$
 
and symmetrically $\mathbf{V}_{ji}^0 = \mathbf{U}_i^{(c)\top}$. Here $\mathbf{U}_j^{(c)} \in \mathbb{R}^{K \times c}$ contains the top-$c$ left singular vectors of the $K \times d_j$ pilot matrix, so $\mathbf{V}_{ij}^0 \in \mathbb{R}^{c \times d_j}$ is not obtained directly from the SVD but rather requires an additional step: $\mathbf{V}_{ij}^0 = \operatorname{thinSVD}_c(\tilde{\mathbf{A}}_j^\top).\mathbf{U}^\top$, i.e. the top-$c$ right singular vectors of $\tilde{\mathbf{A}}_j$ transposed. This aligns each map with the $c$ principal directions of the agent's pilot distribution, and satisfies $\mathbf{V}_{ij}^0\mathbf{V}_{ij}^{0\top} = \mathbf{I}_c$ exactly. A truncated identity (first $c$ rows of $\mathbf{I}_{d_j}$) is a simpler fallback.
 
**Auxiliary variables:** $\mathbf{Y}_i^0 = \mathbf{V}_{ji}^{0\top}$, $\mathbf{Y}_j^0 = \mathbf{V}_{ij}^{0\top}$ (already on the Stiefel manifold by construction).
 
**Dual variables:** $\mathbf{U}_i^0 = \mathbf{0}_{d_i \times c}$, $\mathbf{U}_j^0 = \mathbf{0}_{d_j \times c}$.
 
> **Remark (relationship to the main algorithm).** The main algorithm's single-SVD Phase B is recovered as a special case when $\beta = 0$ (semantic misalignment only) and $c = \max(d_i, d_j)$ (embedding rather than compression). In that limit the Sylvester equations degenerate to Procrustes problems and the auxiliary variable steps are trivially satisfied. For $\beta > 0$ and $c < \min(d_i, d_j)$, the full ADMM recursion above is required.
 
### 9.5 Modified Three-Phase Pipeline (Semantic Compression)
 
The three-phase pipeline of Algorithm 1 carries over with three modifications: (i) the sheaf penalty uses the compressed coboundary; (ii) Phase B runs $T_B$ ADMM iterations per edge; (iii) `send_message` routes through the compressed edge stalk.
 
**Compressed sheaf penalty:**
 
$$\mathcal{TV}_c(\mathbf{A}_i) = \frac{1}{N_i}\sum_{j \in \mathcal{N}(i)} \|\mathbf{V}_{ji}\tilde{\mathbf{A}}_i - \mathbf{V}_{ij}\tilde{\mathbf{A}}_j\|_F^2.$$
 
**`send_message` pipeline (compressed):**
 
$$\operatorname{send\_message}_c(i, j, \mathbf{z}_i) = g^*_{\phi_j}\!\bigl(\mathbf{V}_{ij}^\top\mathbf{V}_{ji}\,g_{\phi_i}(\mathbf{z}_i)\bigr),$$
 
where $\mathbf{V}_{ji} \in \mathbb{R}^{c \times d_i}$ projects from $\mathcal{F}(i)$ into the edge stalk $\mathbb{R}^c$, and $\mathbf{V}_{ij}^\top \in \mathbb{R}^{d_j \times c}$ lifts back to $\mathcal{F}(j)$ before re-colouring. The composition $\mathbf{V}_{ij}^\top\mathbf{V}_{ji} \in \mathbb{R}^{d_j \times d_i}$ is the full (lossy) transport map from sender to receiver.