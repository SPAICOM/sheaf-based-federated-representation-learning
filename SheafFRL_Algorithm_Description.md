# Sheaf-based Federated Representation Learning — Algorithm Description

> Source: NeurIPS 2026 submission + `src/orchestrators/sheaf_frl.py`

---

## 1. Problem Setup

Let $\mathcal{G} = (\mathcal{N}, \mathcal{E})$ be a finite undirected graph with $|\mathcal{N}| = N$ agents. Each agent $i \in \mathcal{N}$ holds a private dataset $\mathcal{T}_i$ drawn from a local distribution $P_i$, and trains a **local model** split into:

- a **neural encoder** $f_{\varphi_i} : \mathbb{R}^{p_i} \to \mathcal{F}(i) = \mathbb{R}^{d_i}$
- a **local decoder / task head** $h_{\psi_i} : \mathcal{F}(i) \to \mathcal{Y}_i$

The local empirical risk for agent $i$ is:

$$\mathcal{L}_i(\boldsymbol{\theta}_i) = \frac{1}{M_i} \sum_{n=1}^{M_i} \ell_i\!\left(h_{\psi_i} \circ f_{\varphi_i}(\mathbf{x}_i^n)\right), \quad \boldsymbol{\theta}_i = (\varphi_i, \psi_i). \tag{1}$$

Agents may have **heterogeneous latent dimensions** $d_i \neq d_j$ and **heterogeneous data distributions**. No shared global latent space is assumed.

---

## 2. Network Sheaf and Restriction Maps

To model geometric relationships among local representation spaces, we endow $\mathcal{G}$ with a **network sheaf** $\mathcal{F}$:

- Each node $i$ has a **node stalk** $\mathcal{F}(i) = \mathbb{R}^{d_i}$ — its latent space.
- Each edge $(i,j) \in \mathcal{E}$ has an **edge stalk** $\mathcal{F}(e_{ij}) = \mathbb{R}^{d_i}$ (with $d_{ij} = \max(d_i, d_j)$, choosing $i$ as the head whenever $d_i > d_j$).
- For each incident relation $i \to e_{ij}$ (head node), $\mathcal{F}$ specifies a **linear restriction map** that aligns neighboring latent spaces. In our case the two maps on each edge are:

$$\mathbf{O}_{ji} \in \mathrm{O}(d_i), \qquad \mathbf{V}_{ij} \in \mathrm{St}(d_i, d_j), \tag{2}$$

where $\mathrm{O}(d) = \{\mathbf{O} \in \mathbb{R}^{d\times d} \mid \mathbf{O}^\top\mathbf{O}^{-1}\}$ is the orthogonal group and $\mathrm{St}(d,k) = \{\mathbf{V} \in \mathbb{R}^{d\times k} \mid \mathbf{V}^\top\mathbf{V}=\mathbf{I}_k\}$ is the Stiefel manifold ($k < d$).

> **Semantic embedding principle:** $\mathbf{V}_{ij}$ embeds the lower-dimensional space $\mathcal{F}(j)$ isometrically into $\mathcal{F}(i)$, preserving semantics without compression. The graph is oriented from lower-$d$ nodes to higher-$d$ nodes.

The reparameterization $\mathbf{O}_{ji} = \mathbf{O}_{ij}^\top \mathbf{V}_{ij}$ (following orthogonal group transitivity) simplifies the coboundary to:

$$(\delta \mathbf{z})_{e_{ij}} = \mathbf{O}_{ji}\mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j \xrightarrow{\text{reparametrize}} \mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j. \tag{3,8}$$

---

## 3. Sheaf Total Variation and Gluing Penalty

The **sheaf Laplacian** $\mathbf{L}_\mathcal{F} = \delta^\top\delta$ is symmetric positive semidefinite. The **total variation** of a collection of latents $\mathbf{z} = \{\mathbf{z}_i\}_{i \in \mathcal{N}}$ is:

$$\mathcal{TV}(\mathbf{z}) = \|\delta\mathbf{z}\|_2^2 = \sum_{e_{ij} \in \mathcal{E}} \|\mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j\|_2^2. \tag{7, 9}$$

A **global section** $\mathbf{z}^\star \in \ker(\mathbf{L}_\mathcal{F})$ satisfies $\mathbf{z}_i^\star = \mathbf{V}_{ij}\mathbf{z}_j^\star$ for all edges — perfect geometric alignment. $\mathcal{TV}$ relaxes this hard constraint into a soft penalty.

The local contribution of node $i$ decomposes into **incoming** and **outgoing** embedding terms:

$$\mathcal{TV}(\mathbf{z})|_i = \frac{1}{2}\underbrace{\sum_{j \in \mathcal{N}(i)^-} \|\mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j\|_2^2}_{\text{Incoming embedding}} + \frac{1}{2}\underbrace{\sum_{j \in \mathcal{N}(i)^+} \|\mathbf{z}_j - \mathbf{V}_{ji}\mathbf{z}_i\|_2^2}_{\text{Outgoing embedding}}, \tag{11}$$

where $\mathcal{N}(i)^-$ are neighbors with $d_j < d_i$ and $\mathcal{N}(i)^+$ are those with $d_j > d_i$. To also reflect the similarity with the few existing federated methods working on te representation level instead that on weight spaces, we can define *local section* condition for each edge incident to a particular node $i$ if

$$\mathbf{z}_i = \sum_{j\in \mathcal{N}(i)^-} \mathbf{V}_{ij} \mathbf{z}_j + \sum_{j\in \mathcal{N}(i)^+} \mathbf{V}_{ji}^\top \mathbf{z}_j$$

this formulation directly generalize the definition of local consensus for graphs, where this is obtained when the representation (signal) on a node is equal to the average signal from its neighbors. In this case we have a local section condition on a neighborhood of the netwrok sheaf, if the signal on a node is equal to sort of after-alignment average of the signals from its neighbors $\bar{\mathbf{z}}_i = \sum_{j\in \mathcal{N}(i)^-} \mathbf{V}_{ij} \mathbf{z}_j + \sum_{j\in \mathcal{N}(i)^+} \mathbf{V}_{ji}^\top \mathbf{z}_j$. Then the total variation term we are going to use to the define our penalization term can be also fomrulated as a generalised distributed version of the penalisation term of *FedProto* and similar approaches:

$$\mathcal{TV}(\mathbf{z})|_i = \frac{1}{2} (\mathbf{z}_i - \bar{\mathbf{z}}_i)$$

>**Federated Representation Learning**: note how in *FedProto* local class-prototype representatons are compared with a global mean representation when building the penalization term. It differs from our setup since: (i) global mean representations of the prototypes are defined globally averaging over all the agents in the network, (ii) are aggregated without using any alignment map, thus assuming directly comparable latent spaces. A distributed implementation of FedProto would be indeed a subcase of our sheaf frl setup, where basically all stalks have same dimensionality and restriction maps to be the identity maps. 


### Anchor-restricted (scalable) gluing penalty

Rather than evaluating $\mathcal{TV}$ over all samples, we restrict to a small **anchor set** $\mathcal{A} \subset \{1, \ldots, M_{\min}\}$ with $|\mathcal{A}| = K \ll M_{\min}$. The **anchor feature matrix** of agent $i$ at round $t$ is:

$$\mathbf{A}_i(\varphi_i) = \left[f_{\varphi_i}(\mathbf{x}_i^k)\right]_{k \in \mathcal{A}} \in \mathbb{R}^{d_i \times K}. \tag{12}$$

The local gluing penalty decouples across nodes:

$$\mathcal{R}_{\mathcal{A}}|_i(\varphi_i, \{\mathbf{V}_{ij}\}, \{\mathbf{V}_{ji}\}) = \frac{\lambda}{2K} \left[\sum_{j \in \mathcal{N}(i)^-} \|\mathbf{A}_i(\varphi_i) - \mathbf{V}_{ij}\mathbf{A}_j(\varphi_j)\|_F^2 + \sum_{j \in \mathcal{N}(i)^+} \|\mathbf{A}_j(\varphi_j) - \mathbf{V}_{ji}\mathbf{A}_i(\varphi_i)\|_F^2\right]. \tag{14}$$

The full **SFRL optimization problem** is:

$$\min_{\substack{\{\boldsymbol{\theta}_i = (\varphi_i, \psi_i)\} \\ \{\mathbf{V}_{ij} \in \mathrm{St}(d_i,d_j)\} \\ \{\mathbf{V}_{ji} \in \mathrm{St}(d_j,d_i)\}}} \sum_{i \in \mathcal{N}} \mathcal{L}_i(\boldsymbol{\theta}_i) + \mathcal{R}_\mathcal{A}(\{\varphi_i\}, \{\mathbf{V}_{ij}\}, \{\mathbf{V}_{ji}\}). \tag{SFRL}$$

---

## 4. Sheaf-FRL Algorithm (Alternating Minimization)

The algorithm alternates between two steps per communication round $t$:

### Step 1 — Isometric Embedding Update (closed form)

Each node $i$ broadcasts its anchor matrix $\mathbf{A}_i^t$ to all $j \in \mathcal{N}(i)$. Then, for each edge $e_{ij}$:

- If $j \in \mathcal{N}(i)^-$ (node $i$ is the head, incoming map):
$$\mathbf{V}_{ij}^t = \mathbf{U}\mathbf{W}^\top, \quad \text{where } [\mathbf{U},\boldsymbol{\Sigma},\mathbf{W}^\top] = \text{SVD}(\mathbf{A}_i^t\mathbf{A}_j^{t\top}). \tag{16}$$

- If $j \in \mathcal{N}(i)^+$ (node $i$ is the tail, outgoing map):
$$\mathbf{V}_{ji}^t = \mathbf{U}\mathbf{W}^\top, \quad \text{where } [\mathbf{U},\boldsymbol{\Sigma},\mathbf{W}^\top] = \text{SVD}(\mathbf{A}_j^t\mathbf{A}_i^{t\top}). \tag{16}$$

This is the **orthogonal Procrustes solution** to $\min_{\mathbf{V} \in \mathrm{St}} \|\mathbf{A}_i - \mathbf{V}\mathbf{A}_j\|_F^2$.

> **Remark (homogeneous case):** When $d_i = d$ for all $i$, the maps reduce to canonical orthogonal Procrustes with $\mathbf{O}_{ij}^t = \mathbf{U}\mathbf{W}^\top$ and $\mathbf{O}_{ji} = \mathbf{O}_{ij}^\top$, halving the number of SVD computations (Algorithm 2).

### Step 2 — Neural Parameter Gradient Update

Each agent $i$ updates its local parameters $\boldsymbol{\theta}_i$ via gradient descent, combining the task loss and the sheaf regularization:

$$\nabla_{\varphi_i}\mathcal{R}_\mathcal{A}|_i = \frac{\lambda}{K} \sum_{j \in \mathcal{N}(i)} \sum_{k \in \mathcal{A}} \left(\nabla_{\varphi_i} f_{\varphi_i}(\mathbf{x}_i^k)\right)^\top \!\left(f_{\varphi_i}(\mathbf{x}_i^k) - \mathbf{V}_{ij} f_{\varphi_j}(\mathbf{x}_j^k)\right), \tag{17}$$

with the regularization contribution to the full parameter gradient:

$$\mathbf{r}_i = \begin{bmatrix} \nabla_{\varphi_i}\mathcal{R}_\mathcal{A}|_i \\ \mathbf{0}_{|\psi_i|} \end{bmatrix}, \tag{18}$$

and the local update rule:

$$\boldsymbol{\theta}_i^{t+1} = \boldsymbol{\theta}_i^t - \eta_\theta \left(\nabla_{\boldsymbol{\theta}_i}\mathcal{L}_i(\boldsymbol{\theta}_i^t) + \mathbf{r}_i^t\right). \tag{19}$$

> **Communication cost (Remark 2):** Each agent broadcasts only $\mathbf{A}_i^t \in \mathbb{R}^{d_i \times K}$, transmitting $\mathcal{O}(d_i K)$ scalars per round. Model parameters are never communicated.

---

## 5. Implementation Details

### 5.1 Global Pilot Set as the Anchor Set $\mathcal{A}$

In the implementation (`sheaf_frl.py`), the anchor set $\mathcal{A}$ corresponds to a **global pilot dataset** — a fixed set of $K$ samples shared by all agents in the network (same samples, possibly different modalities per agent). This is the `global_pilot` key in each training batch.

At each step, each agent encodes the pilot batch through its current encoder:

$$\mathbf{A}_i^t = f_{\varphi_i^t}(\{\mathbf{x}_i^k\}_{k \in \mathcal{A}}) \in \mathbb{R}^{K \times d_i}.$$

The pilot latents are accumulated over the epoch into `_pilot_latent_buffer` and used at epoch end (`on_train_epoch_end`) to update the **Stiefel matrices** (restriction maps) via SVD of the whitened cross-covariance.

> The epoch-level accumulation gives more stable cross-covariance estimates than any single mini-batch.

### 5.2 Pre-whitening Before the Sheaf Penalty

Before computing the sheaf penalty $\|\mathbf{z}_i - \mathbf{V}_{ij}\mathbf{z}_j\|^2$ and before updating the Stiefel matrices, the implementation applies **ZCA whitening** to the pilot latents. This is a key deviation from the base paper formulation that significantly stabilizes training.

**Whitening operator** (fit once per epoch on the full pilot buffer, `fit_whitening` in `src/communication/whitening.py`):

Given pilot latents $\mathbf{Z} \in \mathbb{R}^{N \times d}$, fit via SVD of the centered data matrix $\mathbf{Z}_c = \mathbf{Z} - \bar{\mathbf{z}}$:

$$\mathbf{Z}_c = \mathbf{U}\mathbf{S}\mathbf{V}^\top, \quad \lambda_r = \frac{s_r^2}{\max(N-1,1)},$$
$$\mathbf{W} = \mathbf{V}\,\mathrm{diag}(\lambda_r^{-1/2}), \quad \mathbf{C} = \mathbf{V}\,\mathrm{diag}(\lambda_r^{+1/2}).$$

**Whitening / colouring** (right-multiply convention, batch dimension first):

$$\mathbf{z}_{\text{white}} = (\mathbf{z} - \bar{\mathbf{z}})\,\mathbf{W}, \qquad \mathbf{z}_{\text{orig}} = \mathbf{z}_{\text{white}}\,\mathbf{C}^\top + \bar{\mathbf{z}}.$$

The **forward pass sheaf penalty** then operates on whitened representations, while the task loss are still defined by having each decoder working on the non-whitened corresponding encoder representation, avoiding any information loss for the specific task (normalizing the latent space could make more diffucult to do any classification task)

The **Stiefel matrix update** (at epoch end) also operates on whitened accumulated pilots.

**Why whitening?** It removes the scale and correlation structure of each agent's latent space. It is useful since
1. It makes the Procrustes problem (SVD of cross-covariance) well-conditioned and invariant to the particular coordinate system each encoder happens to learn
2. It handles the anisotropic scaling factor that may come with the isometric map connecting two latent spaces of models solving the same task. Similar normalization techinques are indeed used in semantic alingment literature to make relative representation zero-shot alignment methods more robust.
3. Whitening has also been study to facilitate interpretability and concept representation alignment (and potentially steering) as in Barton et al. 2026.

Now to make a smooth aquisition of the whitening (and coloring) statistics one follow two approeaches:
1. Implement a buffer on the training examples seen during an epoch, and at each traning epoch end fit the whitening operator returning the needed statistics, which are going to be used both for the subsequent Steiefel matrices update and the regularization term in the next epoch.
2. Introduce a learnable whitening layer with running statistics wokring similarly to a batch normalization layer, as implemented in Shengdong Zhang et al. 2021 (the whitening learnable layer we like the most from literature).

>*TODO Issue*: We need to efficiently implement the propsed learnable whitening layer and understand if it can be easly extended to be a learnable coloring layer to actually allow easy semantic communications.


### 5.3. After-Communication Task Loss

To allow more alignable and semantic-communication oriented representations, we can also modify the the training objective to actually include an **after-communication task loss** that directly stresses the semantic communication task. This is the loss obtained by decoding pilot samples that have been transported across the network.

For each edge $(i, j)$ with matched pilot rows $(Z_i, y_i, Z_j, y_j)$ in the whitened space:

**Direction $j \to i$** (agent $j$ sends to agent $i$):
1. Map $Z_j$ into $i$'s whitened space: $\hat{Z}_{j \to i} = Z_j \mathbf{V}^{-1} = Z_j \mathbf{V}^\top$ (for Stiefel maps).
2. Re-colour back to $i$'s original space: $\hat{Z}_{j \to i}^{\text{coloured}} = \text{color}(\hat{Z}_{j \to i},\, \text{op}_i)$.
3. Decode with agent $i$'s task head and evaluate against $j$'s labels:

$$\mathcal{L}_{\text{comm}}^{j \to i} = \ell_i\!\left(h_{\psi_i}(\hat{Z}_{j \to i}^{\text{coloured}}),\; y_j\right).$$

**Direction $i \to j$** (symmetric):

$$\mathcal{L}_{\text{comm}}^{i \to j} = \ell_j\!\left(h_{\psi_j}(Z_i \mathbf{V}^{\text{coloured}}),\; y_i\right).$$

The **total loss** is:

$$\mathcal{L}_{\text{total}} = \underbrace{\sum_i \mathcal{L}_i(\boldsymbol{\theta}_i)}_{\text{private task loss}} + \lambda \cdot \mathcal{R}_{\mathcal{A}} + \mu \cdot \underbrace{\mathcal{L}_{\text{comm}}}_{\text{after-comm task loss}}$$

> **Key remark:** This term is only meaningful in non-strictly-heterogeneous setups where agents share the same label space (same task). When label spaces differ, only the pure sheaf penalty applies. The `comm_task_coeff` hyperparameter gates this term.

> In the code this is going to be painful since we need a way to do learnable coloring inverting the learnable whitening layer (?)


### 5.4 Optimisation decoupling for effcient during-training communication

The main issue with the optimisation of $\mathcal{L}_{\text{total}}$ as in the previous section is on how we can actaulli manage to optimise w.r.t. the different optimisation variables (the network parameters and the restriction maps for alignment) and how we handle the trad-off between the different loss terms by using the correct hyperparameters (regularisation coefficients) and deciding how much to communicate during training.

In classical federated averaging, the optimisation solution is easier since one can simply choose to optimise independetly each local loss without the penalisation term on the netwrok parameters, and then update the parameters by doing distributed averaging from neighbours and redistributing back the new smoothed parameters to each nodes. At this point the communication happens just at the end of each training epoch, when each agent has to communicate the set of parameters to its neighbors.

In sheaf frl, even if we update the alignment maps at each traning epoch end, we still need continuous communication among agents, since at each step they send to their neighbors the latent representations of the global pilots to actually derive an updated version of the sheaf penalty term and the after.communication task loss for enabling semantic communications.

This continous sharing of information among agents, would extremely increase the number of communication rounds getting high communication burden although transmitting lower-dimensional signals if compared with fed averaging where they transmit the whole paramter space (this becomes particularly critical in collabortive training setups with huge models).

There are two points to efficient the communication pipeline:
- Send less pilots to reduce the signal diemsnionality being $d_i \times K$ for each node $i$. In ComFed and FedProto they just send the latent prototypes which are class-wise mean latent representations.
- Send these pilots a smaller number of times to reduce the number of communicaton rounds. This is the strategy already implemented in Fed Avg (as stated before, they simply exchange the parameters at each training epcoh end), and also in representation-ased federated learning approaches such as in FedMuscle, where they alternate the indipendent optimisation of the local task losses and of the global/centralised contrastive learning loss for representation alignment.

FedMuscle is a centralized collaborative traning setup of course, but we can get inspiration from its approach by alternating the update of the neural networks parameters in sheaf frl in 3 steps.

**Phase A**
The neural network parameters are optimised for $T_A$ epochs just w.r.t. their private losses, with an additional sheaf penalty term to avoid getting representations that are too distnat from the ones from a node nighbourhood. In this phase the optimisation loss is then:

$$\mathcal{L}_A = \sum_i \underbrace{\mathcal{L}_i(\boldsymbol{\theta}_i)}_{\text{private task loss}} + \lambda \underbrace{\mathcal{TV}(\mathbf{A}_i)}_{\text{local disagreement}},$$

where the local disagreement loss consists of the a total variation formulation from the previous section, representing misalignment of latent spaces over the sheaf. This local disagreement is then expressed as a difference of the whitened pilot representations at specific node $A_i$ with the post-alignment averaged representations coming from the other nodes, sent at the end of the previous traning epoch:

$$\mathcal{TV}(\mathbf{A}) = \|\mathbf{A}_i - \bar{\mathbf{A}}_i\|^2_F,$$

$$\mathbf{A}_i = g_{\phi_i} \circ f_{\varphi_i}(\{\mathbf{x}_i^k\}_{k \in \mathcal{A}}) \in \mathbb{R}^{K \times d_i}.$$

$$\bar{\mathbf{A}}_i = \frac{1}{N_i} (\sum_{j\in \mathcal{N}(i)^-} \mathbf{V}_{ij} \mathbf{A}_j^\top + \sum_{j\in \mathcal{N}(i)^+} \mathbf{V}_{ji}^\top \mathbf{A}_j^\top)$$

We used a notation $N_i=|\mathcal{N}(i)^- \cup\mathcal{N}(i)^+|$ to indicate the number of neighbous for agent $i$, while $K$ is the number of anchors in the anchor set $\mathcal{A}$, and $g_{\phi_i}$ is the learnable whitening layer parameterised by $\phi_i$. Remember that the restriction maps here are considered as fixed.

**Phase B**
Then the restriction maps $\mathbf{V}_{ij}$ for each edge $(i,j) \in \mathcal{E}$ are updated with the closed form solution applied to the shared dataset of global pilots (anchors) in output from the two models (which in this step are considered as freezed). Thus apply the whitening opreation with the learnable

**Phase C**
Finally we have $T_C$ epochs for optimising the nerual parameters for communication-alignemnt task. In this phase the neural models only work on the shared global dataset and each node is shares a batch of global pilots with its neighbours at each step. The optisimation loss for a single node is given by the post-communication task loss plus the sheaf penality term:
$$\mathcal{L}_C^i = \mathcal{TV}(\mathbf{A}_i) + \mathcal{L}_{\text{comm}}^{j \to i} + \mathcal{L}_{\text{comm}}^{i \to j}$$

Then you repeat the whole pipeline for $E$ times.

---

## 7. Evaluation Pipeline: Private vs. Post-Communication Performance

After training, all orchestrators are evaluated on two metrics:

### 7.1 Private Task Performance

Each agent evaluates its own test data through its own encoder and decoder, with no communication:

$$\text{PrivAcc}_i = \frac{1}{|\mathcal{T}_i^{\text{test}}|} \sum_{(\mathbf{x},y) \in \mathcal{T}_i^{\text{test}}} \mathbf{1}\left[h_{\psi_i}(f_{\varphi_i}(\mathbf{x})) = y\right].$$

Logged as `test/private_task_perf_agent_{i}` and averaged as `test/avg_private_task_perf`.

### 7.2 Post-Communication (Cross-agent) Task Performance

For each directed edge $(i \to j)$: agent $i$ sends its **test latents** to agent $j$ via the `send_message` pipeline, and agent $j$'s decoder classifies them against agent $i$'s ground-truth labels:

$$\text{CommAcc}_{j \leftarrow i} = \frac{1}{|\mathcal{T}_i^{\text{test}}|} \sum_{(\mathbf{x},y) \in \mathcal{T}_i^{\text{test}}} \mathbf{1}\left[h_{\psi_j}(\text{send\_message}(i, j, f_{\varphi_i}(\mathbf{x}))) = y\right].$$

The `send_message` pipeline for **Sheaf-FRL** at test time:
1. **Whiten** sender's test latents with sender's training whitening operator $\mathbf{W}_i$.
2. **Apply Stiefel map**: $\tilde{Z} = Z_{\text{white}} \cdot \mathbf{V}_{ij}$ (or $\mathbf{V}_{ji}^\top$ depending on edge direction).
3. **Re-colour** with receiver's training colouring operator $\mathbf{C}_j$.

Logged as `test/comm_task_perf_agent_{j}` and averaged as `test/avg_comm_task_perf`.

Also computed: **task fidelity** $= \text{CommAcc}_{j \leftarrow i} / \text{PrivAcc}_j$, measuring how well cross-agent communication preserves the receiver's task accuracy.

---

## 8. Comparison Baselines and Post-Training Alignment

For methods that **do not apply collaboration on representations during training** (e.g., `NonCooperativeLearning`, `FederatedLearning`, `dPSGD`), a **post-hoc alignment step** is applied at test time before evaluating communication accuracy. This mirrors the single-update version of the Sheaf-FRL Step 1.

### Post-training alignment pipeline (via `PostTrainingAlignmentMixin`)

1. **Fit whitening** on each agent's full training latents (same `fit_whitening` as above).
2. **Fit pairwise alignment maps** on the shared global pilot set:
   - **Procrustes** (`alignment_method='procrustes'`): solve $\min_{\mathbf{V} \in \mathrm{St}} \|\mathbf{A}_i^{\text{white}} - \mathbf{V}\mathbf{A}_j^{\text{white}}\|_F^2$ via SVD:
     $$\mathbf{V}_{j \leftarrow i} = \mathbf{U}\mathbf{W}^\top, \quad [\mathbf{U},\boldsymbol{\Sigma},\mathbf{W}^\top] = \text{SVD}(\mathbf{A}_i^{\text{white}\top} \mathbf{A}_j^{\text{white}}).$$
   - **General least-squares** (`alignment_method='general'`): solve $\min_\mathbf{A} \|\mathbf{X}_i\mathbf{A}^\top - \mathbf{X}_j\|_F^2$ in closed form:
     $$\mathbf{A}_{j \leftarrow i} = \mathbf{X}_j^\top \mathbf{X}_i (\mathbf{X}_i^\top\mathbf{X}_i + \varepsilon\mathbf{I})^{-1}.$$
3. **`send_message`** at test time: whiten → align ($\mathbf{V}_{j \leftarrow i}$) → re-colour.

> This post-hoc Procrustes alignment corresponds exactly to a single Step 1 update of Sheaf-FRL, applied **once after training** rather than at every round. It is a one-shot oracle estimate of the best achievable alignment given frozen encoders.

### Summary of orchestrators and their alignment strategies

| Orchestrator | Collaboration during training | Alignment at test time |
|---|---|---|
| **Sheaf-FRL** | Sheaf penalty + Stiefel maps, every round | Learned Stiefel maps + whitening (online) |
| Sheaf-FMTL | Sheaf TV on model params (compression maps $\mathbf{P}_{ij}$) | None (not representation-level) |
| d-FedU | Graph Laplacian on model params | None |
| d-PSGD | Gossip averaging of model params | Post-hoc Procrustes |
| FedMuscle | Contrastive loss, single shared latent space | Post-hoc Procrustes |
| CoMFed | Prototype alignment, shared compressed space | Post-hoc Procrustes |
| **Non-cooperative** | None (independent local training) | Post-hoc Procrustes (or general LS) |

The key distinction is that **Sheaf-FRL continuously shapes the encoder geometry** toward geometric consistency during training, so no post-hoc alignment step is needed and the learned Stiefel maps already capture the optimal transport between neighbouring latent spaces.

---

## 9. Algorithm 1 Summary (Heterogeneous Sheaf-FRL)

```
Require: Oriented G = (N, E), anchors A, λ > 0, η_θ > 0, iterations T

Initialize {θ_i^0 = (φ_i^0, ψ_i^0)}_{i ∈ N}

for t = 0, ..., T-1 do
  // Step 1: anchor encoding (parallel across nodes)
  for all i ∈ N in parallel do
    A_i^t ← [f_{φ_i^t}(x_i^k)]_{k ∈ A}         // encode global pilots
  end for
  
  // Broadcast anchor matrices to neighbours
  Each node i broadcasts A_i^t to those j ∈ N(i)

  // Stiefel map update (parallel across nodes)
  for all i ∈ N in parallel do
    for j ∈ N(i) do
      if j ∈ N(i)^-  (i is head, d_j < d_i):
        [U,Σ,W^T] ← thinSVD(A_i^t A_j^{t⊤})
        V_ij^t ← UW^T                             // incoming embedding map
      else  (j ∈ N(i)^+, i is tail):
        [U,Σ,W^T] ← thinSVD(A_j^t A_i^{t⊤})
        V_ji^t ← UW^T                             // outgoing embedding map
    end for
  end for

  // Step 2: neural parameter update (parallel across nodes)
  for all i ∈ N in parallel do
    θ_i^{t+1} ← θ_i^t - η_θ (∇_{θ_i} L_i(θ_i^t) + r_i^t)
  end for
end for
```

> **Implementation note:** In `sheaf_frl.py`, Step 1 occurs over the full training epoch (mini-batch accumulation in `_pilot_latent_buffer`), and the Stiefel update runs in `on_train_epoch_end`. The gradient step (Step 2) runs per mini-batch in `training_step` → `_shared_eval`. Whitening operators are re-fitted at each epoch end on accumulated pilot latents (`_task_latent_buffer`) and stored in `self._whitening_ops` for use in the next epoch's forward pass.

---

*Generated 2026-06-23 from NeurIPS_2026.pdf and src/orchestrators/sheaf_frl.py*
