import torch
import torch.nn as nn

from src.orchestrators.base_orchestrator import BaseOrchestrator

class SheafFRL(BaseOrchestrator):
    def __init__(
        self,
        agents: dict[int, nn.Module],
        neighbors: dict[int, set[int]],
        optimizer,
        lambda_sheaf: float,
        latent_dims: dict,
        anchor_strategy: str,
        num_anchors: int,
        parseval_normalization: bool,
        **kwargs,
    ):
        super().__init__(
            agents=agents,
            neighbors=neighbors,
            optimizer=optimizer,
        )

        self.lambda_sheaf = lambda_sheaf
        self.anchor_strategy = anchor_strategy
        self.num_anchors = num_anchors
        self.stiefel_matrices = nn.ParameterDict()
        self.epoch_anchors = {} 
        self.epoch_labels = []
        self.parseval_normalization = parseval_normalization

        # Force latent_dims to have integer keys and values
        self.latent_dims = {int(k): int(v) for k, v in latent_dims.items()}

        # Loop through neighbors and force i and j to be integers
        for i_raw, neighborset in neighbors.items():
            for j_raw in neighborset:
                i = int(i_raw)
                j = int(j_raw)

                if self.latent_dims[i] > self.latent_dims[j]:
                    node_i, node_j = i, j
                elif self.latent_dims[i] < self.latent_dims[j]:
                    node_i, node_j = j, i
                else:
                    node_i, node_j = max(i, j), min(i, j)
                
                edge_key = f"{node_i}_{node_j}"

                if edge_key not in self.stiefel_matrices:
                    d_i = self.latent_dims[node_i]
                    d_j = self.latent_dims[node_j]

                    stiefel_matrix = torch.eye(d_i, d_j)
                    
                    self.stiefel_matrices[edge_key] = nn.Parameter(
                        stiefel_matrix, requires_grad=False
                    )

    def _compute_anchors(self, latents: dict[int, torch.Tensor], labels: torch.Tensor) -> dict[int, torch.Tensor]:
        """Computes or selects the final anchor representations based on the strategy."""

        tot = len(labels)
        uniques = torch.unique(labels)

        if self.anchor_strategy == "prototype":
            A_proto = {int(idx): [] for idx in self.agents.keys()}
            
            for c in uniques:
                c_mask = (labels == c)
                for idx in self.agents.keys():
                    idx = int(idx)
                    mean_anchor = latents[idx][c_mask].mean(dim=0)
                    A_proto[idx].append(mean_anchor)
            
            return {idx: torch.stack(protos, dim=0) for idx, protos in A_proto.items()}

        elif self.anchor_strategy == "balanced":
            k = min(self.num_anchors, tot)
            anchors_per_class = k // len(uniques)

            selected_indices = []
            for c in uniques:
                c_idx = torch.where(labels == c)[0]
                perm = torch.randperm(len(c_idx), device=labels.device)
                chosen = c_idx[perm[:anchors_per_class]]
                selected_indices.append(chosen)
                
            selected_indices = torch.cat(selected_indices)
            
            remaining = k - len(selected_indices)
            if remaining > 0:
                all_idx = torch.arange(tot, device=labels.device)
                mask = torch.ones(tot, dtype=torch.bool, device=labels.device)
                mask[selected_indices] = False
                avail_idx = all_idx[mask]
                
                extra_perm = torch.randperm(len(avail_idx), device=labels.device)
                extra = avail_idx[extra_perm[:remaining]]
                selected_indices = torch.cat([selected_indices, extra])
                
            final_perm = torch.randperm(len(selected_indices), device=labels.device)
            anchor_indices = selected_indices[final_perm]
            
            return {idx: A[anchor_indices] for idx, A in latents.items()}

        elif self.anchor_strategy == "random":
            k = min(self.num_anchors, tot)
            perm = torch.randperm(tot, device=labels.device)
            anchor_indices = perm[:k]
            return {idx: A[anchor_indices] for idx, A in latents.items()}
            
        else:
            return latents

    def on_train_epoch_start(self) -> None:
        """Initialize/Reset the anchor (and its label) lists at the start of each epoch."""
        for idx_str in self.agents.keys():
            self.epoch_anchors[int(idx_str)] = []
        self.epoch_labels = []
    
    def parseval_normalize(self, A: torch.Tensor) -> torch.Tensor:
        """Apply Parseval normalization to the anchors features as local whitening """
        C = torch.matmul(A.T, A)
        eps = 1e-4
        C = C + eps * torch.eye(C.size(0), device=C.device)
        eigenvalues, eigenvectors = torch.linalg.eigh(C)

        inv_sqrt_eigenvalues = torch.rsqrt(eigenvalues.clamp(min=eps))

        C_inv = torch.matmul(eigenvectors, inv_sqrt_eigenvalues.unsqueeze(1) * eigenvectors.T)

        A_normalized = torch.matmul(A, C_inv)

        return A_normalized

    @torch.no_grad()
    def on_train_epoch_end(self) -> None:
        if not self.epoch_anchors or not any(self.epoch_anchors.values()):
            return  # No anchors to update

        A_dict = {}
        for idx_str in self.agents.keys():
            idx = int(idx_str)
            if self.epoch_anchors[idx]: 
                A_dict[idx] = torch.cat(self.epoch_anchors[idx], dim=0)
        
        labels = torch.cat(self.epoch_labels, dim=0)
        A_dict = self._compute_anchors(A_dict, labels)
            
        for edge_key, V_param in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in A_dict or node_j not in A_dict:
                continue

            A_i_unbiased = A_dict[node_i] - A_dict[node_i].mean(dim=0, keepdim=True)
            A_j_unbiased = A_dict[node_j] - A_dict[node_j].mean(dim=0, keepdim=True)

            C = torch.matmul(A_i_unbiased.T, A_j_unbiased)
            U, S, W_T = torch.linalg.svd(C, full_matrices=False)
            V_new = torch.matmul(U, W_T) 

            ## closed form update for the frozen Stiefel matrix
            V_param.copy_(V_new.to(V_param.device))

        for idx in self.epoch_anchors.keys():
            self.epoch_anchors[idx].clear()

        self.epoch_labels.clear()

    def _shared_eval(
        self,
        batch: dict[int, list[torch.Tensor]],
        batch_idx: int,
        prefix: str,
    ):
        """Shared evaluation logic for validation and test steps."""
        outputs = {}

        total_task_loss = 0.0
        total_task_performance = 0.0

        batch_latents = {}

        # Compute the personalized loss for each agent
        for idx_str, agent in self.agents.items():
            idx = int(idx_str)
            x_key = str(idx) if str(idx) in batch else idx
            x, y = batch[x_key] 

            y_hat, A_i_raw = agent.forward_with_features(x)
            outputs[idx] = (y_hat, y)
            
            # parseval normalization
            if self.parseval_normalization:
                A_i_tilde = self.parseval_normalize(A_i_raw)
            else:
                A_i_tilde = A_i_raw

            # store the latents for sheaf gluing penalty  (now computed for every sample, needs to be refined)          
            batch_latents[idx] = A_i_tilde

            if prefix == 'train':
                self.epoch_anchors[idx].append(A_i_tilde.detach())
                if idx == 0:
                    self.epoch_labels.append(y.detach())

            task_loss = agent.compute_loss(y_hat, y)
            task_performance = agent.task_performance(y_hat, y)

            self.log(
                f'{prefix}/task_loss_agent_{idx}',
                task_loss,
                on_step=False,
                on_epoch=True,
            )

            self.log(
                f'{prefix}/task_performance_agent_{idx}',
                task_performance,
                on_step=False,
                on_epoch=True,
            )

            total_task_loss += task_loss
            total_task_performance += task_performance

        # Compute the sheaf gluing penalty
        sheaf_penalty = 0.0

        for edge_key, V in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))
            node_i, node_j = int(node_i), int(node_j)


            diff = batch_latents[node_i] - torch.matmul(batch_latents[node_j], V.T)
            frob_dist = (diff**2).sum(dim=1).mean()
            sheaf_penalty += frob_dist
        
        self.log(
            f'{prefix}/sheaf_penalty',
            sheaf_penalty,
            on_step=False,
            on_epoch=True
        )
    
        total_loss = total_task_loss + self.lambda_sheaf * sheaf_penalty
        avg_performance = total_task_performance / len(self.agents)

        self.log(
            f'{prefix}/total_loss_epoch',
            total_loss,
            on_step=False,
            on_epoch=True,
        )       

        self.log(
            f'{prefix}/avg_task_performance_epoch',
            avg_performance,
            on_step=False,
            on_epoch=True,
        )

        return outputs, total_loss

if __name__ == '__main__':
    pass