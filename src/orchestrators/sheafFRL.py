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
        self.stiefel_matrices = nn.ParameterDict()
        self.epoch_anchors = {} 
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

    def on_train_epoch_start(self) -> None:
        """Initialize/Reset the anchor lists at the start of each epoch."""
        for idx_str in self.agents.keys():
            self.epoch_anchors[int(idx_str)] = []
    
    def parseval_normalize(self, A: torch.Tensor) -> torch.Tensor:
        """Apply Parseval normalization to the anchors latent representation as suggested"""
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
            
        for edge_key, V_param in self.stiefel_matrices.items():
            node_i, node_j = map(int, edge_key.split('_'))

            if node_i not in A_dict or node_j not in A_dict:
                continue

            C = torch.matmul(A_dict[node_i].T, A_dict[node_j])
            U, S, W_T = torch.linalg.svd(C, full_matrices=False)
            V_new = torch.matmul(U, W_T) 

            ## closed form update for the frozen Stiefel matrix
            V_param.copy_(V_new.to(V_param.device))

        for idx in self.epoch_anchors.keys():
            self.epoch_anchors[idx].clear()

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