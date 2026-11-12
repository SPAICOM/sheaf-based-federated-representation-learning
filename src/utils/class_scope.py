"""Test accuracy split by whether the agents trained on the sample's class.

Under label shift an agent never trains on some classes, so the headline
accuracies mix samples of very different difficulty. This accumulator splits
them by class scope:

* private accuracy of agent i: classes agent i trained on (``seen``) vs the
  rest (``unseen``);
* communication accuracy of a directed edge sender -> receiver (sender's test
  latents, decoded by the receiver):

    both_seen            both agents trained on the class
    tx_unseen_rx_seen    only the receiver did: the receiver's knowledge
                         reaches a sample the sender cannot classify itself
    tx_seen_rx_unseen    only the sender did (the receiver's decoder never
                         learned the class, so this is ~0 by construction)
    both_unseen          neither did
    all                  every test sample

Correct predictions and sample counts are pooled (micro-averaged) at every
level: per edge, per agent (over its incoming edges ``rx`` or outgoing edges
``tx``) and over the whole network. ``logs`` returns the accuracies (scopes
with no samples are left out) and the network-level sample counts as
``n_*`` keys, under ``<prefix>/scope/``.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping

import torch

COMM_SCOPES = (
    'all',
    'both_seen',
    'tx_unseen_rx_seen',
    'tx_seen_rx_unseen',
    'both_unseen',
)
PRIVATE_SCOPES = ('seen', 'unseen')


class ClassScopeAccuracy:
    """Pool correct/total counts by class scope; see the module docstring."""

    def __init__(self, agent_classes: Mapping[int, Iterable[int]]):
        self.seen = {
            int(i): torch.tensor(sorted(int(c) for c in cls))
            for i, cls in agent_classes.items()
        }
        self.counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    def covers(self, *agents: int) -> bool:
        """Whether the class sets of all these agents are known."""
        return all(int(a) in self.seen for a in agents)

    def _add(
        self, keys: list[str], correct: torch.Tensor, mask: torch.Tensor
    ) -> None:
        n, c = int(mask.sum()), int((correct & mask).sum())
        for key in keys:
            self.counts[key][0] += c
            self.counts[key][1] += n

    def add_private(
        self, agent: int, logits: torch.Tensor, y: torch.Tensor
    ) -> None:
        """One agent's own predictions on the test set."""
        y = y.detach().cpu().long()
        correct = logits.detach().argmax(dim=1).cpu() == y
        seen = torch.isin(y, self.seen[int(agent)])
        for scope, mask in (('seen', seen), ('unseen', ~seen)):
            self._add(
                [f'private_{scope}_agent_{agent}', f'private_{scope}'],
                correct,
                mask,
            )

    def add_comm(
        self, sender: int, receiver: int, logits: torch.Tensor, y: torch.Tensor
    ) -> None:
        """Receiver's predictions on the sender's mapped test latents."""
        y = y.detach().cpu().long()
        correct = logits.detach().argmax(dim=1).cpu() == y
        tx = torch.isin(y, self.seen[int(sender)])
        rx = torch.isin(y, self.seen[int(receiver)])
        masks = {
            'all': torch.ones_like(tx),
            'both_seen': tx & rx,
            'tx_unseen_rx_seen': ~tx & rx,
            'tx_seen_rx_unseen': tx & ~rx,
            'both_unseen': ~tx & ~rx,
        }
        for scope, mask in masks.items():
            keys = [
                f'comm_{scope}_edge_{sender}_{receiver}',
                f'comm_{scope}_rx_{receiver}',
                f'comm_{scope}_tx_{sender}',
                f'comm_{scope}',
            ]
            self._add(keys, correct, mask)

    def logs(self, prefix: str) -> dict[str, float]:
        """Accuracies of every non-empty scope plus network-level counts."""
        out: dict[str, float] = {}
        for key, (correct, total) in self.counts.items():
            if total:
                out[f'{prefix}/scope/{key}'] = correct / total
        network = [f'private_{s}' for s in PRIVATE_SCOPES]
        network += [f'comm_{s}' for s in COMM_SCOPES]
        for key in network:
            if key in self.counts:
                out[f'{prefix}/scope/n_{key}'] = float(self.counts[key][1])
        return out
