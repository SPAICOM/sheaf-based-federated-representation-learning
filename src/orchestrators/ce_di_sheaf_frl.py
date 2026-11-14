"""CE-DiSheafFRL: DiSheafFRL that communicates only on a fraction of the epochs.

:class:`CESheafFRL`'s uniform schedule (``comm_percentage`` of the
``trainer.max_epochs`` epochs communicate, spread evenly, the last one always
included) on top of :class:`DiSheafFRL`'s directed, confidence-gated penalty:

    collaborative epoch : exactly a DiSheafFRL epoch — per-step pilot exchange
        and gated penalty, maps refit on the matched pilots at the epoch end.
    local epoch : exactly a non-cooperative epoch — task loss only, and no
        pilot pass at all (the pilots never go through the encoders, so they do
        not touch the BatchNorm / CenterNorm running statistics either); the
        maps keep their last fit.

so 100% is DiSheafFRL and the 0% limit is ``non_cooperative`` (which CESheafFRL
refuses: with no collaborative epoch the maps would never be fit).  Evaluation
is unchanged (post-hoc Procrustes on the pilots under the paper protocol).
``local_reg`` must stay False: its frozen-neighbour regulariser needs the pilot
pass that local epochs here skip.
"""

from __future__ import annotations

from .ce_sheaf_frl import CESheafFRL
from .di_sheaf_frl import DiSheafFRL


class CEDiSheafFRL(CESheafFRL, DiSheafFRL):
    # MRO: CESheafFRL (schedule) -> DiSheafFRL (gated penalty) -> SheafFRL
    def __init__(self, **kwargs):
        if kwargs.get('local_reg'):
            raise ValueError(
                'CEDiSheafFRL skips the pilot pass on local epochs, so local_reg '
                '(a penalty toward the frozen neighbour pilots) is not supported'
            )
        super().__init__(**kwargs)

    def _local_training_step(self) -> bool:
        return self.training and not self._is_collab_epoch(self.current_epoch)

    def _extract_pilot_batch(self, batch, idx):
        # SheafFRL's step treats a ValueError here as "no pilots this step":
        # it skips the pilot encoding, and _compute_alignment_losses (CESheafFRL)
        # returns zero penalty on a local epoch, so the step is task loss only.
        if self._local_training_step():
            raise ValueError('local epoch: no pilot pass')
        return super()._extract_pilot_batch(batch, idx)
