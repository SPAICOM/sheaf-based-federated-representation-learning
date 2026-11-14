"""Client-first classification datamodule for heterogeneous non-IID splits.

This module mirrors :mod:`classification_datamodule` but changes the split
construction for non-IID settings:

1. Remove any shared pilot set from the full dataset.
2. Partition the remaining samples across clients exactly once.
3. Split each client's local pool into train/val/test with a stratified split.

That ordering preserves each client's local label and quantity skew across
its held-out splits, which the global-first ``ClassificationDataModule`` does
not guarantee.
"""

from __future__ import annotations

import torch
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from torchvision import transforms

from src.datamodules.classification_datamodule import (
    ClassificationDataModule,
    ClassificationDataset,
)
from src.datamodules.utils import compute_split_indices
from src.utils.data_partitioner import (
    _group_class_distribution,
    partition_grouped_non_iid,
    partition_non_iid,
    partition_non_iid_fair,
    partition_non_iid_with_margin,
    sample_shifted_subsets,
)


class HeteroClassificationDataModule(ClassificationDataModule):
    """Classification datamodule with client-first non-IID splitting.

    The public interface intentionally matches ``ClassificationDataModule`` so
    existing experiment wiring can switch targets with no code changes.
    """

    _CLIENT_FIRST_SPLITS = {
        'non_iid',
        'non_iid_with_margin',
        'non_iid_fair',
        'grouped_non_iid',
    }

    #: Split-first, overlapping-draw strategies (see ``_setup_overlapping_shift``).
    _OVERLAPPING_SPLITS = {'overlapping_shift'}

    def __init__(
        self,
        *args,
        safety_margin: int = 10,
        groups: dict | None = None,
        shift_strength: float = 0.0,
        agent_train_samples: int | str | None = None,
        agent_train_frac: float | None = None,
        agent_test_samples: int | None = None,
        agent_test_frac: float | None = None,
        pilot_exclusive: bool = False,
        pilot_class_filter: bool = True,
        test_source: str = 'pool',
        pilot_source: str = 'pool',
        agent_label_noise: dict | None = None,
        agent_label_map: dict | None = None,
        label_subset: list[int] | None = None,
        writer_key: str | None = None,
        pilot_writer_frac: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Keep only the rows whose label is in this list (e.g. the 10 digits of
        # FEMNIST's 62 characters); labels keep their original values.
        self.label_subset = None if label_subset is None else sorted(int(c) for c in label_subset)
        # Column holding the writer (style) of each row. When set,
        # overlapping_shift gives every agent a DISJOINT set of writers, so no
        # two agents ever see the same handwriting style (see
        # _writer_disjoint_draws); pilot_writer_frac of the writers are kept
        # apart for the shared pilots.
        self.writer_key = writer_key
        if not 0.0 < float(pilot_writer_frac) < 1.0:
            raise ValueError(f'pilot_writer_frac must be in (0, 1), got {pilot_writer_frac}')
        self.pilot_writer_frac = float(pilot_writer_frac)
        # {agent: {classes: [a, b], fraction: p}}: in that agent's TRAINING rows
        # (overlapping_shift only), round(p * n) of the n rows labelled a or b
        # get the other label (a <-> b). With direction: one_way, round(p * n_a)
        # of the rows labelled a get label b and b is untouched (a -> b).
        # Val, test and pilots keep true labels.
        self.agent_label_noise = {}
        for k, v in (agent_label_noise or {}).items():
            classes = [int(c) for c in v['classes']]
            frac = float(v['fraction'])
            direction = v.get('direction', 'swap')
            if len(classes) != 2 or not 0.0 <= frac <= 1.0 or direction not in ('swap', 'one_way'):
                raise ValueError(f'agent_label_noise[{k}] needs 2 classes, fraction in [0, 1] '
                                 f"and direction 'swap' | 'one_way', got {v}")
            self.agent_label_noise[int(k)] = (classes, frac, direction)
        # {agent: [t_0, ..., t_{C-1}]}: that agent's TRAINING rows (overlapping_shift
        # only) get label t_y instead of y, e.g. a coarse task (animate vs inanimate).
        # The model keeps C outputs; only the targets used are ever trained.
        # Val, test and pilots keep the original labels (score coarse agents offline).
        self.agent_label_map = {int(k): [int(t) for t in v] for k, v in (agent_label_map or {}).items()}
        self.safety_margin = int(safety_margin)
        self.shift_strength = float(shift_strength)
        # Agent subset sizes for split_strategy='overlapping_shift'. Absolute
        # counts win over fractions; leaving the test size unset mirrors the
        # global train/test ratio onto each agent.
        self.agent_train_samples = agent_train_samples
        self.agent_train_frac = agent_train_frac
        self.agent_test_samples = agent_test_samples
        self.agent_test_frac = agent_test_frac
        # Pilot rows are drawn from the TRAIN pool and, by default, remain
        # eligible for the agent draws — contamination is then exactly
        # n/|TRAIN| and grows linearly with the agent size. Set True to remove
        # them from the agent pool, which zeroes contamination at any n and
        # only costs a (1 - pilot_split) factor in the feasibility bound.
        self.pilot_exclusive = bool(pilot_exclusive)
        # False: every agent gets every pilot, whatever its class — an unlabeled
        # shared set the agents also encode on classes they never trained on.
        self.pilot_class_filter = bool(pilot_class_filter)
        # Where overlapping_shift takes its test rows from:
        #   pool     - carve test_split out of the merged corpus, then let each
        #              agent draw a shifted test subset from it (default)
        #   original - the dataset's own 'test' split, given IN FULL and
        #              unshifted to every agent; train rows and pilots come
        #              only from the original 'train' split. test_split and
        #              agent_test_* are ignored.
        if test_source not in ('pool', 'original'):
            raise ValueError(
                f"test_source must be 'pool' or 'original', got {test_source!r}"
            )
        self.test_source = test_source
        # Where overlapping_shift takes its pilots from:
        #   pool         - pilot_num_samples rows uniform over the TRAIN pool,
        #                  drawn before the agent draws (default)
        #   intersection - the rows held by EVERY agent's training split (the
        #                  data they already share); pilot_num_samples rows
        #                  are subsampled from it (all of it when null or
        #                  larger). The pilots stay in the training sets, so
        #                  the agents' training data do not depend on the
        #                  pilot budget; incompatible with pilot_exclusive.
        if pilot_source not in ('pool', 'intersection'):
            raise ValueError(
                f"pilot_source must be 'pool' or 'intersection', got {pilot_source!r}"
            )
        if pilot_source == 'intersection' and self.pilot_exclusive:
            raise ValueError("pilot_source='intersection' requires pilot_exclusive: false")
        self.pilot_source = pilot_source
        # True when every agent is scored on the very same test samples (with
        # matching sample_ids), so cross-agent instance metrics (misalignment
        # loss, MRR) can be measured on held-out test rows instead of on the
        # pilots the alignment maps are fitted on.
        self.shared_test_set = False

        # groups accepts two formats:
        #   {0: [agents], 1: [agents], ...}
        #   {0: {agents: [...], target_classes: [...]}, ...}
        self.groups: dict[int, list[int]] = {}
        self.group_target_classes: dict[int, list[int]] = {}
        for k, v in (groups or {}).items():
            gid = int(k)
            if isinstance(v, (list, tuple)):
                self.groups[gid] = list(v)
            else:
                self.groups[gid] = list(v['agents'])
                if 'target_classes' in v:
                    self.group_target_classes[gid] = list(v['target_classes'])

    def _partition_client_indices(
        self,
        labels: list[int],
    ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Partition the full dataset once to assign local client pools."""
        if self.split_strategy == 'non_iid':
            return partition_non_iid(
                labels=labels,
                n_agents=self.n_agents,
                classes_per_agent=self.classes_per_agent,
                seed=self.seed,
                alpha=self.alpha,
                return_agent_classes=True,
            )

        if self.split_strategy == 'non_iid_with_margin':
            return partition_non_iid_with_margin(
                labels=labels,
                n_agents=self.n_agents,
                classes_per_agent=self.classes_per_agent,
                seed=self.seed,
                alpha=self.alpha,
                return_agent_classes=True,
                safety_margin=self.safety_margin,
            )

        if self.split_strategy == 'non_iid_fair':
            return partition_non_iid_fair(
                labels=labels,
                n_agents=self.n_agents,
                classes_per_agent=self.classes_per_agent,
                seed=self.seed,
                alpha=self.alpha,
                return_agent_classes=True,
                safety_margin=self.safety_margin,
            )

        if self.split_strategy == 'grouped_non_iid':
            if not self.groups:
                raise ValueError(
                    'split_strategy="grouped_non_iid" requires a non-empty '
                    '"groups" mapping in the dataset config.'
                )
            return partition_grouped_non_iid(
                labels=labels,
                n_agents=self.n_agents,
                groups=self.groups,
                group_target_classes=self.group_target_classes or None,
                classes_per_agent=self.classes_per_agent,
                shift_strength=self.shift_strength,
                seed=self.seed,
                return_agent_classes=True,
            )

        raise ValueError(
            f'Unsupported hetero split_strategy: {self.split_strategy}'
        )

    def _filter_pilots_to_agent_classes(
        self,
        pilot_data,
        pilot_indices: list[int],
        label_key: str,
    ) -> None:
        """Replace shared pilot datasets with per-agent views filtered to trained classes.

        Called after train_datasets are built so we know each agent's actual
        classes. Pilot samples from classes an agent never trained on are dropped,
        preventing the anchor builder from receiving unseen-class embeddings.
        Falls back to keeping at least one pilot sample if an agent trained on
        no classes present in the global pilot pool.

        Also builds ``self._intersection_pilot_dataset``: a pilot dataset
        restricted to classes seen by *every* agent.  This is used by
        ``shared_global_pilots`` mode so that no agent is asked to encode
        out-of-distribution samples, which would corrupt the alignment map.
        """
        pilot_labels = pilot_data[label_key]
        all_seen: list[set] = []
        for i in range(self.n_agents):
            rows = self.train_datasets[i].dataset
            # original classes, not the remapped/noisy training targets
            col = f'{label_key}_clean' if f'{label_key}_clean' in rows.column_names else label_key
            seen_classes = set(rows[col])
            all_seen.append(seen_classes)
            if not self.pilot_class_filter:
                seen_classes = set(pilot_labels)
            keep = [
                idx
                for idx, lbl in enumerate(pilot_labels)
                if lbl in seen_classes
            ]
            if not keep:
                keep = [0]
            filtered_pilot = pilot_data.select(keep)
            filtered_ids = [pilot_indices[idx] for idx in keep]
            self.pilot_datasets[i] = ClassificationDataset(
                filtered_pilot,
                self.data_key,
                label_key,
                sample_ids=filtered_ids,
                noise=self.agent_noise.get(i),
            )

        # Intersection pilot: only classes every agent has been trained on.
        shared_classes = set.intersection(*all_seen) if len(all_seen) > 1 else (all_seen[0] if all_seen else set())
        if not self.pilot_class_filter:
            shared_classes = set(pilot_labels)
        keep_shared = [idx for idx, lbl in enumerate(pilot_labels) if lbl in shared_classes]
        if not keep_shared:
            keep_shared = [0]
        self._intersection_pilot_dataset: ClassificationDataset = ClassificationDataset(
            pilot_data.select(keep_shared),
            self.data_key,
            label_key,
            sample_ids=[pilot_indices[idx] for idx in keep_shared],
        )

    def _local_stratified_split_indices(
        self,
        labels: list[int],
        *,
        seed: int,
        train_class_minimum: int = 0,
    ) -> dict[str, list[int]]:
        """Split one client's local pool while preserving class skew.

        The split is stratified by label so each held-out split follows the
        client's local class distribution as closely as integer counts allow.
        """
        if not labels:
            return {'train': [], 'val': [], 'test': []}

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)

        train_indices: list[int] = []
        val_indices: list[int] = []
        test_indices: list[int] = []

        for class_label in torch.unique(labels_tensor, sorted=True).tolist():
            class_indices = torch.where(labels_tensor == int(class_label))[0]
            shuffled_indices = class_indices[
                torch.randperm(len(class_indices), generator=generator)
            ].tolist()
            class_size = len(shuffled_indices)

            val_count = int(round(class_size * self.val_split))
            test_count = int(round(class_size * self.test_split))

            min_train = min(
                class_size,
                max(int(train_class_minimum), 1),
            )
            max_held_out = max(class_size - min_train, 0)

            while val_count + test_count > max_held_out:
                if test_count >= val_count and test_count > 0:
                    test_count -= 1
                elif val_count > 0:
                    val_count -= 1
                else:
                    break

            train_count = class_size - val_count - test_count
            if train_count <= 0 and class_size > 0:
                if test_count >= val_count and test_count > 0:
                    test_count -= 1
                elif val_count > 0:
                    val_count -= 1
                train_count = class_size - val_count - test_count

            train_indices.extend(shuffled_indices[:train_count])
            val_indices.extend(
                shuffled_indices[train_count : train_count + val_count]
            )
            test_indices.extend(shuffled_indices[train_count + val_count :])

        return {
            'train': sorted(train_indices),
            'val': sorted(val_indices),
            'test': sorted(test_indices),
        }

    def _stratified_holdout_indices(
        self,
        labels: list[int],
        fraction: float,
        *,
        seed: int,
    ) -> tuple[list[int], list[int]]:
        """Split one agent's rows into (keep, held-out), stratified by label."""
        if not labels or fraction <= 0.0:
            return list(range(len(labels))), []

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        generator = torch.Generator().manual_seed(seed)
        keep: list[int] = []
        held: list[int] = []
        for class_label in torch.unique(labels_tensor, sorted=True).tolist():
            class_indices = torch.where(labels_tensor == int(class_label))[0]
            shuffled = class_indices[
                torch.randperm(len(class_indices), generator=generator)
            ].tolist()
            held_count = min(
                int(round(len(shuffled) * fraction)), max(len(shuffled) - 1, 0)
            )
            held.extend(shuffled[:held_count])
            keep.extend(shuffled[held_count:])
        return sorted(keep), sorted(held)

    def _agent_subset_sizes(self, train_pool_size: int) -> tuple[int, int]:
        """Resolve the per-agent train/test draw sizes for ``overlapping_shift``.

        ``agent_train_samples='max'`` resolves to the largest draw every agent
        can take at the current shift (see :meth:`_max_feasible_train_size`),
        so the training set shrinks naturally as ``shift_strength`` grows.

        Absolute counts take precedence over fractions.  With neither set the
        train draw defaults to ``train_pool_size / n_agents`` — the volume a
        disjoint partition would have produced — so switching strategies does
        not silently change how much data each agent holds.  The test draw
        defaults to mirroring the global train/test ratio onto the agent.
        """
        if self.agent_train_samples == 'max':
            n_train = self._max_train_size
        elif self.agent_train_samples is not None:
            n_train = int(self.agent_train_samples)
        elif self.agent_train_frac is not None:
            n_train = int(round(train_pool_size * float(self.agent_train_frac)))
        else:
            n_train = train_pool_size // int(self.n_agents)

        if self.agent_test_samples is not None:
            n_test = int(self.agent_test_samples)
        elif self.agent_test_frac is not None:
            n_test = int(round(n_train * float(self.agent_test_frac)))
        elif self.test_split > 0.0:
            n_test = int(
                round(n_train * self.test_split / (1.0 - self.test_split))
            )
        else:
            n_test = 0
        return max(n_train, 1), max(n_test, 0)

    def _max_feasible_train_size(
        self, labels: list[int], pool: list[int]
    ) -> int:
        """Largest per-agent draw that every agent can take from ``pool``.

        Agent i needs ~n·P_i(k) rows of class k, and largest-remainder rounding
        adds at most one, so n is feasible iff n·P_i(k) + 1 <= N_k for every
        class k and agent i.
        """
        pool_labels = torch.tensor(labels, dtype=torch.long)[
            torch.tensor(pool, dtype=torch.long)
        ]
        classes = sorted(torch.unique(pool_labels).tolist())
        counts = torch.tensor(
            [(pool_labels == c).sum().item() for c in classes],
            dtype=torch.float64,
        )
        bound = float('inf')
        for agent_list_gid, agent_list in self.groups.items():
            dist = _group_class_distribution(
                classes,
                self.group_target_classes.get(agent_list_gid),
                self.shift_strength,
            ).double()
            positive = dist > 0
            bound = min(
                bound,
                float(((counts[positive] - 1) / dist[positive]).min()),
            )
        return max(int(bound), 1)

    def _writer_disjoint_draws(
        self, all_data, labels: list[int]
    ) -> tuple[dict[int, list[int]], list[int], list[int], dict[int, list[int]]]:
        """Train draws, shared test rows and pilots with writer-disjoint agents.

        ::

            writers ── shuffle ─┬▶ pilot_writer_frac ──▶ pilots (uniform rows)
                                └▶ n_agents disjoint writer groups
                                     agent i rows ─┬▶ test_split (stratified) ─▶ shared TEST
                                                   └▶ TRAIN pool_i ─▶ shifted draw ~ P_i

        Every agent draws n_train rows from its OWN writers only, with the usual
        label shift P_i. The test set is the union of the agents' held-out rows:
        the same rows, all classes and every agent's styles, for every agent.
        The pilots come from writers no agent trains on.
        """
        gen = torch.Generator().manual_seed(self.seed)
        writer_of = all_data[self.writer_key][:]
        writers = sorted(set(writer_of))
        order = [writers[k] for k in torch.randperm(len(writers), generator=gen).tolist()]
        n_pilot_writers = max(1, int(round(len(writers) * self.pilot_writer_frac)))
        if len(writers) - n_pilot_writers < self.n_agents:
            raise ValueError(f'{len(writers)} writers cannot give {self.n_agents} agents one each')
        group_of = dict.fromkeys(order[:n_pilot_writers], -1)
        for k, w in enumerate(order[n_pilot_writers:]):
            group_of[w] = k % self.n_agents

        rows: dict[int, list[int]] = {g: [] for g in range(-1, self.n_agents)}
        for idx, w in enumerate(writer_of):
            rows[group_of[w]].append(idx)

        pilot_pool = rows[-1]
        n_pilots = min(int(self.pilot_num_samples or 0), len(pilot_pool))
        picks = torch.randperm(len(pilot_pool), generator=gen)[:n_pilots].tolist()
        pilot_indices = sorted(pilot_pool[p] for p in picks)

        test_rows: list[int] = []
        pools: dict[int, list[int]] = {}
        for i in range(self.n_agents):
            own = rows[i]
            keep, held = self._stratified_holdout_indices(
                [labels[r] for r in own], self.test_split, seed=self.seed + 31 * (i + 1)
            )
            pools[i] = [own[k] for k in keep]
            test_rows.extend(own[k] for k in held)

        group_of_agent = {a: gid for gid, agents in self.groups.items() for a in agents}
        if self.agent_train_samples == 'max':
            self._max_train_size = min(
                self._max_feasible_train_size(labels, pools[i]) for i in range(self.n_agents)
            )
        n_train, _ = self._agent_subset_sizes(min(len(p) for p in pools.values()))
        train_draws: dict[int, list[int]] = {}
        agent_classes: dict[int, list[int]] = {}
        for i in range(self.n_agents):
            gid = group_of_agent[i]
            target = self.group_target_classes.get(gid)
            draws, classes = sample_shifted_subsets(
                labels=labels,
                n_agents=1,
                groups={0: [0]},
                group_target_classes={0: target} if target else None,
                num_samples=n_train,
                shift_strength=self.shift_strength,
                pool_indices=pools[i],
                seed=self.seed + i,
                return_agent_classes=True,
            )
            train_draws[i], agent_classes[i] = draws[0], classes[0]
        self.writer_counts = {i: sum(1 for g in group_of.values() if g == i) for i in range(-1, self.n_agents)}
        return train_draws, sorted(test_rows), pilot_indices, agent_classes

    def _swap_labels(self, rows, label_key: str, client_idx: int):
        """Swap the labels of a random fraction of the rows of two classes
        (one_way: relabel a random fraction of the class-a rows as b).
        """
        (a, b), frac, direction = self.agent_label_noise[client_idx]
        labels = [int(y) for y in rows[label_key]]
        src = (a,) if direction == 'one_way' else (a, b)
        pair = [k for k, y in enumerate(labels) if y in src]
        gen = torch.Generator().manual_seed(self.seed + 104729 + client_idx)
        n_flip = round(frac * len(pair))
        flip = [pair[k] for k in torch.randperm(len(pair), generator=gen)[:n_flip].tolist()]
        for k in flip:
            labels[k] = b if labels[k] == a else a
        self.label_noise_stats = getattr(self, 'label_noise_stats', {})
        self.label_noise_stats[client_idx] = {
            'classes': (a, b), 'direction': direction, 'rows': len(pair), 'flipped': n_flip,
            f'{a}->{b}': sum(1 for k in flip if labels[k] == b),
            f'{b}->{a}': sum(1 for k in flip if labels[k] == a),
        }
        print(f'[label noise] agent {client_idx}: {self.label_noise_stats[client_idx]}')
        feature = rows.features[label_key]
        # keep the true label next to the noisy one (diagnostics only; unused in training)
        rows = rows.add_column(f'{label_key}_clean', [int(y) for y in rows[label_key]], feature=feature)
        return rows.remove_columns(label_key).add_column(label_key, labels, feature=feature)

    def _map_labels(self, rows, label_key: str, client_idx: int):
        """Replace each label y by agent_label_map[client_idx][y]."""
        target = self.agent_label_map[client_idx]
        clean = [int(y) for y in rows[label_key]]
        labels = [target[y] for y in clean]
        counts = {t: labels.count(t) for t in sorted(set(labels))}
        print(f'[label map] agent {client_idx}: {target} -> train label counts {counts}')
        feature = rows.features[label_key]
        if f'{label_key}_clean' not in rows.column_names:
            rows = rows.add_column(f'{label_key}_clean', clean, feature=feature)
        return rows.remove_columns(label_key).add_column(label_key, labels, feature=feature)

    def _pool_draws(
        self, ds, split_names: list[str], labels: list[int], total_size: int
    ) -> tuple[dict[int, list[int]], dict[int, list[int]], list[int], dict[int, list[int]]]:
        """Draw the test pool, the pilots and the shifted train rows (steps 1-3)."""
        # ── 1. Global train/test split ───────────────────────────────────────
        generator = torch.Generator().manual_seed(self.seed)
        if self.test_source == 'original':
            if 'train' not in split_names or 'test' not in split_names:
                raise ValueError(
                    "test_source='original' needs 'train' and 'test' splits; "
                    f'{self.repo}/{self.name} has {split_names}'
                )
            offsets, start = {}, 0
            for name in split_names:
                offsets[name] = (start, start + len(ds[name]))
                start += len(ds[name])
            train_pool = list(range(*offsets['train']))
            test_pool = list(range(*offsets['test']))
        else:
            permutation = torch.randperm(
                total_size, generator=generator
            ).tolist()
            test_count = int(round(total_size * self.test_split))
            test_pool = sorted(permutation[:test_count])
            train_pool = sorted(permutation[test_count:])
        if not train_pool:
            raise ValueError('test_split leaves no training samples')

        # ── 2. Pilots: uniform over the TRAIN pool ───────────────────────────
        if self.pilot_num_samples is not None:
            pilot_count = min(int(self.pilot_num_samples), len(train_pool))
        else:
            pilot_count = int(round(len(train_pool) * self.pilot_split))
        pilot_indices: list[int] = []
        if self.pilot_source == 'intersection':
            pilot_count = 0  # drawn after the agent draws (step 4b)
        if pilot_count > 0:
            picks = torch.randperm(len(train_pool), generator=generator)[
                :pilot_count
            ].tolist()
            pilot_indices = sorted(train_pool[p] for p in picks)
            # A pilot batch wider than the pool itself is degenerate: the
            # loader repeats the pool to fill it, so one batch carries the same
            # sample several times — the penalty then double-counts those rows,
            # and the duplicate sample ids force the edge matcher off its
            # vectorised path onto an O(n^2) Python loop. Clamp instead, which
            # is a no-op whenever the pool is at least one batch wide. Matters
            # when sweeping small pilot budgets against a fixed batch size.
            if self.pilot_batch_size > pilot_count:
                self.pilot_batch_size = pilot_count

        # ── 3. Per-agent overlapping draws from each pool ────────────────────
        agent_pool = train_pool
        if self.pilot_exclusive and pilot_indices:
            excluded = set(pilot_indices)
            agent_pool = [i for i in train_pool if i not in excluded]

        if self.agent_train_samples == 'max':
            self._max_train_size = self._max_feasible_train_size(
                labels, agent_pool
            )
        n_train, n_test = self._agent_subset_sizes(len(agent_pool))
        train_draws, agent_classes = sample_shifted_subsets(
            labels=labels,
            n_agents=self.n_agents,
            groups=self.groups,
            group_target_classes=self.group_target_classes or None,
            num_samples=n_train,
            shift_strength=self.shift_strength,
            pool_indices=agent_pool,
            seed=self.seed,
            return_agent_classes=True,
        )
        if self.test_source == 'original':
            # Every agent is scored on the same, complete, unshifted test set.
            test_draws = {i: list(test_pool) for i in range(self.n_agents)}
            self.shared_test_set = self.n_agents > 1
        elif n_test > 0 and test_pool:
            test_draws = sample_shifted_subsets(
                labels=labels,
                n_agents=self.n_agents,
                groups=self.groups,
                group_target_classes=self.group_target_classes or None,
                num_samples=n_test,
                shift_strength=self.shift_strength,
                pool_indices=test_pool,
                seed=self.seed + 1,
                return_agent_classes=False,
            )
        else:
            test_draws = {i: [] for i in range(self.n_agents)}
        return train_draws, test_draws, pilot_indices, agent_classes

    def _setup_overlapping_shift(self) -> None:
        """Split-first pipeline with overlapping, label-shifted agent draws.

        ::

            D  ──  test_split  ──▶  TEST pool ──▶ agent i draws n_test ~ P_i
               └── remainder   ──▶  TRAIN pool ─┬▶ pilots: pilot_split, uniform
                                                └▶ agent i draws n_train ~ P_i

        with ``P_i(y) = s·Unif{C_i} + (1-s)·Unif{Y}``.  Agents draw
        *independently*, so their subsets overlap each other and may include
        pilot rows; only TRAIN and TEST stay disjoint, which is the invariant
        the communication-accuracy evaluation depends on (maps are fit on
        pilots, scored on test latents).

        Two properties follow from dropping disjointness, neither of which
        the ``grouped_non_iid`` partition can offer:

        * every agent holds exactly the same number of rows at any
          ``shift_strength``, and
        * the realised label marginal equals ``P_i`` exactly, instead of the
          contention-normalised ``P_i(k) / Σ_j P_j(k)`` a partition yields.

        ``pilot_split`` is read as a fraction of the TRAIN pool here (not of
        the whole corpus), so sweeping the pilot budget leaves every agent's
        training rows untouched.
        """
        if not self.groups:
            raise ValueError(
                "split_strategy='overlapping_shift' requires a non-empty "
                '"groups" mapping in the dataset config.'
            )

        ds = load_dataset(f'{self.repo}/{self.name}')
        split_names = list(ds)
        all_data = concatenate_datasets([ds[s] for s in split_names])
        label_key = self.attributes[0]
        labels = all_data[label_key]
        total_size = len(all_data)

        if self.n_agents is None:
            self.n_agents = (
                max(a for agents in self.groups.values() for a in agents) + 1
            )

        self.train_datasets: dict[int, ClassificationDataset] = {}
        self.val_datasets: dict[int, ClassificationDataset] = {}
        self.test_datasets: dict[int, ClassificationDataset] = {}
        self.pilot_datasets: dict[int, ClassificationDataset] = {}
        self.num_classes: dict[int, int] = {}

        if self.label_subset is not None or self.writer_key is not None:
            # plain lists: element-wise indexing of a lazy datasets Column is slow
            labels = all_data[label_key][:]
        if self.label_subset is not None:
            wanted = set(self.label_subset)
            keep = [k for k, y in enumerate(labels) if y in wanted]
            all_data = all_data.select(keep)
            labels = [labels[k] for k in keep]
            total_size = len(all_data)

        if self.writer_key is not None:
            train_draws, test_pool, pilot_indices, agent_classes = (
                self._writer_disjoint_draws(all_data, labels)
            )
            test_draws = {i: list(test_pool) for i in range(self.n_agents)}
            self.shared_test_set = self.n_agents > 1
            if pilot_indices and self.pilot_batch_size > len(pilot_indices):
                self.pilot_batch_size = len(pilot_indices)
        else:
            train_draws, test_draws, pilot_indices, agent_classes = self._pool_draws(
                ds, split_names, labels, total_size
            )
        self.agent_classes = agent_classes

        # ── 4. Materialise the per-agent datasets ────────────────────────────
        kept_train_ids: dict[int, list[int]] = {}
        for client_idx in range(self.n_agents):
            rotation = self.agent_rotations.get(client_idx, 0)
            local_data = all_data.select(train_draws[client_idx])
            keep, held = self._stratified_holdout_indices(
                local_data[label_key],
                self.val_split,
                seed=self.seed + client_idx,
            )
            kept_train_ids[client_idx] = [train_draws[client_idx][k] for k in keep]
            noise = self.agent_noise.get(client_idx)
            train_rows = local_data.select(keep)
            if client_idx in self.agent_label_noise:
                train_rows = self._swap_labels(train_rows, label_key, client_idx)
            if client_idx in self.agent_label_map:
                train_rows = self._map_labels(train_rows, label_key, client_idx)
            # Augmentation (CIFAR-10 crop+flip, None otherwise) on the
            # training rows only; val/test/pilots keep the raw inputs.
            self.train_datasets[client_idx] = ClassificationDataset(
                train_rows,
                self.data_key,
                label_key,
                rotation_angle=rotation,
                transform=self.train_transform,
                noise=noise,
            )
            self.val_datasets[client_idx] = ClassificationDataset(
                local_data.select(held),
                self.data_key,
                label_key,
                rotation_angle=rotation,
                noise=noise,
            )
            self.test_datasets[client_idx] = ClassificationDataset(
                all_data.select(test_draws[client_idx]),
                self.data_key,
                label_key,
                rotation_angle=rotation,
                sample_ids=list(test_draws[client_idx]),
                noise=noise,
            )
            self.num_classes[client_idx] = len(
                set(self.train_datasets[client_idx].dataset[label_key])
            )

        # ── 4b. Pilots from the agents' shared training rows ─────────────────
        self.pilot_intersection_size = None
        if self.pilot_source == 'intersection':
            shared = set(kept_train_ids[0])
            for ids in kept_train_ids.values():
                shared &= set(ids)
            shared = sorted(shared)
            self.pilot_intersection_size = len(shared)
            want = (
                len(shared) if self.pilot_num_samples is None
                else min(int(self.pilot_num_samples), len(shared))
            )
            if self.pilot_num_samples is not None and int(self.pilot_num_samples) > len(shared):
                print(
                    f'[pilots] requested {self.pilot_num_samples} but the training '
                    f'intersection holds {len(shared)}; using all of it'
                )
            pick_gen = torch.Generator().manual_seed(self.seed + 7919)
            picks = torch.randperm(len(shared), generator=pick_gen)[:want].tolist()
            pilot_indices = sorted(shared[p] for p in picks)
            if pilot_indices and self.pilot_batch_size > len(pilot_indices):
                self.pilot_batch_size = len(pilot_indices)

        # ── 5. Shared pilots (same rows, same ids, on every agent) ───────────
        if pilot_indices:
            pilot_data = all_data.select(pilot_indices)
            self._build_pilot_datasets(
                pilot_data=pilot_data,
                pilot_indices=pilot_indices,
                label_key=label_key,
            )
            self._filter_pilots_to_agent_classes(
                pilot_data, pilot_indices, label_key
            )

        self.overlap_stats = self._compute_overlap_stats(
            train_draws, pilot_indices
        )

        sample, _ = self.train_datasets[0][0]
        if isinstance(sample, Image.Image):
            self.input_shape = transforms.ToTensor()(sample).shape
        else:
            self.input_shape = sample.shape

        self.num_classes['label'] = len(set(labels))
        self.models = list(range(self.n_agents))
        self.input_dims = {str(i): self.input_shape[0] for i in self.models}

    def _compute_overlap_stats(
        self,
        train_draws: dict[int, list[int]],
        pilot_indices: list[int],
    ) -> dict[str, float]:
        """Quantify how much agents share, for reporting alongside results.

        Overlap is a deliberate feature of ``overlapping_shift``, not a bug,
        but it is a number a reader will ask for: ``pairwise_*`` is the
        fraction of an agent's training rows also held by another agent, and
        ``pilot_contamination`` the fraction of pilots that sit inside a given
        agent's training set (expected value ``n / |TRAIN|``).
        """
        sets = {i: set(idx) for i, idx in train_draws.items()}
        sizes = [len(s) for s in sets.values() if s]
        if not sizes:
            return {}
        mean_size = sum(sizes) / len(sizes)

        overlaps = [
            len(sets[i] & sets[j])
            for i in sets
            for j in sets
            if i < j and sets[i] and sets[j]
        ]
        pilots = set(pilot_indices)
        contamination = (
            [len(pilots & s) / len(pilots) for s in sets.values() if s]
            if pilots
            else [0.0]
        )
        return {
            'agent_train_size': mean_size,
            'pairwise_overlap_mean': (
                sum(overlaps) / len(overlaps) / mean_size if overlaps else 0.0
            ),
            'pairwise_overlap_max': (
                max(overlaps) / mean_size if overlaps else 0.0
            ),
            'pilot_contamination_mean': sum(contamination)
            / len(contamination),
        }

    def setup(self, stage: str | None = None) -> None:
        """Load the dataset and build client-first non-IID splits."""
        if self.split_strategy in self._OVERLAPPING_SPLITS:
            self._setup_overlapping_shift()
            return
        if self.split_strategy not in self._CLIENT_FIRST_SPLITS:
            super().setup(stage=stage)
            return

        ds = load_dataset(f'{self.repo}/{self.name}')
        splits = [ds[s] for s in ds]
        all_data = concatenate_datasets(splits)

        self.train_datasets: dict[int, ClassificationDataset] = {}
        self.val_datasets: dict[int, ClassificationDataset] = {}
        self.test_datasets: dict[int, ClassificationDataset] = {}
        self.pilot_datasets: dict[int, ClassificationDataset] = {}
        self.num_classes: dict[int, int] = {}

        if self.n_agents is None:
            max_key = 0
            if self.agent_rotations:
                max_key = max(max_key, max(self.agent_rotations.keys()))
            if self.agent_classes:
                max_key = max(max_key, max(self.agent_classes.keys()))
            self.n_agents = max_key + 1

        label_key = self.attributes[0]

        pilot_split_indices = compute_split_indices(
            total_size=len(all_data),
            val_split=0.0,
            test_split=0.0,
            seed=self.seed,
            pilot_split=self.pilot_split,
            pilot_num_samples=self.pilot_num_samples,
        )
        pilot_indices = pilot_split_indices['pilot']
        remaining_data = all_data.select(pilot_split_indices['train'])

        pilot_data = None
        if pilot_indices:
            pilot_data = all_data.select(pilot_indices)
            self._build_pilot_datasets(
                pilot_data=pilot_data,
                pilot_indices=pilot_indices,
                label_key=label_key,
            )

        client_indices, sampled_agent_classes = self._partition_client_indices(
            labels=remaining_data[label_key]
        )
        self.agent_classes = sampled_agent_classes

        train_class_minimum = (
            self.safety_margin
            if (
                self.starve_clients
                and self.split_strategy
                in {'non_iid_with_margin', 'non_iid_fair'}
            )
            else 0
        )

        for client_idx in range(self.n_agents):
            local_indices = client_indices[client_idx]
            local_data = remaining_data.select(local_indices)
            local_split_indices = self._local_stratified_split_indices(
                labels=local_data[label_key],
                seed=self.seed + client_idx,
                train_class_minimum=train_class_minimum,
            )

            rotation = self.agent_rotations.get(client_idx, 0)
            self.train_datasets[client_idx] = ClassificationDataset(
                local_data.select(local_split_indices['train']),
                self.data_key,
                label_key,
                rotation_angle=rotation,
            )
            self.val_datasets[client_idx] = ClassificationDataset(
                local_data.select(local_split_indices['val']),
                self.data_key,
                label_key,
                rotation_angle=rotation,
            )
            self.test_datasets[client_idx] = ClassificationDataset(
                local_data.select(local_split_indices['test']),
                self.data_key,
                label_key,
                rotation_angle=rotation,
            )
            self.num_classes[client_idx] = len(
                set(self.train_datasets[client_idx].dataset[label_key])
            )

        if self.starve_clients:
            self._starve_training_datasets(label_key)

        if (
            self.split_strategy == 'grouped_non_iid'
            and pilot_data is not None
            and pilot_indices
        ):
            self._filter_pilots_to_agent_classes(
                pilot_data, pilot_indices, label_key
            )

        sample, _ = self.train_datasets[0][0]
        if isinstance(sample, Image.Image):
            self.input_shape = transforms.ToTensor()(sample).shape
        else:
            self.input_shape = sample.shape

        self.num_classes['label'] = len(set(all_data[label_key]))
        self.models = list(range(self.n_agents))
        self.input_dims = {str(i): self.input_shape[0] for i in self.models}

    def _create_loaders(self, split: str):
        """Like the parent, but for ``shared_global_pilots`` swap in the
        intersection-only pilot so no agent encodes out-of-distribution classes.
        """
        intersection_ds = getattr(self, '_intersection_pilot_dataset', None)
        if (
            self.comm_data == 'shared_global_pilots'
            and intersection_ds is not None
            and self.pilot_datasets
        ):
            orig = self.pilot_datasets.get(0)
            self.pilot_datasets[0] = intersection_ds
            result = super()._create_loaders(split)
            if orig is not None:
                self.pilot_datasets[0] = orig
            return result
        return super()._create_loaders(split)
