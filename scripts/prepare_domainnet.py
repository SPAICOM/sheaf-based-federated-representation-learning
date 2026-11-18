"""Cache a class subset of DomainNet as small square RGB images, loadable by the datamodules.

DomainNet (Peng et al., 2019): 586k images, 345 classes, 6 visual domains
(clipart, infograph, painting, quickdraw, real, sketch), from the Hugging Face
copy ``wltjr1007/DomainNet`` (columns image, label, domain, image_path; 18.5 GB).
Images in different domains are different instances: there is no cross-domain
pairing.

The script keeps the ``--top-k`` classes whose SMALLEST count over ``--rank-domains``
is largest (or the classes given with ``--classes``), merges the official
train/test splits (the datamodule draws its own held-out rows), resizes the
shorter side to ``--size`` and center-crops a square, and writes one parquet file

    data/domainnet/<out>/train-00000-of-00001.parquet
        image   (Image, PNG)      label  (ClassLabel, the kept classes in rank order, 0..K-1)
        domain  (ClassLabel, 6)   orig_label (int, DomainNet's 345-class id)

with every domain kept, so the experiment config picks the agents' domains.
``--counts-only`` prints the per-domain class counts of the top classes and stops.

The paper's cache, dn64_k20_d6 (20 classes ranked over all six domains, 64 px; the
experiments use the top 10), is the default:

    uv run python scripts/prepare_domainnet.py               # -> data/domainnet/dn64_k20_d6
    uv run python scripts/prepare_domainnet.py --counts-only --top-k 40
"""

import argparse
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPO = 'wltjr1007/DomainNet'
DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']


def parquet_files() -> tuple[list[Path], list[str]]:
    """Local parquet shards (downloaded on first use) and the 345 class names."""
    root = Path(
        snapshot_download(
            REPO, repo_type='dataset', allow_patterns=['data/*', 'README.md']
        )
    )
    files = sorted((root / 'data').glob('*.parquet'))
    meta = pq.read_schema(files[0]).metadata[b'huggingface']
    names = json.loads(meta)['info']['features']['label']['names']
    return files, names


def count(files: list[Path]) -> np.ndarray:
    """counts[domain, label] over every shard (both official splits)."""
    out = np.zeros((len(DOMAINS), 345), dtype=np.int64)
    for f in files:
        t = pq.read_table(f, columns=['label', 'domain'])
        np.add.at(out, (t['domain'].to_numpy(), t['label'].to_numpy()), 1)
    return out


def square(png: bytes, size: int) -> bytes:
    """Shorter side to ``size`` (bicubic, antialiased), center crop, RGB PNG."""
    img = Image.open(io.BytesIO(png)).convert('RGB')
    w, h = img.size
    s = size / min(w, h)
    img = img.resize(
        (max(size, round(w * s)), max(size, round(h * s))),
        Image.BICUBIC,
        reducing_gap=3.0,
    )
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    img = img.crop((left, top, left + size, top + size))
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=False)
    return buf.getvalue()


def _shard(args) -> dict[str, list]:
    """Rows of one row group whose label is kept, images resized."""
    path, rg, keep, size = args
    t = pq.ParquetFile(path).read_row_group(
        rg, columns=['image', 'label', 'domain']
    )
    labels = t['label'].to_numpy()
    mask = np.isin(labels, keep)
    if not mask.any():
        return {'image': [], 'orig_label': [], 'domain': []}
    idx = np.flatnonzero(mask)
    images = t['image'].combine_chunks().field('bytes')
    domains = t['domain'].to_numpy()
    return {
        'image': [square(images[int(k)].as_py(), size) for k in idx],
        'orig_label': labels[idx].tolist(),
        'domain': domains[idx].tolist(),
    }


def main() -> None:
    """Rank the classes, then cache the kept ones (or print the counts)."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--rank-domains',
        default=','.join(DOMAINS),
        help='domains whose minimum class count ranks the classes (default: all six)',
    )
    p.add_argument('--top-k', type=int, default=20)
    p.add_argument(
        '--classes',
        default=None,
        help='comma-separated class names (overrides --top-k)',
    )
    p.add_argument('--size', type=int, default=64)
    p.add_argument(
        '--out',
        default=None,
        help='folder under data/domainnet/ (default dn<size>_k<K>_d<number of rank domains>)',
    )
    p.add_argument('--counts-only', action='store_true')
    p.add_argument('--workers', type=int, default=10)
    a = p.parse_args()

    files, names = parquet_files()
    counts = count(files)
    rank = [DOMAINS.index(d) for d in a.rank_domains.split(',')]
    if a.classes:
        keep = [names.index(c) for c in a.classes.split(',')]
    else:
        floor = counts[rank].min(axis=0)
        keep = sorted(
            np.argsort(-floor, kind='stable')[: a.top_k].tolist(),
            key=lambda c: -floor[c],
        )

    header = (
        f'{"class":<18}'
        + ''.join(f'{d:>10}' for d in DOMAINS)
        + f'{"min(rank)":>11}'
    )
    print(header)
    for c in keep:
        row = ''.join(f'{counts[d, c]:>10}' for d in range(len(DOMAINS)))
        print(f'{names[c]:<18}{row}{counts[rank, c].min():>11}')
    tot = counts[:, keep].sum(axis=1)
    print(f'{"TOTAL":<18}' + ''.join(f'{n:>10}' for n in tot))
    if a.counts_only:
        return

    # new label = position in the ranking (best covered = 0), so the top-k of a
    # cached subset is labels 0..k-1 (dataset.label_subset)
    out = (
        ROOT
        / 'data'
        / 'domainnet'
        / (a.out or f'dn{a.size}_k{len(keep)}_d{len(rank)}')
    )
    out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (str(f), rg, np.array(keep), a.size)
        for f in files
        for rg in range(pq.ParquetFile(f).num_row_groups)
    ]
    cols: dict[str, list] = {'image': [], 'orig_label': [], 'domain': []}
    with ProcessPoolExecutor(a.workers) as ex:
        for k, part in enumerate(ex.map(_shard, jobs, chunksize=1)):
            for key in cols:
                cols[key].extend(part[key])
            if (k + 1) % 50 == 0 or k + 1 == len(jobs):
                print(
                    f'  row groups {k + 1}/{len(jobs)}, kept {len(cols["image"])}',
                    flush=True,
                )

    from datasets import ClassLabel, Dataset, Features, Value
    from datasets import Image as HFImage

    remap = {c: i for i, c in enumerate(keep)}
    features = Features(
        {
            'image': HFImage(),
            'label': ClassLabel(names=[names[c] for c in keep]),
            'domain': ClassLabel(names=DOMAINS),
            'orig_label': Value('int32'),
        }
    )
    ds = Dataset.from_dict(
        {
            'image': [{'bytes': b, 'path': None} for b in cols['image']],
            'label': [remap[c] for c in cols['orig_label']],
            'domain': cols['domain'],
            'orig_label': cols['orig_label'],
        },
        features=features,
    )
    target = out / 'train-00000-of-00001.parquet'
    ds.to_parquet(str(target))
    summary = {
        'source': REPO,
        'size': a.size,
        'classes': [names[c] for c in keep],
        'orig_ids': keep,
        'rows': len(ds),
        'rows_per_domain': {
            d: int(n)
            for d, n in zip(DOMAINS, np.bincount(cols['domain'], minlength=6))
        },
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(
        f'wrote {len(ds)} rows to {target} ({target.stat().st_size / 1e6:.0f} MB)'
    )


if __name__ == '__main__':
    main()
