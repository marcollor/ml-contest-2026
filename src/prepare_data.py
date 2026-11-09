"""Download the datasets and prepare the fixed training and IID splits."""

from cli import HelpArgumentParser
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
RELEASES = "https://github.com/GDG-UAM/ml-contest-materials/releases"


def download_if_missing(path, url):
    if not path.exists():
        temporary = path.with_suffix(".download")
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(path)


def main(argv=None):
    HelpArgumentParser(description=__doc__).parse_args(argv)
    sys.path.insert(0, str(ROOT))
    out = ROOT / "data"
    out.mkdir(parents=True, exist_ok=True)
    data = out / "train_100k.npz"
    download_if_missing(data, f"{RELEASES}/download/v1.0/train_100k.npz")

    # 3.- Verifies dataset integrity and loads it
    assert data.stat().st_size == 413201540, 'Unexpected dataset size.'
    with data.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    assert digest == '1c231e9a807c348b34290415f56e7c7b99543a81673a9d3cef1002b1a42e5668', 'Unexpected dataset signature.'
    with zipfile.ZipFile(data) as z:
        assert z.testzip() is None

    from competition_api import load_dataset
    ds = load_dataset(data)
    assert ds.series.shape == (100000, 4, 256) and ds.series.dtype == np.float32, 'Unexpected dataset size'
    assert np.isin(ds.labels, np.arange(8)).all(), 'Unexpected dataset classes.'

    # 4.- Makes splits
    seed = 20261112
    tr, iid = train_test_split(np.arange(len(ds)), test_size=.2,
                                stratify=ds.labels, random_state=seed)
    tr, dev = train_test_split(tr, test_size=.0625,
                                stratify=ds.labels[tr], random_state=seed + 1)
    order = np.random.default_rng(seed).permutation(tr)
    parts = dict(train=order[:70000], reserve=order[70000:], dev=dev, iid=iid)
    assert len(np.unique(np.concatenate(list(parts.values())))) == 100000, 'Unexpected partition sizes'

    # 5.- Saves split
    np.savez(out / 'split_indices.npz', **parts)
    np.savez(out / 'val.npz', series=ds.series[iid], labels=ds.labels[iid],
            **{'meta__' + k: v[iid] for k, v in ds.meta.items()})
    print('Verification successful:', {k: len(v) for k, v in parts.items()})
    print('Harness file:', out / 'val.npz')
    download_if_missing(out / "test_demo.npz", f"{RELEASES}/latest/download/test_demo.npz")

if __name__ == '__main__':
    main()

