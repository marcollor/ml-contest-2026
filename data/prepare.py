"""Verified download, splits and IID for harness."""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def main():
    # 1.- Gets arguments
    p = argparse.ArgumentParser()
    p.add_argument('--repo', default='.')
    p.add_argument('--out', default='model/data')
    a = p.parse_args()

    # 2.- Gets paths and downloads dataset if missing
    repo, out = Path(a.repo).resolve(), Path(a.out).resolve()
    sys.path.insert(0, str(repo))
    data = repo / 'data/train_100k.npz'
    data.parent.mkdir(parents=True, exist_ok=True)
    if not data.exists():
        url = ('https://github.com/GDG-UAM/ml-contest-materials/releases/'
               'download/v1.0/train_100k.npz')
        temp = data.with_suffix('.download')
        urllib.request.urlretrieve(url, temp)
        temp.replace(data)

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
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'split_indices.npz', **parts)
    np.savez(out / 'val.npz', series=ds.series[iid], labels=ds.labels[iid],
            **{'meta__' + k: v[iid] for k, v in ds.meta.items()})
    print('Verification successful:', {k: len(v) for k, v in parts.items()})
    print('Harness file:', out / 'val.npz')

if __name__ == '__main__':
    main()

    