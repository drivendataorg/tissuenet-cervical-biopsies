import functools as f
import multiprocessing
from pathlib import Path

import boto3
import pandas as pd
import tqdm
from awscli.customizations.s3.utils import split_s3_bucket_key


def download(row, root):
    _, item = row

    s3 = boto3.client("s3")

    url = item.us_tif_url
    bucket_name, key_name = split_s3_bucket_key(url)
    fname = key_name.split("/")[-1]
    path = root / fname
    if path.is_file():
        return

    s3.download_file(bucket_name, key_name, path.as_posix())


def main():
    root = Path("./data")
    metadata = pd.read_csv(root / "train_metadata_eRORy1H.csv")
    data_path = root / "train"
    data_path.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(40) as p:
        with tqdm.tqdm(
            p.imap_unordered(
                func=f.partial(download, root=data_path), iterable=metadata.iterrows()
            ),
            total=len(metadata),
        ) as pbar:
            _ = list(pbar)

    return 0


if __name__ == "__main__":
    exit(main())
