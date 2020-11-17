from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
from sklearn.model_selection import KFold


def to_gpu(inp, gpu=0):
    return inp.cuda(gpu, non_blocking=True)


def split_df(df, args):
    kf = KFold(n_splits=args.train.n_folds, shuffle=True, random_state=args.train.seed)
    df["fold"] = -1
    fold_idx = len(df.columns) - 1

    y = df[args.data.labels]
    for i, (_, dev_index) in enumerate(kf.split(range(len(df)), y.values.argmax(-1))):
        df.iloc[dev_index, fold_idx] = i

    return (
        df[df.fold != args.train.fold].reset_index(drop=True),
        df[df.fold == args.train.fold].reset_index(drop=True),
    )


def get_data_groups(args):
    train = pd.read_csv(args.data.train, nrows=100 if args.general.debug else None)
    root = Path(args.data.root)
    train.filename = train.filename.apply(lambda x: str(root / x))
    args.data.labels = train.columns[1:].tolist()

    train, dev = split_df(train, args)

    if args.train.ft:
        train = pd.concat([train, dev])

    return train, dev


def read_img(path, page=4):
    slide = pyvips.Image.new_from_file(str(path), page=page)
    region = pyvips.Region.new(slide).fetch(0, 0, slide.width, slide.height)

    return np.ndarray(
        buffer=region, dtype=np.uint8, shape=(slide.height, slide.width, 3)
    )


def get_tiles(img, tile_size=256, n_tiles=36, mode=0):
    h, w, c = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255,
    )
    img = img.reshape(
        img.shape[0] // tile_size, tile_size, img.shape[1] // tile_size, tile_size, 3
    )
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)

    n_tiles_with_info = (
        img.reshape(img.shape[0], -1).sum(1) < tile_size ** 2 * 3 * 255
    ).sum()
    if len(img) < n_tiles:
        img = np.pad(
            img, [[0, n_tiles - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255
        )

    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:n_tiles]
    img = img[idxs]

    return img, n_tiles_with_info >= n_tiles


def concat_tiles(tiles, n_tiles, image_size, rand=False, transform=None):
    if rand:
        idxes = np.random.choice(list(range(n_tiles)), n_tiles, replace=False)
    else:
        idxes = list(range(n_tiles))

    n_row_tiles = int(np.sqrt(n_tiles))
    img = np.zeros(
        (image_size * n_row_tiles, image_size * n_row_tiles, 3), dtype="uint8"
    )
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w

            if len(tiles) > idxes[i]:
                this_img = tiles[idxes[i]]
            else:
                this_img = np.ones((image_size, image_size, 3), dtype="uint8") * 255

            if transform is not None:
                this_img = transform(this_img)

            h1 = h * image_size
            w1 = w * image_size
            img[h1 : h1 + image_size, w1 : w1 + image_size] = this_img

    if transform is not None:
        img = transform(img)

    return img
