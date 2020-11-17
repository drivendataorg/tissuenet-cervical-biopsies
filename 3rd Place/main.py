import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyvips
import torch

DATA_ROOT = Path(__file__).parent / "data"

logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)s:%(message)s", level=logging.INFO
)


def to_gpu(inp, gpu=0):
    return inp.cuda(gpu, non_blocking=True)


def to_tensor(x):
    x = x.astype("float32") / 255

    return torch.from_numpy(x).permute(2, 0, 1)


def read_img(path, page=4):
    slide = pyvips.Image.new_from_file(str(path), page=page)
    region = pyvips.Region.new(slide).fetch(0, 0, slide.width, slide.height)

    return np.ndarray(
        buffer=region, dtype=np.uint8, shape=(slide.height, slide.width, 3)
    )


def get_tiles(img, tile_size, n_tiles, mode=0):
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


def concat_tiles(tiles, n_tiles, image_size):
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

            h1 = h * image_size
            w1 = w * image_size
            img[h1 : h1 + image_size, w1 : w1 + image_size] = this_img

    return img


class DS(torch.utils.data.Dataset):
    def __init__(self, df, root, n_tiles, tile_size):
        self.df = df
        self.root = root
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        logging.info(
            f"Dataset with #tiles: {self.n_tiles}, tile size: {self.tile_size}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        img = read_img(self.root / item.filename)
        img, _ = get_tiles(
            img,
            tile_size=self.tile_size,
            n_tiles=self.n_tiles,
        )
        img = concat_tiles(
            img,
            n_tiles=self.n_tiles,
            image_size=self.tile_size,
        )

        img = to_tensor(img)

        return img, item.filename

    @staticmethod
    def collate_fn(x):
        x, y = list(zip(*x))

        return torch.stack(x), y


def perform_inference(input_metadata, submission_format):
    all_preds = []
    tta = 0
    batch_size = 8
    logits = torch.zeros((batch_size, 3), dtype=torch.float32, device="cuda")
    for n_tiles, tile_size in [
        (36, 256),
        (64, 192),
        (144, 128),
    ]:
        model_path = Path("assets") / f"{tile_size}"
        logging.info(f"Reading models from {model_path}")
        models = [
            torch.jit.load(str(p)).cuda().eval()
            for p in model_path.rglob("model_best.pt")
        ]
        n_models = len(models)
        n_augs = n_models * (tta + 1)
        logging.info(f"#augs {n_augs}")

        ds = DS(input_metadata, DATA_ROOT, n_tiles=n_tiles, tile_size=tile_size)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=batch_size,
            shuffle=False,
            collate_fn=DS.collate_fn,
            pin_memory=True,
        )

        preds = []
        with torch.no_grad():
            for x, y in loader:
                x = to_gpu(x)
                bs = len(x)

                logits.zero_()
                for model in models:
                    logits[:bs] += model(x).sigmoid()

                logits /= n_augs
                preds.extend(logits.cpu().numpy())

        all_preds.append(preds)

    all_preds = np.array(all_preds).mean(0).sum(-1).round().astype("int")
    for pred, filename in zip(all_preds, input_metadata.filename.values):
        submission_format.loc[filename, str(pred)] = 1

    # save as "submission.csv" in the root folder, where it is expected
    submission_format.to_csv("submission.csv")


if __name__ == "__main__":
    # load metadata
    input_metadata = pd.read_csv(DATA_ROOT / "test_metadata.csv")

    # load sumission format
    submission_format = pd.read_csv(DATA_ROOT / "submission_format.csv", index_col=0)

    perform_inference(input_metadata, submission_format)
