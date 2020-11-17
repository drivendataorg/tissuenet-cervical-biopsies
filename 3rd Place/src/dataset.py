import albumentations as A
import numpy as np
import torch

from utils import concat_tiles, get_data_groups, get_tiles, read_img

albu_train = A.Compose(
    [
        A.Transpose(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
    ]
)


def to_tensor(x):
    x = x.astype("float32") / 255

    return torch.from_numpy(x).permute(2, 0, 1)


def train_transform(x):
    return albu_train(image=x)["image"]


def dev_transform(x):
    return x


class DNADS(torch.utils.data.Dataset):
    def __init__(self, df, args, is_train):
        self.df = df
        self.config = args
        self.is_train = is_train
        self.transform = train_transform if is_train else dev_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        img = read_img(item.filename, page=self.config.train.page)
        img, _ = get_tiles(
            img,
            tile_size=self.config.train.tile_size,
            n_tiles=self.config.train.n_tiles,
            mode=0,
        )
        img = concat_tiles(
            img,
            n_tiles=self.config.train.n_tiles,
            image_size=self.config.train.image_size,
            rand=self.is_train,
            transform=self.transform,
        )

        img = to_tensor(img)

        label = np.zeros(self.config.network.num_classes, dtype="float32")
        label[: item[self.config.data.labels].values.argmax()] = 1.0

        return img, torch.from_numpy(label)

    @staticmethod
    def collate_fn(x):
        x, y = list(zip(*x))

        return torch.stack(x), torch.stack(y)


def get_loaders(args, test_only=False):
    train_gps, dev_gps = get_data_groups(args)
    batch_size = args.train.batch_size
    num_workers = min(batch_size, args.general.workers)

    dev_ds = DNADS(dev_gps, args, is_train=False)
    dev_sampler = None
    if args.dist.distributed:
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_ds)

    dev_loader = torch.utils.data.DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=dev_sampler,
        num_workers=num_workers,
        collate_fn=DNADS.collate_fn,
        pin_memory=True,
    )

    print(batch_size)

    if test_only:
        return dev_loader

    train_ds = DNADS(train_gps, args, is_train=True)

    train_sampler = None
    if args.dist.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=DNADS.collate_fn,
        pin_memory=True,
    )

    return (train_loader, train_sampler), dev_loader
