import argparse
import copy
from pathlib import Path

import torch
import tqdm

from dataset import get_loaders
from metric import Accuracy, Score
from model import get_model
from utils import to_gpu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str, required=True, help="Load model")

    return parser.parse_args()


def epoch_step(loader, desc, model, metrics):
    model.eval()

    with tqdm.tqdm(loader, desc=desc, leave=False, mininterval=2) as pbar:
        for x, y in pbar:
            x = to_gpu(x)
            y = to_gpu(y)

            logits = model(x)

            for metric in metrics.values():
                metric.update(logits, y)

            pbar.set_postfix(
                **{k: f"{metric.evaluate():.4}" for k, metric in metrics.items()}
            )


def main():
    global args

    args = parse_args()

    path_to_load = Path(args.load).expanduser()
    if path_to_load.is_file():
        print(f"=> Loading checkpoint '{path_to_load}'")
        checkpoint = torch.load(
            path_to_load, map_location=lambda storage, loc: storage.cuda(0)
        )
        print(f"=> Loaded checkpoint '{path_to_load}'")
    else:
        raise

    args = checkpoint["args"]

    model = get_model(args)

    model.cuda()

    work_dir = path_to_load.parent

    state_dict = copy.deepcopy(checkpoint["state_dict"])
    for p in checkpoint["state_dict"]:
        if p.startswith("module."):
            state_dict[p[len("module.") :]] = state_dict.pop(p)

    model.load_state_dict(state_dict)

    x = torch.rand(2, 3, 256 * 6, 256 * 6).cuda()
    model = model.eval()
    if "efficientnet" in args.network.name:
        model.set_swish(memory_efficient=False)

    with torch.no_grad():
        traced_model = torch.jit.trace(model, x)

    traced_model.save(str(work_dir / f"model_{path_to_load.stem}.pt"))
    del traced_model
    del model

    dev_loader = get_loaders(args, test_only=True)
    metrics = {"score": Score(), "acc": Accuracy()}

    model = (
        torch.jit.load(str(work_dir / f"model_{path_to_load.stem}.pt")).cuda().eval()
    )

    with torch.no_grad():
        for metric in metrics.values():
            metric.clean()

        epoch_step(dev_loader, "[ Validating dev.. ]", model=model, metrics=metrics)
        for key, metric in metrics.items():
            print(f"{key} dev {metric.evaluate()}")


if __name__ == "__main__":
    main()
