import itertools as it

import apex
import efficientnet_pytorch


def get_model(config):
    return efficientnet_pytorch.EfficientNet.from_pretrained(
        config.network.name,
        num_classes=config.network.num_classes,
    )


def add_weight_decay(model, weight_decay=1e-4, skip_list=("bn",)):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def get_opt(args, model, criterion):
    # Scale learning rate based on global batch size
    if args.opt.opt == "Adam":
        opt = apex.optimizers.FusedAdam(
            it.chain(
                model.parameters(), criterion.parameters()
            ),  # add_weight_decay(model, args.weight_decay, ('bn', )),
            lr=args.opt.lr,
            weight_decay=args.opt.weight_decay,
        )
    elif args.opt.opt == "SGD":
        opt = apex.optimizers.FusedSGD(
            add_weight_decay(model, args.opt.weight_decay, ("bn",)),
            args.opt.lr,
            momentum=args.opt.momentum,
            weight_decay=args.opt.weight_decay,
        )
    else:
        raise

    return opt
