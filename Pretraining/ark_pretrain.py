import os
import time
import copy
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from model.multitask_head import MultiTaskHead
from utils import slide_collate_fn_two_views, adjust_learning_rate, log_writer


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep):
    warmup_iters = 0
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    return schedule


def ema_update_teacher(model, teacher, momentum):
    with torch.no_grad():
        for p_q, p_k in zip(model.parameters(), teacher.parameters()):
            p_k.data.mul_(momentum).add_((1.0 - momentum) * p_q.detach().data)


def build_datasets_from_yaml(cfg_path, split: str, build_dataset_fn):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    datasets = []
    names = []
    num_classes_list = []
    criterions = []
    for name, d in cfg.items():
        names.append(name)
        ds, crit = build_dataset_fn(d, split)
        datasets.append(ds)
        criterions.append(crit)
        num_classes_list.append(int(d["num_classes"]))
    return names, datasets, criterions, num_classes_list


def train_one_epoch(model, teacher, data_loaders, dataset_names, criterion_list, optimizer, epoch, args, momentum_sched, coef_sched, it_start, writer=None):
    device = args.device
    model.train()
    mse = torch.nn.MSELoss()
    it = it_start
    for i, (name, loader, criterion) in enumerate(zip(dataset_names, data_loaders, criterion_list)):
        coff = coef_sched[it]
        start = time.time()
        for step, batch in enumerate(loader):
            imgs1 = batch["imgs1"].to(device)
            coords1 = batch["coords1"].to(device)
            imgs2 = batch["imgs2"].to(device)
            coords2 = batch["coords2"].to(device)
            labels = batch["labels"].to(device).float()

            feat_t, _ = teacher(imgs2, coords2, head_n=i)
            feat_s, pred_s = model(imgs1, coords1, head_n=i)

            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                labels = labels.squeeze(-1).long()
            loss_cls = criterion(pred_s, labels)
            loss_const = mse(feat_s, feat_t)
            loss = (1 - coff) * loss_cls + coff * loss_const

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                dt = time.time() - start
                lr = optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch} [{name}] step {step}/{len(loader)} loss={loss.item():.4f} cls={loss_cls.item():.4f} cons={loss_const.item():.4f} lr={lr:.2e} t/it={dt/(step+1):.3f}s")
            if args.ema_mode == "iteration":
                ema_update_teacher(model, teacher, momentum_sched[it])
                it += 1

        if args.ema_mode == "epoch":
            ema_update_teacher(model, teacher, momentum_sched[it])
            it += 1
    return it


@torch.no_grad()
def evaluate_losses(model, data_loaders, dataset_names, criterion_list, device):
    model.eval()
    losses = []
    for i, (name, loader, criterion) in enumerate(zip(dataset_names, data_loaders, criterion_list)):
        total, n = 0.0, 0
        for batch in loader:
            imgs = batch["imgs1"].to(device)
            coords = batch["coords1"].to(device)
            labels = batch["labels"].to(device).float()
            _, logits = model(imgs, coords, head_n=i)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                labels = labels.squeeze(-1).long()
            loss = criterion(logits, labels)
            bsz = imgs.size(0)
            total += loss.item() * bsz
            n += bsz
        losses.append(total / max(1, n))
    return losses


def ark_pretrain_run(args, build_dataset_fn, datasets_yaml: str):
    device = torch.device(args.device)
    # build datasets
    train_names, train_sets, train_criterions, num_classes_list = build_datasets_from_yaml(datasets_yaml, "train", build_dataset_fn)
    val_names, val_sets, val_criterions, _ = build_datasets_from_yaml(datasets_yaml, "val", build_dataset_fn)

    train_loaders = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, collate_fn=slide_collate_fn_two_views)
        for ds in train_sets
    ]
    val_loaders = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=slide_collate_fn_two_views)
        for ds in val_sets
    ]

    # model + teacher
    model = MultiTaskHead(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        feat_layer=args.feat_layer,
        num_classes_list=num_classes_list,
        model_arch=args.model_arch,
        pretrained=args.pretrained,
        freeze=args.freeze,
        projector_dim=args.projector_features,
        use_mlp=args.use_mlp,
        global_pool=args.global_pool,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
    ).to(device)
    teacher = copy.deepcopy(model).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        teacher = torch.nn.DataParallel(teacher)

    # losses per dataset from builder
    criterion_list = train_criterions

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.optim_wd)

    # schedules
    if args.ema_mode == "epoch":
        momentum_sched = cosine_scheduler(args.momentum_teacher, 1.0, args.epochs, len(train_loaders))
        coef_sched = cosine_scheduler(0.0, 0.5, args.epochs, len(train_loaders))
    else:
        iters_per_epoch = sum(len(dl) for dl in train_loaders)
        momentum_sched = cosine_scheduler(args.momentum_teacher, 1.0, args.epochs, iters_per_epoch)
        coef_sched = cosine_scheduler(0.0, 0.5, args.epochs, iters_per_epoch)

    # logging/checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_base = os.path.join(args.save_dir, "ark_pretrain")

    best_metric = float("inf")
    it = 0
    writer = None
    for epoch in range(args.epochs):
        # step LR cosine per-iteration handled by adjust_learning_rate (optional); here keep simple per-epoch
        adjust_learning_rate(optimizer, epoch, args)
        it = train_one_epoch(model, teacher, train_loaders, train_names, criterion_list, optimizer, epoch, args, momentum_sched, coef_sched, it, writer)
        val_losses = evaluate_losses(model, val_loaders, val_names, val_criterions, device)
        avg_val = float(np.mean(val_losses))
        print(f"Epoch {epoch}: avg_val_loss={avg_val:.5f} per-dataset={dict(zip(val_names, [round(v,5) for v in val_losses]))}")

        # simple metric: average val loss
        if avg_val < best_metric:
            best_metric = avg_val
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "teacher": teacher.state_dict(),
                "val_losses": val_losses,
            }, ckpt_base + ".pth.tar")

        # periodic snapshot
        if (epoch + 1) % args.test_epoch == 0 or (epoch + 1) == args.epochs:
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "teacher": teacher.state_dict(),
                "val_losses": val_losses,
            }, ckpt_base + f"_e{epoch}.pth.tar")

    return ckpt_base + ".pth.tar"
