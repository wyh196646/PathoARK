import os
import time
import copy
import yaml
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from model.multitask_head import MultiTaskHead
from utils import slide_collate_fn_two_views, adjust_learning_rate, log_writer


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    """Initialize DDP.
    修复要点:
    1) 如果通过 torchrun 启动但没有加 --distributed，也自动开启.
    2) 每个进程设置独立 device: cuda:local_rank
    3) 仅在成功初始化后才保持 args.distributed=True
    """
    torchrun_env = ('RANK' in os.environ and 'WORLD_SIZE' in os.environ)
    if torchrun_env and not args.distributed:
        # 自动打开分布式
        args.distributed = True
    if not args.distributed:
        args.distributed = False
        return

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    elif args.rank != -1 and args.world_size != -1:
        pass  # provided by launch utility
    else:
        print('Distributed parameters not set. Disabling distributed.')
        args.distributed = False
        return

    # Properly set device for each process according to local_rank
    torch.cuda.set_device(args.local_rank)
    args.device = f"cuda:{args.local_rank}"
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()
    # 每个进程都打印一次，方便核对 (可以只主进程打印 rank map)
    print(f"[DDP Init] rank={args.rank} local_rank={args.local_rank} world_size={args.world_size} device={args.device} current_device={torch.cuda.current_device()}")


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

            # 混合精度前向，模型保持原始精度，权重不被永久 cast
            with torch.amp.autocast('cuda', dtype=torch.float16):
                # Use keyword arguments to safeguard against DataParallel edge cases
                feat_t, _ = teacher(images=imgs2, coords=coords2, head_n=i)
                feat_s, pred_s = model(images=imgs1, coords=coords1, head_n=i)

                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    labels = labels.squeeze(-1).long()
                loss_cls = criterion(pred_s, labels)
                loss_const = mse(feat_s, feat_t)
                loss = (1 - coff) * loss_cls + coff * loss_const

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0 and is_main_process():
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
            with torch.amp.autocast('cuda', dtype=torch.float16):
                _, logits = model(images=imgs, coords=coords, head_n=i)
                if isinstance(criterion, torch.nn.CrossEntropyLoss):
                    labels = labels.squeeze(-1).long()
                loss = criterion(logits, labels)
            bsz = imgs.size(0)
            total += loss.item() * bsz
            n += bsz
        loss_val = total / max(1, n)
        # reduce across processes
        if is_dist_avail_and_initialized():
            t = torch.tensor([loss_val], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            loss_val = (t / dist.get_world_size()).item()
        losses.append(loss_val)
    return losses


def ark_pretrain_run(args, build_dataset_fn, datasets_yaml: str):
    # init DDP if requested
    init_distributed_mode(args)
    # after init_distributed_mode we ensure args.device is specific per rank when distributed
    device = torch.device(args.device)
    # build datasets
    train_names, train_sets, train_criterions, num_classes_list = build_datasets_from_yaml(datasets_yaml, "train", build_dataset_fn)
    val_names, val_sets, val_criterions, _ = build_datasets_from_yaml(datasets_yaml, "val", build_dataset_fn)

    # distributed samplers
    train_samplers = []
    val_samplers = []
    if args.distributed:
        for ds in train_sets:
            train_samplers.append(torch.utils.data.distributed.DistributedSampler(ds, shuffle=True))
        for ds in val_sets:
            val_samplers.append(torch.utils.data.distributed.DistributedSampler(ds, shuffle=False))
    else:
        train_samplers = [None] * len(train_sets)
        val_samplers = [None] * len(val_sets)

    train_loaders = []
    for ds, sampler in zip(train_sets, train_samplers):
        train_loaders.append(
            DataLoader(ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=slide_collate_fn_two_views)
        )
    val_loaders = []
    for ds, sampler in zip(val_sets, val_samplers):
        val_loaders.append(
            DataLoader(ds, batch_size=args.batch_size, shuffle=False, sampler=sampler,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=slide_collate_fn_two_views)
        )

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

    if args.distributed:
        # teacher 只在本地复制，不包 DDP（EMA 手动更新）
        # 如果逐个数据集单独 forward 某个 head，会导致其它 head 的参数在该 iteration 未被使用 -> DDP 报错
        # 解决方案: 启用 find_unused_parameters 或者一次性使用所有 heads. 这里优先使用 find_unused_parameters.
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=getattr(args, 'find_unused_parameters', True)
        )

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

    # 移除显式 half 转换，统一由 autocast 控制
    if is_main_process():
        os.makedirs(args.save_dir, exist_ok=True)
    ckpt_base = os.path.join(args.save_dir, "ark_pretrain")

    best_metric = float("inf")
    it = 0
    writer = None
    for epoch in range(args.epochs):
        # step LR cosine per-iteration handled by adjust_learning_rate (optional); here keep simple per-epoch
        #adjust_learning_rate(optimizer, epoch, args)
        # set epoch for samplers
        if args.distributed:
            for s in train_loaders:
                if hasattr(s, 'sampler') and isinstance(s.sampler, torch.utils.data.distributed.DistributedSampler):
                    s.sampler.set_epoch(epoch)
        it = train_one_epoch(model, teacher, train_loaders, train_names, criterion_list, optimizer, epoch, args, momentum_sched, coef_sched, it, writer)
        val_losses = evaluate_losses(model, val_loaders, val_names, val_criterions, device)
        avg_val = float(np.mean(val_losses))
        if is_main_process():
            print(f"Epoch {epoch}: avg_val_loss={avg_val:.5f} per-dataset={dict(zip(val_names, [round(v,5) for v in val_losses]))}")

        # simple metric: average val loss
        if is_main_process():
            if avg_val < best_metric:
                best_metric = avg_val
                torch.save({
                    "epoch": epoch,
                    "state_dict": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                    "teacher": teacher.state_dict(),
                    "val_losses": val_losses,
                }, ckpt_base + ".pth.tar")

        # periodic snapshot
        if is_main_process() and ((epoch + 1) % args.test_epoch == 0 or (epoch + 1) == args.epochs):
            torch.save({
                "epoch": epoch,
                "state_dict": (model.module.state_dict() if isinstance(model, DDP) else model.state_dict()),
                "teacher": teacher.state_dict(),
                "val_losses": val_losses,
            }, ckpt_base + f"_e{epoch}.pth.tar")

    return ckpt_base + ".pth.tar"
