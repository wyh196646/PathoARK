#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import shlex
import sys
import datetime as dt
from pathlib import Path
from typing import Dict, Any, List, Tuple
import asyncio
import signal

# ============== 仅在这里改这两个即可 ==============
GPU_IDS = ["6", "7"]          # 可用 GPU 列表；只有一张卡就写 ["0"]
MAX_TASKS_PER_GPU = 3         # 同时并行在每张卡上跑的最大进程数
# ================================================

# ============== 终端输出控制（可选） ==============
PRINT_TO_TERMINAL = True      # True: 打到终端并带前缀; False: 仅写日志不打印
LINE_PREFIX_FMT = "[GPU {gpu}][{task}] "  # 终端前缀模板
# ================================================

# ------------------ 公共默认参数（按需改） ------------------
DEFAULT_COMMON: Dict[str, Any] = {
    "model_arch": "gigapath_slide_enc12l768d",
    "tile_embed_size": 1536,   # --input_dim
    "latent_dim": 768,
    "epochs": 25,
    "gc": 32,
    "blr": 0.0002,
    "optim_wd": 0.05,
    "layer_decay": 0.95,
    "feat_layer": "11",
    "dropout": 0.1,
    "drop_path_rate": 0.0,
    "val_r": 0.1,
    "warmup_epochs": 1,
    "model_select": "last_epoch",
    "lr_scheduler": "cosine",
    "folds": 5,
    "report_to": "tensorboard",
    "max_wsi_size": 250000,
    "ark_pretrained_ckpt": "/home/yuhaowang/project/PathARK/Pretraining/outputs/ark_pretrain/ark_pretrain_e149.pth.tar",
}

# ------------------ 任务定义（照你原来的配） ------------------
TASK_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "BCNB_ER":{
        "task_cfg_path": "task_configs/biomarker/BCNB_ER.yaml",
        "dataset_csv":   "dataset_csv/biomarker/BCNB_ER.csv",
        "pre_split_dir": "dataset_split/BCNB_ER/",
        "root_path":     "/data4/embedding/BCNB/Gigapath_tile",
    },
    "BCNB_PR":{ 
        "task_cfg_path": "task_configs/biomarker/BCNB_PR.yaml",
        "dataset_csv":   "dataset_csv/biomarker/BCNB_PR.csv",
        "pre_split_dir": "dataset_split/BCNB_PR/",
        "root_path":     "/data4/embedding/BCNB/Gigapath_tile",
    },
    "BCNB_HER2":{
        "task_cfg_path": "task_configs/biomarker/BCNB_HER2.yaml",
        "dataset_csv":   "dataset_csv/biomarker/BCNB_HER2.csv",
        "pre_split_dir": "dataset_split/BCNB_HER2/",
        "root_path":     "/data4/embedding/BCNB/Gigapath_tile",
    },
    "TCGA-BRCA-SUBTYPE": {
        "task_cfg_path": "task_configs/subtype/TCGA-BRCA-SUBTYPE.yaml",
        "dataset_csv":   "dataset_csv/subtype/TCGA-BRCA-SUBTYPE.csv",
        "pre_split_dir": "dataset_split/TCGA-BRCA-SUBTYPE/",
        "root_path":     "/data4/embedding/TCGA-BRCA/Gigapath_tile",
    },

    "AIDPATH_CERB2": {
        "task_cfg_path": "task_configs/biomarker/AIDPATH_CERB2.yaml",
        "dataset_csv":   "dataset_csv/biomarker/AIDPATH_CERB2.csv",
        "pre_split_dir": "dataset_split/AIDPATH_CERB2/",
        "root_path":     "/data4/embedding/AIDPATH/Gigapath_tile",
    },

    "TCGA-BRCA_M": {
        "task_cfg_path": "task_configs/subtype/TCGA-BRCA_M.yaml",
        "dataset_csv":   "dataset_csv/subtype/TCGA-BRCA_M.csv",
        "pre_split_dir": "dataset_split/TCGA-BRCA_M/",
        "root_path":     "/data4/embedding/TCGA-BRCA/Gigapath_tile",
    },
    "TCGA-BRCA_N": {
        "task_cfg_path": "task_configs/subtype/TCGA-BRCA_N.yaml",
        "dataset_csv":   "dataset_csv/subtype/TCGA-BRCA_N.csv",
        "pre_split_dir": "dataset_split/TCGA-BRCA_N/",
        "root_path":     "/data4/embedding/TCGA-BRCA/Gigapath_tile",
    },
    "TCGA-BRCA_T": {
        "task_cfg_path": "task_configs/subtype/TCGA-BRCA_T.yaml",
        "dataset_csv":   "dataset_csv/subtype/TCGA-BRCA_N.csv",
        "pre_split_dir": "dataset_split/TCGA-BRCA_T/",
        "root_path":     "/data4/embedding/TCGA-BRCA/Gigapath_tile",
    },
    "SLNBREAST_SUBTYPE": {
        "task_cfg_path": "task_configs/subtype/SLNBREAST_SUBTYPE.yaml",
        "dataset_csv":   "dataset_csv/subtype/SLNBREAST_SUBTYPE.csv",
        "pre_split_dir": "dataset_split/SLNBREAST_SUBTYPE/",
        "root_path":     "/data4/embedding/SLN-Breast/Gigapath_tile",
    },

    "CPTAC-IDC": {
        "task_cfg_path": "task_configs/subtype/CPTAC_IDC.yaml",
        "dataset_csv":   "dataset_csv/subtype/CPTAC_IDC.csv",
        "pre_split_dir": "dataset_split/CPTAC_IDC/",
        "root_path":     "/data4/embedding/CPTAC/Gigapath_tile",
    },
    "AIDPATH_GRADE":{
        "task_cfg_path": "task_configs/subtype/AIDPATH_GRADE.yaml",
        "dataset_csv":   "dataset_csv/subtype/AIDPATH_GRADE.csv",
        "pre_split_dir": "dataset_split/AIDPATH_GRADE/",
        "root_path":     "/data4/embedding/AIDPATH/Gigapath_tile",
   
    },
    "CAMELYON17_STAGE.yaml":{
        "task_cfg_path": "task_configs/subtype/CAMELYON17_STAGE.yaml",
        "dataset_csv":   "dataset_csv/subtype/CAMELYON17_STAGE.csv",
        "pre_split_dir": "dataset_split/CAMELYON17_STAGE/",
        "root_path":     "/data4/embedding/CAMELYON17/Gigapath_tile",
       
    },
    "DORID_2":{
        "task_cfg_path": "task_configs/subtype/DORID_2.yaml",
        "dataset_csv":   "dataset_csv/subtype/DORID_2.csv",
        "pre_split_dir": "dataset_split/DORID_2/",
        "root_path":     "/data4/embedding/DORID/Gigapath_tile",
    }
    
}

# ------------------ 工具函数（内部使用） ------------------
def ensure_path(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else base / p

def validate_task_paths(base: Path, task_cfg: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for key in ["task_cfg_path", "dataset_csv"]:
        p = ensure_path(base, task_cfg[key])
        if not p.exists():
            errs.append(f"缺失 {key}: {p}")
    pre = task_cfg.get("pre_split_dir")
    if pre:
        pp = ensure_path(base, pre)
        if not pp.exists():
            errs.append(f"缺失 pre_split_dir: {pp}")
    return errs

def build_command(task_name: str,
                  task_cfg: Dict[str, Any],
                  merged: Dict[str, Any],
                  base: Path) -> Tuple[str, List[str]]:
    task_cfg_path = str(ensure_path(base, task_cfg["task_cfg_path"]))
    dataset_csv   = str(ensure_path(base, task_cfg["dataset_csv"]))
    pre_split_dir = task_cfg.get("pre_split_dir")
    if pre_split_dir:
        pre_split_dir = str(ensure_path(base, pre_split_dir))
    root_path     = task_cfg["root_path"]  # 允许是绝对路径

    exp_name = (
        f"{task_name}_e{merged['epochs']}_blr-{merged['blr']}"
        f"_wd-{merged['optim_wd']}_ld-{merged['layer_decay']}_feat-{merged['feat_layer']}"
    )

    save_dir = Path("outputs") / task_name
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd_parts = [
        sys.executable,
        "main.py",
        "--task_cfg_path", task_cfg_path,
        "--dataset_csv", dataset_csv,
        "--root_path", root_path,
        "--model_arch", str(merged["model_arch"]),
        "--blr", str(merged["blr"]),
        "--layer_decay", str(merged["layer_decay"]),
        "--optim_wd", str(merged["optim_wd"]),
        "--dropout", str(merged["dropout"]),
        "--drop_path_rate", str(merged["drop_path_rate"]),
        "--val_r", str(merged["val_r"]),
        "--epochs", str(merged["epochs"]),
        "--input_dim", str(merged["tile_embed_size"]),
        "--latent_dim", str(merged["latent_dim"]),
        "--feat_layer", str(merged["feat_layer"]),
        "--warmup_epochs", str(merged["warmup_epochs"]),
        "--gc", str(merged["gc"]),
        "--ark_pretrained_ckpt", str(merged["ark_pretrained_ckpt"]),
        "--model_select", str(merged["model_select"]),
        "--lr_scheduler", str(merged["lr_scheduler"]),
        "--folds", str(merged["folds"]),
        "--save_dir", str(save_dir),
        "--report_to", str(merged["report_to"]),
        "--exp_name", exp_name,
        "--max_wsi_size", str(merged["max_wsi_size"]),
    ]
    if pre_split_dir:
        cmd_parts.extend(["--pre_split_dir", pre_split_dir])

    return exp_name, cmd_parts

# =============== 并行运行（asyncio） ===============
class TaskItem:
    def __init__(self, task_name: str, exp_name: str, cmd: List[str]):
        self.task_name = task_name
        self.exp_name = exp_name
        self.cmd = cmd

async def _stream_lines(proc: asyncio.subprocess.Process, prefix: str, fh):
    assert proc.stdout is not None
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        s = line.decode() if isinstance(line, (bytes, bytearray)) else line
        # 写日志
        fh.write(s); fh.flush()
        # 可选：终端实时输出（行级，不会把一行拆开）
        if PRINT_TO_TERMINAL:
            sys.stdout.write(prefix + s)
            sys.stdout.flush()

async def run_one_async(task: TaskItem, gpu_id: str, log_dir: Path, base_env: Dict[str, str]) -> int:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task.exp_name}.log"
    env = base_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = dt.datetime.now()
    header = (f"[START] {task.task_name} {task.exp_name} at {start.isoformat()} | GPU {gpu_id}\n"
              f"COMMAND: CUDA_VISIBLE_DEVICES={gpu_id} " + ' '.join(shlex.quote(x) for x in task.cmd) + "\n\n")
    if PRINT_TO_TERMINAL:
        print(header, end='')

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(header); fh.flush()

        proc = await asyncio.create_subprocess_exec(
            *task.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env
        )
        prefix = LINE_PREFIX_FMT.format(gpu=gpu_id, task=task.task_name)
        rc = 0
        try:
            await _stream_lines(proc, prefix, fh)
            rc = await proc.wait()
        except asyncio.CancelledError:
            try: proc.terminate()
            except ProcessLookupError: pass
            try: await asyncio.wait_for(proc.wait(), timeout=10)
            except asyncio.TimeoutError:
                try: proc.kill()
                except ProcessLookupError: pass
            rc = -1
        finally:
            tail = f"\n[END] return_code={rc} duration={dt.datetime.now() - start}\n"
            fh.write(tail); fh.flush()
            if PRINT_TO_TERMINAL:
                print(tail, end='')
        return rc

async def worker(worker_id: int, gpu_id: str, q: "asyncio.Queue[TaskItem]", log_dir: Path, env_base: Dict[str, str], results: Dict[str, int]):
    while True:
        task = await q.get()
        if task is None:
            q.task_done()
            return
        if PRINT_TO_TERMINAL:
            print(f"[RUN] (worker {worker_id} | GPU {gpu_id}) -> {task.task_name} :: {task.exp_name}")
        rc = await run_one_async(task, gpu_id, log_dir, env_base)
        results[task.task_name] = rc
        q.task_done()

async def run_all_parallel(prepared: List[Tuple[str, str, List[str]]], gpu_ids: List[str], max_tasks_per_gpu: int, base_dir: Path, env_base: Dict[str, str]) -> Dict[str, int]:
    log_dir = base_dir / 'outputs' / 'logs'
    q: asyncio.Queue[TaskItem] = asyncio.Queue()
    results: Dict[str, int] = {}

    # 把所有任务放入队列
    for tname, ename, cmd in prepared:
        await q.put(TaskItem(tname, ename, cmd))

    # 为每张 GPU 启动 max_tasks_per_gpu 个并行 worker
    workers = []
    wid = 0
    for gid in gpu_ids:
        for _ in range(max_tasks_per_gpu):
            wid += 1
            workers.append(asyncio.create_task(worker(wid, gid, q, log_dir, env_base, results)))

    # 信号处理：Ctrl+C 优雅停止
    loop = asyncio.get_running_loop()
    def _stop():
        if PRINT_TO_TERMINAL:
            print("\n[INFO] 收到中断信号，正在停止所有任务…")
        for w in workers:
            w.cancel()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            # Windows 可能不支持
            pass

    await q.join()
    # 发哨兵结束 worker
    for _ in workers:
        await q.put(None)
    await asyncio.gather(*workers, return_exceptions=True)
    return results

# ------------------ 主流程（并行） ------------------
def main():
    if not GPU_IDS:
        print("[ERROR] 未配置 GPU_IDS")
        sys.exit(1)
    if MAX_TASKS_PER_GPU < 1:
        print("[ERROR] MAX_TASKS_PER_GPU 必须 >= 1")
        sys.exit(1)

    base_dir = Path(__file__).resolve().parent
    env_base = os.environ.copy()

    # 1) 验证并准备任务
    prepared: List[Tuple[str, str, List[str]]] = []  # (task_name, exp_name, cmd)
    results: Dict[str, int] = {}

    for task_name, cfg in TASK_DEFINITIONS.items():
        # errs = validate_task_paths(base_dir, cfg)
        # if errs:
        #     print(f"[SKIP] 任务 {task_name} 路径校验失败：")
        #     for e in errs:
        #         print("   -", e)
        #     results[task_name] = 998  # 特殊退出码：校验失败
        #     continue

        merged = dict(DEFAULT_COMMON)
        merged.update(cfg.get("overrides", {}) or {})
        exp_name, cmd = build_command(task_name, cfg, merged, base_dir)
        prepared.append((task_name, exp_name, cmd))

    if not prepared:
        print("[ERROR] 无可运行任务（均已校验失败或未定义）。")
        sys.exit(2)

    total_capacity = len(GPU_IDS) * MAX_TASKS_PER_GPU
    print(f"[INFO] 并行执行 {len(prepared)} 个任务；GPU: {','.join(GPU_IDS)}；每卡并行: {MAX_TASKS_PER_GPU}（总并发: {total_capacity}）")

    # 2) 并行执行
    try:
        par_results = asyncio.run(run_all_parallel(prepared, GPU_IDS, MAX_TASKS_PER_GPU, base_dir, env_base))
        results.update(par_results)
    except KeyboardInterrupt:
        print("\n[INFO] 主进程收到 Ctrl+C，退出。")

    # 3) 汇总
    print("\n===== SUMMARY =====")
    ok = 0
    for n in sorted(results):
        rc = results[n]
        status = 'OK' if rc == 0 else f'FAIL({rc})'
        if rc == 0:
            ok += 1
        print(f"{n}: {status}")
    print(f"Successful: {ok}/{len(results)}")
    all_done = (ok == len(results))
    print(f"All finished: {all_done}")
    sys.exit(0 if all_done else 3)

if __name__ == '__main__':
    main()
