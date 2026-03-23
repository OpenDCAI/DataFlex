import os
import sys
import random
import importlib
import subprocess
from omegaconf import OmegaConf
from pathlib import Path

def uncache(exclude):
    """Remove package modules from cache except excluded ones.
    On next import they will be reloaded.
    
    Args:
        exclude (iter<str>): Sequence of module paths.
    """
    pkgs = []
    for mod in exclude:
        pkg = mod.split('.', 1)[0]
        pkgs.append(pkg)

    print(f'{pkgs=}')
    to_uncache = []
    for mod in sys.modules:
        if mod in exclude:
            continue

        if mod in pkgs:
            to_uncache.append(mod)
            continue

        for pkg in pkgs:
            if mod.startswith(pkg + '.'):
                to_uncache.append(mod)
                break

    print(f'{to_uncache=}')
    for mod in to_uncache:
        del sys.modules[mod]


def patch_finetune_params():
    from dataflex.train.hparams.dynamic_params import DynamicFinetuningArguments
    from dataflex.train.hparams.dynamic_data_params import DataArguments
    import llamafactory.hparams
    llamafactory.hparams.finetuning_args.FinetuningArguments = DynamicFinetuningArguments
    llamafactory.hparams.data_args.DataArguments = DataArguments

    uncache(["llamafactory.hparams.finetuning_args", "llamafactory.hparams.data_args"])

def patch_trainer(train_type: str):
    """
    Monkey-patch LlamaFactory's CustomSeq2SeqTrainer based on train_type.

    Args:
        train_type (str): Must be one of ["static", "dynamic_select", "dynamic_mix", "dynamic_weight"].
                          Determines which trainer class to inject.
    """
    valid_types = ["static", "dynamic_select", "dynamic_mix", "dynamic_weight"]
    if train_type not in valid_types:
        raise ValueError(f"Invalid train_type '{train_type}'. Must be one of {valid_types}.")

    if train_type == "dynamic_select":
        from dataflex.train.trainer.select_trainer import SelectTrainer
        TrainerCls = SelectTrainer
    elif train_type == "dynamic_mix":
        from dataflex.train.trainer.mix_trainer import MixTrainer
        TrainerCls = MixTrainer
    elif train_type == "dynamic_weight":
        from dataflex.train.trainer.weight_trainer import WeightTrainer
        TrainerCls = WeightTrainer
    else:  # static
        TrainerCls = None

    if TrainerCls is not None:
        # 1) 替换源头模块
        tmod = importlib.import_module("llamafactory.train.sft.trainer")
        tmod.CustomSeq2SeqTrainer = TrainerCls

        # 2) 替换包层 re-export
        sft_pkg = importlib.import_module("llamafactory.train.sft")
        setattr(sft_pkg, "CustomSeq2SeqTrainer", TrainerCls)

        # 3) 替换 workflow 内部引用
        wflow = importlib.import_module("llamafactory.train.sft.workflow")
        setattr(wflow, "CustomSeq2SeqTrainer", TrainerCls)
        
        # 4) 替换 PT 训练器
        pt_tmod = importlib.import_module("llamafactory.train.pt.trainer")
        pt_tmod.CustomTrainer = TrainerCls
        
        # 5) 替换 PT workflow 内部引用
        pt_wflow = importlib.import_module("llamafactory.train.pt.workflow")
        setattr(pt_wflow, "CustomTrainer", TrainerCls)

    print(f"[PatchTrainer] Using trainer type: '{train_type}'")


def patch_get_dataset(do_uncache_reload: bool = False):
    """
    将 LlamaFactory 的 get_dataset 替换为 dataflex 版本。
    - 源头: llamafactory.data.loader.get_dataset -> dataflex.train.data.loader.get_dataset
    - 包层 re-export: 覆盖 llamafactory.data.get_dataset（如有）
    - 就地覆盖: 对已 from-import 的使用方（包含 workflow）直接改其全局符号

    Args:
        do_uncache_reload: 为 True 时，会清理下游依赖缓存并预热导入，以确保后续 import 也拿到新函数。
                          默认为 False（与“就地打补丁”策略一致）。
    """
    # 1) 引入新实现
    from dataflex.train.data.loader import get_dataset as _new_get_dataset
    # 2) 覆盖源头模块
    data_loader_mod = importlib.import_module("llamafactory.data.loader")
    setattr(data_loader_mod, "get_dataset", _new_get_dataset)
    # 3) 覆盖包层 re-export（若其它代码从包层 import）
    data_pkg = importlib.import_module("llamafactory.data")
    setattr(data_pkg, "get_dataset", _new_get_dataset)
    # 4) 就地覆盖已 from-import 的使用方（包含 workflow）
    wflow = importlib.import_module("llamafactory.train.sft.workflow")
    setattr(wflow, "get_dataset", _new_get_dataset)
    
    # 5) 也要patch PT workflow
    pt_wflow = importlib.import_module("llamafactory.train.pt.workflow")
    setattr(pt_wflow, "get_dataset", _new_get_dataset)

def read_args():
    file_path = sys.argv[1]
    override_config = OmegaConf.from_cli(sys.argv[2:])
    
    if file_path.endswith((".yaml", ".yml", ".json")):
        dict_config = OmegaConf.load(Path(file_path).absolute())
        cfg = OmegaConf.merge(dict_config, override_config)
    else:
        cfg = OmegaConf.create({})  # CLI 直接传参时

    return OmegaConf.to_container(cfg)

def print_welcome():
    try:
        import importlib.metadata as importlib_metadata  # py3.8+
    except ImportError:
        import importlib_metadata

    try:
        version = importlib_metadata.version("dataflex")
    except importlib_metadata.PackageNotFoundError:
        version = "unknown"

    print("=" * 60)
    print(" 🎉 Welcome to DataFlex, a data-centric training system.")
    print(f" 🚀 Installed version: {version}")
    print("=" * 60)

def main():
    command = sys.argv.pop(1)
    if command == "version":
        # 只打印版本和欢迎
        print_welcome()
        return
    elif command != 'train':
        raise ValueError(f'Unknown command: {command}')
    cfg = read_args()
    patch_finetune_params()
    patch_trainer(cfg['train_type'] if 'train_type' in cfg else 'static')
    if cfg['train_type'] == 'dynamic_mix':
        patch_get_dataset()

    from llamafactory.train.tuner import run_exp
    from llamafactory.extras.misc import is_env_enabled, get_device_count, use_ray
    from llamafactory.extras import logging
    from dataflex import launcher


    logger = logging.get_logger(__name__)

    force_torchrun = is_env_enabled("FORCE_TORCHRUN")
    if force_torchrun or (get_device_count() > 1 and not use_ray()):
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(random.randint(20001, 29999)))
        logger.info_rank0(f"Initializing distributed tasks at: {master_addr}:{master_port}")
        process = subprocess.run(
            (
                "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
            )
            .format(
                nnodes=os.getenv("NNODES", "1"),
                node_rank=os.getenv("NODE_RANK", "0"),
                nproc_per_node=os.getenv("NPROC_PER_NODE", str(get_device_count())),
                master_addr=master_addr,
                master_port=master_port,
                file_name=launcher.__file__,
                args=" ".join(sys.argv[1:]),
            )
            .split()
        )
        sys.exit(process.returncode)
    else:
        run_exp()