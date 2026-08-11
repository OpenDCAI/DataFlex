import sys
import importlib
from omegaconf import OmegaConf
from pathlib import Path
from llamafactory.train.tuner import run_exp  # use absolute import

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
    valid_types = ["static", "dynamic_select", "dynamic_mix", "dynamic_weight", "dynamic_reorder"]
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
    elif train_type == "dynamic_reorder":
        from dataflex.train.trainer.reorder_trainer import ReorderTrainer
        TrainerCls = ReorderTrainer
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

def patch_reorder_get_dataset(cfg):
    """
    将 get_dataset 替换为"先按分数重排原始行、再做预处理"的版本。

    只有 apply_at == 'raw' 时才需要：分数字段在 align_dataset 里就被删掉了，
    而预处理不保 index（脏样本会被丢弃、packing 会合并行），所以直接重排原始
    数据集，让顺序自然传递下去。apply_at == 'index' 时顺序在 trainer 里施加，
    数据加载流程无需改动。

    Returns:
        bool: 是否真的打了补丁。
    """
    from dataflex.utils.load_component import load_component

    name = cfg.get('component_name')
    cfg_file = cfg.get('components_cfg_file', 'src/dataflex/configs/components.yaml')
    if not name:
        raise ValueError("train_type='dynamic_reorder' requires `component_name`.")

    from dataflex.core.registry import REGISTRY
    from dataflex.train.data.loader import make_reorder_get_dataset
    from dataflex.train.reorderer import resolve_reorderer_kind  # also registers the reorderers

    params = load_component('reorderers', cfg_file, name, runtime_vars={})
    kind = resolve_reorderer_kind(name, params)

    # 只有"静态 + 在原始行上重排"才需要改数据加载。动态排序的分数来自当前模型，
    # 顺序必然是在 trainer 里按 dataset index 施加的。
    if kind != 'static' or params.get('apply_at', 'raw') != 'raw':
        print(f"[PatchReorder] reorderer '{name}' orders by dataset index; dataset loading left untouched.")
        return False

    def reorderer_factory():
        return REGISTRY.build('reorderer', kind, runtime={}, cfg=params)

    _new_get_dataset = make_reorder_get_dataset(reorderer_factory)

    # 与 patch_get_dataset 同样的四处覆盖：源头模块、包层 re-export、以及
    # sft/pt 两个 workflow 里已经 from-import 过的全局符号。
    data_loader_mod = importlib.import_module("llamafactory.data.loader")
    setattr(data_loader_mod, "get_dataset", _new_get_dataset)
    data_pkg = importlib.import_module("llamafactory.data")
    setattr(data_pkg, "get_dataset", _new_get_dataset)
    wflow = importlib.import_module("llamafactory.train.sft.workflow")
    setattr(wflow, "get_dataset", _new_get_dataset)
    pt_wflow = importlib.import_module("llamafactory.train.pt.workflow")
    setattr(pt_wflow, "get_dataset", _new_get_dataset)

    print(f"[PatchReorder] reorderer '{name}' will permute raw rows before preprocessing.")
    return True


def read_args():
    file_path = sys.argv[1]
    override_config = OmegaConf.from_cli(sys.argv[2:])
    
    if file_path.endswith((".yaml", ".yml", ".json")):
        dict_config = OmegaConf.load(Path(file_path).absolute())
        cfg = OmegaConf.merge(dict_config, override_config)
    else:
        cfg = OmegaConf.create({})  # CLI 直接传参时
    
    return OmegaConf.to_container(cfg)


def launch():
    print("Launching DataFlex")
    cfg = read_args()
    train_type = cfg.get('train_type', 'static')
    patch_finetune_params()
    patch_trainer(train_type)
    if train_type == 'dynamic_mix':
        patch_get_dataset()
    elif train_type == 'dynamic_reorder':
        patch_reorder_get_dataset(cfg)
    patch_train_from_scratch_pad()

    from llamafactory.train.tuner import run_exp
    run_exp()


if __name__ == "__main__":
    launch()