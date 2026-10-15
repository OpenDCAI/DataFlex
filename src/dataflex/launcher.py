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
    valid_types = ["static", "dynamic_select", "dynamic_mix", "dynamic_weight", "dynamic_reorder", "dynamic_lego"]
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
    elif train_type == "dynamic_lego":
        from dataflex.train.trainer.lego_trainer import LegoTrainer
        TrainerCls = LegoTrainer
    else:  # static
        TrainerCls = None

    if TrainerCls is not None:
        # 1) Replace source module
        tmod = importlib.import_module("llamafactory.train.sft.trainer")
        tmod.CustomSeq2SeqTrainer = TrainerCls

        # 2) Replace package layer re-export
        sft_pkg = importlib.import_module("llamafactory.train.sft")
        setattr(sft_pkg, "CustomSeq2SeqTrainer", TrainerCls)

        # 3) Replace workflow internal references
        wflow = importlib.import_module("llamafactory.train.sft.workflow")
        setattr(wflow, "CustomSeq2SeqTrainer", TrainerCls)
        
        # 4) Replace PT trainer
        pt_tmod = importlib.import_module("llamafactory.train.pt.trainer")
        pt_tmod.CustomTrainer = TrainerCls
        
        # 5) Replace PT workflow internal references
        pt_wflow = importlib.import_module("llamafactory.train.pt.workflow")
        setattr(pt_wflow, "CustomTrainer", TrainerCls)

    print(f"[PatchTrainer] Using trainer type: '{train_type}'")

def patch_get_dataset(do_uncache_reload: bool = False):
    """
    Replace LlamaFactory's get_dataset with dataflex version.
    - Source: llamafactory.data.loader.get_dataset -> dataflex.train.data.loader.get_dataset
    - Package layer re-export: Overwrite llamafactory.data.get_dataset (if any)
    - In-place overwrite: Directly modify the global symbol for already from-imported users (including workflow)

    Args:
        do_uncache_reload: When True, will clear downstream dependency cache and warm up imports to ensure subsequent imports also get the new function.
                           Default is False (consistent with "in-place patching" strategy).
    """
    # 1) Introduce new implementation
    from dataflex.train.data.loader import get_dataset as _new_get_dataset
    # 2) Overwrite source module
    data_loader_mod = importlib.import_module("llamafactory.data.loader")
    setattr(data_loader_mod, "get_dataset", _new_get_dataset)
    # 3) Overwrite package layer re-export (if other code imports from package layer)
    data_pkg = importlib.import_module("llamafactory.data")
    setattr(data_pkg, "get_dataset", _new_get_dataset)
    # 4) In-place overwrite already from-imported users (including workflow)
    wflow = importlib.import_module("llamafactory.train.sft.workflow")
    setattr(wflow, "get_dataset", _new_get_dataset)
    
    # 5) Also patch PT workflow
    pt_wflow = importlib.import_module("llamafactory.train.pt.workflow")
    setattr(pt_wflow, "get_dataset", _new_get_dataset)

def patch_reorder_get_dataset(cfg):
    """
    Replace get_dataset with the version that "first reorder raw rows by score, then preprocess".

    Only needed when apply_at == 'raw': the score field is removed in align_dataset,
    and preprocessing does not preserve index (dirty samples are discarded, packing merges rows),
    so we directly reorder the raw dataset to pass the order naturally.
    When apply_at == 'index', the order is applied in trainer, and the data loading process remains unchanged.

    Returns:
        bool: Whether the patch is actually applied.
    """
    from dataflex.utils.load_component import load_component

    name = cfg.get('component_name')
    cfg_file = cfg.get('components_cfg_file', 'src/dataflex/configs/components.yaml')
    if not name:
        raise ValueError("train_type='dynamic_reorder' requires `component_name`.")

    from dataflex.core.registry import REGISTRY
    from dataflex.train.data.loader import make_reorder_get_dataset
    from dataflex.train.reorder import resolve_reorder_kind  # also registers the reorders

    params = load_component('reorders', cfg_file, name, runtime_vars={})
    kind = resolve_reorder_kind(name, params)

    # Only "static + reorder on raw rows" needs to modify data loading. The dynamic sorting scores come from the current model,
    # the order must be applied in trainer by dataset index.
    if kind != 'static' or params.get('apply_at', 'raw') != 'raw':
        print(f"[PatchReorder] reorder '{name}' orders by dataset index; dataset loading left untouched.")
        return False

    def reorder_factory():
        return REGISTRY.build('reorder', kind, runtime={}, cfg=params)

    _new_get_dataset = make_reorder_get_dataset(reorder_factory)

    # Same four patches as patch_get_dataset: source module, package layer re-export, and
    # already from-imported global symbols in sft/pt workflows.
    data_loader_mod = importlib.import_module("llamafactory.data.loader")
    setattr(data_loader_mod, "get_dataset", _new_get_dataset)
    data_pkg = importlib.import_module("llamafactory.data")
    setattr(data_pkg, "get_dataset", _new_get_dataset)
    wflow = importlib.import_module("llamafactory.train.sft.workflow")
    setattr(wflow, "get_dataset", _new_get_dataset)
    pt_wflow = importlib.import_module("llamafactory.train.pt.workflow")
    setattr(pt_wflow, "get_dataset", _new_get_dataset)

    print(f"[PatchReorder] reorder '{name}' will permute raw rows before preprocessing.")
    return True


def patch_lego_get_dataset():
    """
    Replace get_dataset with the version for combined training.

    The key difference from dynamic_mix: dynamic_mix makes each data source an independent dataset and sets train_dataset to None, so selector/reorder have no indexed datasets.
    The combined path goes the other way around: combine into a dataset, then give an array of domain_id labels for each sample, so the mixer becomes pure index operations, and the other three families work as usual.
    """
    from dataflex.train.data.loader import lego_get_dataset as _new_get_dataset

    # Same four patches as patch_get_dataset: source module, package layer re-export, and
    # already from-imported global symbols in sft/pt workflows.
    data_loader_mod = importlib.import_module("llamafactory.data.loader")
    setattr(data_loader_mod, "get_dataset", _new_get_dataset)
    data_pkg = importlib.import_module("llamafactory.data")
    setattr(data_pkg, "get_dataset", _new_get_dataset)
    wflow = importlib.import_module("llamafactory.train.sft.workflow")
    setattr(wflow, "get_dataset", _new_get_dataset)
    pt_wflow = importlib.import_module("llamafactory.train.pt.workflow")
    setattr(pt_wflow, "get_dataset", _new_get_dataset)

    print("[PatchLego] dataset will carry a per-sample domain_id for the pipeline.")


def read_args():
    file_path = sys.argv[1]
    override_config = OmegaConf.from_cli(sys.argv[2:])
    
    if file_path.endswith((".yaml", ".yml", ".json")):
        dict_config = OmegaConf.load(Path(file_path).absolute())
        cfg = OmegaConf.merge(dict_config, override_config)
    else:
        cfg = OmegaConf.create({})  # When passing CLI arguments directly
    
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
    elif train_type == 'dynamic_lego':
        patch_lego_get_dataset()

    from llamafactory.train.tuner import run_exp
    run_exp(cfg)


if __name__ == "__main__":
    launch()
