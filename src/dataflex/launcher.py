import sys
import importlib
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


def launch():
    print("Launching DataFlex")
    from dataflex.train.hparams.dynamic_params import DynamicFinetuningArguments
    import llamafactory.hparams.finetuning_args
    llamafactory.hparams.finetuning_args.FinetuningArguments = DynamicFinetuningArguments
    
    uncache(["llamafactory.hparams.finetuning_args"])


    # do some hack
    from dataflex.train.trainer.dynamic_trainer import DynamicTrainer
    # do some hack
    from dataflex.train.trainer.dynamic_trainer import DynamicTrainer
    # import llamafactory.train.sft.trainer
    # 1) 先确保源头也改了（以防别的地方用到）
    tmod = importlib.import_module("llamafactory.train.sft.trainer")
    tmod.CustomSeq2SeqTrainer = DynamicTrainer

    # 2) 覆盖包层 re-export
    sft_pkg = importlib.import_module("llamafactory.train.sft")
    setattr(sft_pkg, "CustomSeq2SeqTrainer", DynamicTrainer)

    # 3) 给 workflow 模块的全局把同名符号也替换掉
    wflow = importlib.import_module("llamafactory.train.sft.workflow")
    setattr(wflow, "CustomSeq2SeqTrainer", DynamicTrainer)   # 名字必须与它 `from .trainer import ...` 的名字一致



    from llamafactory.train.tuner import run_exp
    run_exp()


if __name__ == "__main__":
    launch()
