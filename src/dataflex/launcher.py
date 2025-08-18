import sys
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
    import llamafactory.train.sft.trainer
    llamafactory.train.sft.trainer.CustomSeq2SeqTrainer = DynamicTrainer



    from llamafactory.train.tuner import run_exp
    run_exp()


if __name__ == "__main__":
    launch()
