'''
DISABLE_VERSION_CHECK=1 python main_dhj.py examples/train_lora/zzytest.yaml


FORCE_TORCHRUN=1 DISABLE_VERSION_CHECK=1 python main_dhj.py examples/train_lora/zzytest.yaml

DISABLE_VERSION_CHECK=1 torchrun --nnodes 1 --node_rank 0 --nproc_per_node 1 --master_addr 127.0.0.1 --master_port 23456 main_dhj.py examples/train_lora/zzytest.yaml


'''
import os
import sys
import random
import importlib
import subprocess

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



def main():
    command = sys.argv.pop(1)
    if command != 'train':
        raise ValueError(f'Unknown command: {command}')
    from dataflex.train.hparams.dynamic_params import DynamicFinetuningArguments
    import llamafactory.hparams.finetuning_args
    llamafactory.hparams.finetuning_args.FinetuningArguments = DynamicFinetuningArguments

    uncache(["llamafactory.hparams.finetuning_args"])


    # do some hack
    from dataflex.train.trainer.dynamic_trainer import DynamicTrainer
    import llamafactory.train.sft.trainer
    llamafactory.train.sft.trainer.CustomSeq2SeqTrainer = DynamicTrainer



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