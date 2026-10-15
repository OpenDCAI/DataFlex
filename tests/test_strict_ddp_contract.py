"""Replicated-weight strict selector: shared budget and rank-safe failures."""

import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_multimodal_selection import Collator, CPUAccelerator, TinyVisionModel, fixture_dataset
from torch.nn.parallel import DistributedDataParallel

from dataflex.train.selector.strict_adapt_inf_selector import StrictAdaptInfSelector


def worker(rank, root, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        root = Path(root)
        if rank == 0:
            _, dataset = fixture_dataset(root)
            (root / "rows.json").write_text(json.dumps(dataset))
        dist.barrier()
        dataset = json.loads((root / "rows.json").read_text())
        acc = CPUAccelerator()
        acc.is_main_process = rank == 0

        def selector(directory):
            s = StrictAdaptInfSelector(
                dataset,
                acc,
                Collator(),
                str(root / directory),
                unique_budget=6,
                proj_dim=16,
                projection="countsketch",
                projection_backend="cpu",
                parameter_pattern="lora_",
                cluster_grid=[2],
                num_clusters=2,
                dataset_revision="ddp-fixture",
            )
            s.set_candidate_pool([1, 2, 3, 4, 5, 6, 7, 8])
            return s

        torch.manual_seed(42)
        model = DistributedDataParallel(TinyVisionModel())
        s = selector("cache")
        warm = s.warmup(2)
        first = s.select(model, 2, 2)
        state = s.export_checkpoint_state(global_step=3, opened_step=2, active_indices=first)
        restored = selector("restored")
        restored.restore_checkpoint_state(state, global_step=3, opened_step=2, active_indices=first)
        last = restored.select(model, 4, 4)
        ids = warm + first + last
        gathered = [None, None]
        dist.all_gather_object(gathered, ids)
        assert gathered[0] == gathered[1] and len(set(ids)) == 6
        assert set(ids) <= {1, 2, 3, 4, 5, 6, 7, 8}
        # One corrupt sidecar must make every rank fail before any rank writes.
        error = None
        try:
            selector("corrupt").restore_checkpoint_state(
                state if rank == 0 else None, global_step=3, opened_step=2, active_indices=first
            )
        except ValueError as exc:
            error = str(exc)
        assert error and "state missing" in error
        # A rank-zero filesystem error must be broadcast to rank one too.
        import dataflex.train.selector.strict_adapt_inf_selector as module

        original = module.write_json

        def fail(*args, **kwargs):
            raise OSError("fixture denied")

        if rank == 0:
            module.write_json = fail
        try:
            error = None
            try:
                selector("failed_write").restore_checkpoint_state(
                    state, global_step=3, opened_step=2, active_indices=first
                )
            except RuntimeError as exc:
                error = str(exc)
            assert error and "fixture denied" in error
        finally:
            module.write_json = original
    finally:
        dist.destroy_process_group()


def test_strict_two_rank_candidate_history_and_failure_propagation(tmp_path):
    mp.spawn(worker, args=(str(tmp_path), str(tmp_path / "rendezvous")), nprocs=2, join=True)
