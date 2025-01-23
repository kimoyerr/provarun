import logging
import os
from typing import Any, Dict, Optional
import itertools
from dataclasses import dataclass, field    
import math
from pathlib import Path
import torch
from torch import nn
from torch.optim import Optimizer
from datasets import DatasetDict, Dataset



class StatefulDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """
    From: https://github.com/pytorch/data/blob/main/torchdata/stateful_dataloader/sampler.py#L132
    """

    _YIELDED = "yielded"

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.yielded = 0
        self.next_yielded = None

    def __iter__(self):
        self.yielded = 0
        if self.next_yielded is not None:
            self.yielded = self.next_yielded
            self.next_yielded = None
        it = super().__iter__()
        for idx in itertools.islice(it, self.yielded, None):
            self.yielded += 1
            yield idx

    def state_dict(self) -> Dict[str, Any]:
        return {self._YIELDED: self.yielded}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self._YIELDED not in state_dict:
            raise ValueError("Invalid state_dict")
        if state_dict[self._YIELDED] < 0:
            raise ValueError("Cannot load state_dict with negative yielded value")
        self.next_yielded = state_dict[self._YIELDED]


# Modified from: https://github.com/facebookresearch/flow_matching/blob/main/examples/text/logic/state.py
@dataclass
class Dataset:
    dataset: DatasetDict = field(metadata={"help": "Huggingface dataset"})
    sampler: StatefulDistributedSampler = field(
        metadata={"help": "Stateful sampler for `dataset`"}
    )

@dataclass
class DataState:
    train: Dataset = field(metadata={"help": "Train dataset"})
    test: Dataset = field(metadata={"help": "Test dataset"})


# TODO: I turned off the data_state for now. Add it back in later
class TrainState:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        step: int,
        # data_state: DataState,
    ):
        self._model = model
        self._optimizer = optimizer
        self._step = step
        # self._data_state = data_state

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._step = value

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def data_state(self) -> DataState:
        return self._data_state

    def compile_model(self) -> None:
        self._model = torch.compile(self._model)

    def restore_checkpoint(
        self, ckpt_dir: Path, device: torch.device, rank: int
    ) -> None:
        if ckpt_dir.exists():
            loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=True)

            # self.optimizer.load_state_dict(loaded_state["optimizer"])
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(loaded_state["model"])
            else:
                self.model.load_state_dict(loaded_state["model"])
            self.step = loaded_state["step"]
            # self._data_state.test.load_state_dict(loaded_state["test_sampler"])
            # self._data_state.train.sampler.load_state_dict(
            #     loaded_state["train_sampler"]
            # )
        else:
            ckpt_dir.parent.mkdir(exist_ok=True, parents=True)

            if rank == 0:
                logging.warning(
                    f"No checkpoint found at {ckpt_dir}. Returned the same state as input"
                )

    def save_checkpoint(self, ckpt_dir: str, rank: int) -> None:
        saved_state = {
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            "step": self.step,
            # "train_sampler": self._data_state.train.sampler.state_dict(),
            # "test_sampler": self._data_state.test.sampler.state_dict(),
        }

        if rank == 0:
            torch.save(saved_state, ckpt_dir)

    def eval(self) -> None:
        self.train(training=False)

    def train(self, training: bool = True) -> None:
        self._model.train(mode=training)



def get_lr(it, all, learning_rate, warmup_iters):
    warmup_iters = warmup_iters
    lr_decay_iters = all
    min_lr = learning_rate / 10

    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)



@dataclass
class WorkDirectory:
    root: Path = field(metadata={"help": "Root work directory"})
    checkpoint: Path = field(metadata={"help": "Checkpoint directory"})
    samples: Path = field(metadata={"help": "Samples directory"})


def get_work_dirs(work_dir: str, rank: int) -> WorkDirectory:
    work_dir = Path(work_dir)

    sample_dir = work_dir / "samples"
    checkpoint_dir = work_dir / "checkpoints" / "checkpoint.pth"

    if rank == 0:
        sample_dir.mkdir(exist_ok=True)
        checkpoint_dir.parents[0].mkdir(exist_ok=True)

    return WorkDirectory(root=work_dir, checkpoint=checkpoint_dir, samples=sample_dir)