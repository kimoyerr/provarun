from abc import ABC
import torch
from torch import nn, Tensor

from provarun.solver_utils import MixtureDiscreteEulerSolver

    

# From: https://github.com/facebookresearch/flow_matching/blob/main/examples/text/logic/generate.py
class WrappedModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        # Note: logit's precision is important.
        model_out = self.model(x=x, times=t)
        return torch.softmax(model_out.logits.float(), -1)
    

# Modified from: https://github.com/facebookresearch/flow_matching/blob/main/examples/text/logic/generate.py
def generate_samples(
    model,
    step,
    vocab_size,
    tokenizer,
    rank,
    device,
    path,
    source_distribution,
    sample_batch_size,
    sequence_length,
    sampling_steps,
    time_epsilon = 0.0,
    sample_dir = None,
    dtype_categorical = torch.float64,
):
    wrapped_probability_denoiser = WrappedModel(model=model)

    add_token = 1 if source_distribution.masked else 0
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_probability_denoiser,
        path=path,
        vocabulary_size=vocab_size + add_token,
    )

    x_init = source_distribution.sample(
        tensor_size=(sample_batch_size, sequence_length), device=device
    )

    sample = solver.sample(
        x_init=x_init,
        step_size=1 / sampling_steps,
        verbose=True,
        dtype_categorical=dtype_categorical,
        time_grid=torch.tensor([0.0, 1.0 - time_epsilon]),
    )

    sentences = tokenizer.batch_decode(sample)

    if sample_dir is not None:
        file_name = sample_dir / f"iter_{step}" / f"sample_{rank}.txt"
        file_name.parents[0].mkdir(exist_ok=True, parents=True)

        with open(file_name, "w") as file:
            for sentence in sentences:
                file.write(f"{sentence}\n{'=' * 20} New sample {'=' * 20}\n")

    return sample
