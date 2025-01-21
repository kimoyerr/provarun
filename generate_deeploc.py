from pathlib import Path
import torch
from torch import optim
from transformers import EsmTokenizer
from hydra import initialize, compose
from provarun.generate_utils import generate_samples
from provarun.train_utils import TrainState
from provarun.models import GPT
from provarun.flow_utils import MaskedSourceDistribution, MixtureDiscreteProbPath, PolynomialConvexScheduler


# Config
# TODO: Add a main at some point to make this cleaner with using Hydra
# Initialize Hydra and load the configuration 
config_dir = "config/" 
with initialize(config_path=config_dir): 
    all_cfg = compose(config_name="generate_config")
model_cfg = all_cfg.model
generate_cfg = all_cfg.generate
flow_cfg = all_cfg.flow

# Load model
model_seed = 1
model_name = f'facebook/esm1v_t33_650M_UR90S_{model_seed}'
tokenizer = EsmTokenizer.from_pretrained(model_name)
mask_token_id = tokenizer._convert_token_to_id(tokenizer.mask_token)



# Flow setup
# TODO: In the original meta implementation, they created an extra token at the end and used that for masking
source_distribution = MaskedSourceDistribution(mask_token=mask_token_id)
if flow_cfg.scheduler_type == "polynomial":
    scheduler = PolynomialConvexScheduler(n=flow_cfg.exponent)
else:
    raise ValueError(f"{flow_cfg.scheduler_type} is not supported")

path = MixtureDiscreteProbPath(scheduler=scheduler)


# Load model with config
model = GPT(model_cfg)  
model.to("cuda")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Load checkpoint
device = "cuda"
rank = 0
# TODO: Turned off data_state for now but need to add it back in
state = TrainState(model=model, optimizer=optimizer, step=1)
state.restore_checkpoint(ckpt_dir=Path(generate_cfg.ckpt_path), device=device, rank=rank)
# Elbo may have singularity at 1
time_epsilon = 0.0
vocab_size = tokenizer.vocab_size
samples = generate_samples(
                model=state.model,
                step=state.step,
                sample_dir=generate_cfg.sample_dir,
                vocab_size=vocab_size,
                tokenizer=tokenizer,
                rank=rank,
                device=device,
                path=path,
                source_distribution=source_distribution,
                sample_batch_size=generate_cfg.batch_size,
                sequence_length=model_cfg.max_seq_len,
                sampling_steps=flow_cfg.sampling_steps,
                time_epsilon=time_epsilon,
            )