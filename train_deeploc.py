import os
import time
from pathlib import Path
from contextlib import nullcontext
from hydra import initialize, compose
import torch
from transformers import EsmTokenizer
from torchinfo import summary

from omegaconf import DictConfig
from provarun.data_utils import split_deeploc_data, build_hf_data_loader, corrupt_data
from provarun.models import GPT
from provarun.train_utils import get_lr, TrainState, get_work_dirs
from provarun.flow_utils import MaskedSourceDistribution, MixtureDiscreteProbPath, PolynomialConvexScheduler, flow_matching_path_sample


# TODO: Add a main at some point to make this cleaner with using Hydra
# Initialize Hydra and load the configuration 
config_dir = "config/" 
with initialize(config_path=config_dir): 
    all_cfg = compose(config_name="config")

model_cfg = all_cfg.model
train_cfg = all_cfg.train
flow_cfg = all_cfg.flow

# Working directory
if train_cfg.work_dir is None:
    work_dir = "working_dir"
else:
    work_dir = train_cfg.work_dir
os.makedirs(work_dir, exist_ok=True)
rank = 0
work_dirs = get_work_dirs(work_dir=work_dir, rank=rank)

# Prepare data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data/deeploc2_1")  # Data from deeploc2.1: https://pmc.ncbi.nlm.nih.gov/articles/PMC11223819/
dataset_name = "deeploc_2_1"
data_csv_path = os.path.join(data_dir, "Swissprot_Train_Validation_dataset.csv")
dataloader_dir_path = os.path.join(data_dir, "dataloader")
# split data into train and validation
if train_cfg.overfit_mode:
    # If debug mode, only use 2*batch_size sequences for training
    _ = split_deeploc_data(data_csv_path, dataloader_dir_path, debug=train_cfg.overfit_mode, num_training_seq=2*train_cfg.batch_size)
else:
    _ = split_deeploc_data(data_csv_path, dataloader_dir_path, debug=False)
num_training_seq = 1000
num_validation_seq = 100
random_seed = 42
data_tag_dict = {
    "dataset_name": dataset_name,
    "num_training_seq": num_training_seq,
    "num_validation_seq": num_validation_seq,
    "random_seed_train_val": random_seed,
}


# Create huggingface dataloader (inspired by torchtitan and my regular way to create a DDP aware dataloader)
model_seed = 1
model_name = f'facebook/esm1v_t33_650M_UR90S_{model_seed}'
tokenizer = EsmTokenizer.from_pretrained(model_name)
mask_token_id = tokenizer._convert_token_to_id(tokenizer.mask_token)
# tokenizer.add_tokens(['<bos>', '<eos>'])
data_loader, num_train_seqs = build_hf_data_loader('ecoli_protein_train', dataloader_dir_path, "train", "Sequence", tokenizer, batch_size=train_cfg.batch_size, seq_len=model_cfg.max_seq_len, world_size=1, rank=0, infinite=True, flow_matching=model_cfg.flow_matching)
val_data_loader, num_val_seqs = build_hf_data_loader('ecoli_protein_val', dataloader_dir_path, "validation", "Sequence", tokenizer, batch_size=train_cfg.batch_size, seq_len=model_cfg.max_seq_len, world_size=1, rank=0, infinite=False, flow_matching=model_cfg.flow_matching)
dataloader_tag_dict = {
    "tokenizer": "facebook/esm1v_t33_650M_UR90S_1",
    "batch_size": 8,
    "seq_len": 256,
}


# Flow setup
# TODO: In the original meta implementation, they created an extra token at the end and used that for masking
source_distribution = MaskedSourceDistribution(mask_token=mask_token_id)
if flow_cfg.scheduler_type == "polynomial":
    scheduler = PolynomialConvexScheduler(n=flow_cfg.exponent)
else:
    raise ValueError(f"{flow_cfg.scheduler_type} is not supported")

path = MixtureDiscreteProbPath(scheduler=scheduler)


# Dataloader and checking batches look goo
data_iterator = iter(data_loader)
batch = next(data_iterator)
# print(batch)
x_1 = batch[0]["aa_inputs"]["input_ids"]
labels = batch[1]["aa_labels"]
times = batch[3]["times"]
x_1 = x_1.to("cuda")
labels = labels.to("cuda")

# Sample from path
time_epsilon=0.0
with torch.no_grad():
    x_0 = source_distribution.sample_like(x_1)
    t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

# # Corrupt the data
# x_corrupt, corrupt_mask = corrupt_data(x, times, mask_token_id)

# Load model with config
model = GPT(model_cfg)  
model.to("cuda")
# Get number of parameters
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {pytorch_total_params}")

model_tag_dict = {
    "model_name": "GPT",
    "model_config": model_cfg,
    "num_parameters": pytorch_total_params,
    "hidden_size": model_cfg.dim,
    "model_seq_len": model_cfg.max_seq_len,
    "model_num_layers": model_cfg.num_transformer_layers,
    "model_num_heads": model_cfg.num_heads,
    "model_ffn_size": model_cfg.ffn_multiple,
}
# Forward pass check
# output = model(x, labels)


# Wandb logging
wandb_run_name = f"DeepLoc-Epoch-{train_cfg.num_epochs}-BatchSize-{train_cfg.batch_size}-LearningRate-{train_cfg.learning_rate}"
if train_cfg.use_wandb:
    import wandb

    wandb.init(project=train_cfg.wandb_project, name=wandb_run_name)
else:
    wandb = None

# Add dictionary to wandb
if wandb is not None:
    # Expand the dictionary
    for key, value in train_cfg.items():
        wandb.config[key] = value
    for key, value in model_cfg.items():
        wandb.config[key] = value
    for key, value in flow_cfg.items():
        wandb.config[key] = value

# Training
# Get training context
ctx = nullcontext() if  train_cfg.device == "cpu" else torch.amp.autocast("cuda")
scaler = torch.amp.GradScaler(enabled=train_cfg.dtype in ["float16", "bfloat16"])
optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

# Train state
device = "cuda"
rank = 0
# TODO: Turned off data_state for now but need to add it back in
state = TrainState(model=model, optimizer=optimizer, step=1)
state.restore_checkpoint(ckpt_dir=work_dirs.checkpoint, device="device", rank=rank)

# TODO: Compile model
if model_cfg.compile_model:
    unoptimized_model = model
    model = torch.compile(model)


# Train loop
iter_per_epoch = num_train_seqs // train_cfg.batch_size
epoch = 0
model.train()
while epoch < train_cfg.num_epochs:
    start_time = time.time()
    for step, batch in enumerate(data_loader):
        X = batch[0]["aa_inputs"]["input_ids"].to("cuda")
        Y = batch[1]["aa_labels"].to("cuda")
        loss_mask = batch[2]["loss_mask"].to("cuda")  # Loss mask is used to mask out the loss for padded tokens during training
        if len(batch) > 3:  # Check if 4th element exists
            times = batch[3]["times"]
            if model_cfg.flow_matching:
                path_sample = flow_matching_path_sample(x_1, source_distribution, path, time_epsilon)
                X = path_sample.x_t.to("cuda")
                times = path_sample.t.to("cuda")
                # TODO: Test what happens if we calculate loss on only the mutated tokens
                # Corrupt_mask is all ones. We want to calculate loss over all tokens to replicate what the meta's example does
                corrupt_mask = torch.ones_like(loss_mask).to("cuda")
            else:
                X, corrupt_mask = corrupt_data(X, times, mask_token_id, flow_matching=model_cfg.flow_matching)
                corrupt_mask = corrupt_mask.to("cuda")
                times = times.to("cuda")
            # Update loss mask to account for the corrupted tokens
            loss_mask = loss_mask * (corrupt_mask)
        # Get new learning rate
        lr = get_lr(epoch*iter_per_epoch + step, train_cfg.num_epochs*iter_per_epoch, train_cfg.learning_rate, train_cfg.warmup_iters)
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        with ctx:
            if times is not None:
                output = model(X, Y, times)
            else:
                output = model(X,Y)
            loss = output.last_loss / train_cfg.grad_accumulation_steps
            loss_mask = loss_mask.view(-1)
            # Only get loss for tokens that are not padded
            loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask)

            # Scale the loss and do backpropagation
            scaler.scale(loss).backward()

            # Step optimizer
            if (step + 1) % train_cfg.grad_accumulation_steps == 0:
                # Unscale the gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

                # step optimizer
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Logging
            if (step + 1) % train_cfg.log_interval == 0:
                step_time = time.time() - start_time
                print(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    train_cfg.num_epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * train_cfg.grad_accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    step_time / (step + 1) * iter_per_epoch // 60 - step_time // 60))
                if (wandb is not None):
                    wandb.log({"loss": loss.item() * train_cfg.grad_accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": step_time / (step + 1) * iter_per_epoch // 60 - step_time // 60})
            
            # Validation
            if (step + 1) % train_cfg.validation_interval == 0:
                model.eval()
                with torch.no_grad():
                    loss_val = 0
                    for val_step, val_batch in enumerate(val_data_loader):
                        X_val = val_batch[0]["aa_inputs"]["input_ids"].to("cuda")
                        Y_val = val_batch[1]["aa_labels"].to("cuda")
                        loss_mask_val = val_batch[2]["loss_mask"].to("cuda")
                        output_val = model(X_val, Y_val)
                        loss_val = output_val.last_loss
                        loss_mask_val = loss_mask_val.view(-1)
                        loss_val = torch.sum(loss_val * loss_mask_val) / torch.sum(loss_mask_val)
                        loss_val = loss_val.item()
                    loss_val /= (val_step + 1)
                    print(f"Validation loss: {loss_val}")
                    if (wandb is not None):
                        wandb.log({"val_loss": loss_val})
                model.train()


            # Save checkpoings
            if (step + 1) % train_cfg.ckpt_save_interval == 0:
                state.save_checkpoint(ckpt_dir=work_dirs.checkpoint, rank=rank)
                print(f"Checkpoint for step {step} saved at {work_dirs.checkpoint}")

    
    epoch += 1
