import os
from contextlib import nullcontext
from hydra import initialize, compose
import torch
from transformers import EsmTokenizer
from torchinfo import summary

from omegaconf import DictConfig
from provarun.data_utils import split_deeploc_data, build_hf_data_loader
from provarun.models import GPT
from provarun.train_utils import get_lr


# TODO: Add a main at some point to make this cleaner with using Hydra
# Initialize Hydra and load the configuration 
config_dir = "config/" 
with initialize(config_path=config_dir): 
    all_cfg = compose(config_name="config")

model_cfg = all_cfg.model
train_cfg = all_cfg.train

# Prepare data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data/deeploc2_1")  # Data from deeploc2.1: https://pmc.ncbi.nlm.nih.gov/articles/PMC11223819/
dataset_name = "deeploc_2_1"
data_csv_path = os.path.join(data_dir, "Swissprot_Train_Validation_dataset.csv")
dataloader_dir_path = os.path.join(data_dir, "dataloader")
# split data into train and validation
_ = split_deeploc_data(data_csv_path, dataloader_dir_path)
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
# tokenizer.add_tokens(['<bos>', '<eos>'])
data_loader, num_train_seqs = build_hf_data_loader('ecoli_protein_train', dataloader_dir_path, "train", "Sequence", tokenizer, batch_size=4, seq_len=model_cfg.max_seq_len, world_size=1, rank=0, infinite=True)
val_data_loader, num_val_seqs = build_hf_data_loader('ecoli_protein_val', dataloader_dir_path, "validation", "Sequence", tokenizer, batch_size=4, seq_len=model_cfg.max_seq_len, world_size=1, rank=0, infinite=True)
dataloader_tag_dict = {
    "tokenizer": "facebook/esm1v_t33_650M_UR90S_1",
    "batch_size": 8,
    "seq_len": 256,
}

data_iterator = iter(data_loader)
batch = next(data_iterator)
print(batch)


# Load model with config
model = GPT(model_cfg)  
model.to("cuda")

# Forward pass
x = batch[0]["aa_inputs"]["input_ids"]
labels = batch[1]["aa_labels"]
x = x.to("cuda")
labels = labels.to("cuda")
output = model(x, labels)

# Training
# Get training context
ctx = nullcontext() if  train_cfg.device == "cpu" else torch.amp.autocast("cuda")
scaler = torch.amp.GradScaler(enabled=train_cfg.dtype in ["float16", "bfloat16"])
optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

# TODO: Compile model
if model_cfg.compile_model:
    unoptimized_model = model
    model = torch.compile(model)


# Train loop
iter_per_epoch = num_train_seqs // train_cfg.batch_size
epoch = 0
while epoch < train_cfg.num_epochs:
    for step, batch in enumerate(data_loader):
        X = batch[0]["aa_inputs"]["input_ids"].to("cuda")
        Y = batch[1]["aa_labels"].to("cuda")
        loss_mask = batch[2]["loss_mask"].to("cuda")  # Loss mask is used to mask out the loss for padded tokens during training
        # Get new learning rate
        lr = get_lr(epoch*iter_per_epoch + step, train_cfg.num_epochs*iter_per_epoch, train_cfg.learning_rate, train_cfg.warmup_iters)
        # Set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        with ctx:
            output = model(X,Y)
            loss = output.last_loss / train_cfg.grad_accumulation_steps
            loss_mask = loss_mask.view(-1)
            # Only get loss for tokens that are not padded
            loss = torch.sum(loss * loss_mask) / torch.sum(loss_mask)

            # Scale the loss and do backpropagation
            scaler.scale(loss).backward()

            if (step + 1) % train_cfg.grad_accumulation_steps == 0:
                # Unscale the gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters())

                # step optimizer
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
    
    epoch += 1
