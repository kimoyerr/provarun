import os
import hydra
from hydra import initialize, compose
from transformers import EsmTokenizer

from omegaconf import DictConfig
from provarun.data_utils import split_deeploc_data, build_hf_data_loader
from provarun.models import GPT


# TODO: Add a main at some point to make this cleaner with using Hydra
# Initialize Hydra and load the configuration 
config_dir = "config/" 
with initialize(config_path=config_dir): 
    all_cfg = compose(config_name="config")

model_cfg = all_cfg.model

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
data_loader = build_hf_data_loader('ecoli_protein_train', dataloader_dir_path, "train", "Sequence", tokenizer, batch_size=24, seq_len=256, world_size=1, rank=0, infinite=True)
val_data_loader = build_hf_data_loader('ecoli_protein_val', dataloader_dir_path, "validation", "Sequence", tokenizer, batch_size=24, seq_len=256, world_size=1, rank=0, infinite=True)
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

