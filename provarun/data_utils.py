import os
from typing import Dict, Optional, Any
import pickle
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import IterableDataset
from torch.distributed.checkpoint.stateful import Stateful
from torchdata.stateful_dataloader import StatefulDataLoader
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torchtitan.datasets.tokenizer import Tokenizer
from torchtitan.logging import init_logger, logger


def split_deeploc_data(csv_path, output_dir, debug=False):


    # Split the dataset into train and validation sets based on the  Partition column
    # Use the 4 folds for training and the last fold for validation    
    df = pd.read_csv(csv_path)
    train_df = df[df['Partition'] != 4]
    val_df = df[df['Partition'] == 4]
    # Write to output directory
    # Check if output_dir exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # If debug only select the first 8 rows. This is to test overfitting
    if debug:
        train_df = train_df.iloc[:8,]
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    
    return output_dir


class HuggingFaceProteinDataset(IterableDataset, Stateful):
    """PyTorch Representation of the HuggingFace Dataset.

    Args:
        dataset_name (str): name of the dataset to load
        dataset_path (Optional[str]):
            Path to the dataset in the file system. If provided, data will be loaded
            from this path instead of downloaded.
        dataset_split (Optional[str]): name of the dataset split to load
        tokenizer (Tokenizer):
            Tokenizer used to encode data. Tokenize must implement an `encode` and `decode` method.
        seq_len (int): max sequence length
        world_size (int): number of data parallel processes participating in training
        rank (int): rank of the current data parallel process
        infinite (bool): whether to loop infinitely over the dataset

    """

    def __init__(
        self,
        dataset_name: str,
        dataset_path: Optional[str],
        dataset_split: Optional[str],
        seq_column: str,
        tokenizer: Tokenizer,
        seq_len: int = 1024,
        world_size: int = 1,
        rank: int = 0,
        infinite: bool = False,
        flow_matching: bool = False,
    ) -> None:

        logger.info(f"Preparing {dataset_name} dataset from {dataset_path}")
        ds = load_dataset(dataset_path, split=dataset_split)
        # TODO: support shuffling and checkpointing
        self.dataset_name = dataset_name
        self._data = split_dataset_by_node(ds, rank, world_size)
        self._seq_column = seq_column
        self._tokenizer = tokenizer
        self.seq_len = seq_len
        self.infinite = infinite
        self.num_repeats = 0
        self.flow_matching = flow_matching

        # variables for checkpointing
        self._sample_idx = 0

    def __iter__(self):
        max_buffer_token_len = 1 + self.seq_len

        while True:
            for sample in self._get_data_iter():
                sample_text = sample[self._seq_column]
                sample_tokens = self._tokenizer.encode(sample_text)
                
                # Generate input and label
                x = torch.LongTensor(sample_tokens[:max_buffer_token_len])
                # update tokens to the remaining tokens
                sample_tokens = sample_tokens[max_buffer_token_len:]
                input = x[:-1]
                label = x[1:]

                yield input, label

            if not self.infinite:
                logger.warning(f"Dataset {self.dataset_name} has run out of data.")
                break
            else:
                # Reset offset for the next iteration
                self._sample_idx = 0
                self.num_repeats += 1
                logger.warning(
                    f"Dataset {self.dataset_name} is being re-looped. "
                    "Loss related metrics might be misleading."
                )

    def collate_fn(self, batch):
        aa_seqs, aa_label_ids = zip(*batch)

        # Create a loss mask tensor filled with 1s with shape (batch_size, seq_len)
        # Loss mask is used to mask out the loss for padded tokens during training
        loss_mask = torch.ones(len(aa_seqs), self.seq_len)
        
        # For each sequence in the batch
        for i, seq in enumerate(aa_seqs):
            # Calculate how many padding tokens are needed to reach seq_len
            pad_len = self.seq_len - len(seq)
            # If padding is needed (pad_len > 0), set the loss mask to 0 
            # for all padding positions at the end of the sequence
            if pad_len > 0:
                loss_mask[i, -pad_len:] = 0
        loss_mask = {"loss_mask": loss_mask}
        
        # Encoder
        encoder_info = self._tokenizer.pad({"input_ids": aa_seqs}, return_tensors='pt', padding=True)
        # Pad to the sequence length
        padding_id = self._tokenizer._convert_token_to_id("<pad>")
        encoder_info = nn.functional.pad(encoder_info["input_ids"], (0, self.seq_len - encoder_info["input_ids"].shape[1]), value=padding_id)
        encoder_info = {"input_ids": encoder_info}
        aa_inputs = {"aa_inputs": encoder_info}
        inputs = {**aa_inputs}

        aa_label_ids = pad_sequences(aa_label_ids, -1)
        aa_label_ids = nn.functional.pad(aa_label_ids, (0, self.seq_len - aa_label_ids.shape[1]), value=-1)
        labels = {
                  "aa_labels": aa_label_ids,
                  }
        
        # If self.flow_matching, also output times
        min_t = 0
        times = torch.rand(len(aa_seqs),) * (1.0 - min_t) + min_t
        # Reshape to same size as labels
        times = times.unsqueeze(1)
        times = {"times": times}

        return inputs, labels, loss_mask, times

    def _get_data_iter(self):
        if self._sample_idx == 0:
            return iter(self._data)

        # Skip samples
        if isinstance(self._data, IterableDataset):
            it = iter(self._data)
            # Naively iterate through the samples as skip may not be supported
            for _ in range(self._sample_idx):
                next(it)
            return it

        # As skipping to the end throws an error in case of map-style dataset, return an empty iterator
        if self._sample_idx == len(self._data):
            return iter([])
        return iter(self._data.skip(self._sample_idx))

    def load_state_dict(self, state_dict):
        self._sample_idx = state_dict["sample_idx"]
        self._all_tokens = state_dict["token_buffer"]
        self._all_catalytic_tokens = state_dict.get("catalytic_token_buffer", [])

    def state_dict(self):
        return {"token_buffer": self._all_tokens, "catalytic_token_buffer": self._all_catalytic_tokens, "sample_idx": self._sample_idx}


class DPAwareDataLoader(StatefulDataLoader, Stateful):
    """
    A wrapper around the StatefulDataLoader that ensures that the state is stored only once per DP rank.
    """

    def __init__(self, dp_rank: int, hf_ds: IterableDataset, batch_size: int, collate_fn):
        super().__init__(hf_ds, batch_size, collate_fn=collate_fn)
        self._dp_rank = dp_rank
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> Dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions
        return {self._rank_id: pickle.dumps(super().state_dict())}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # State being empty is valid, don't log a warning
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self._dp_rank}, expected key {self._rank_id}."
            )
            return
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))


def build_hf_data_loader(
    dataset_name: str,
    dataset_path: Optional[str],
    dataset_split: Optional[str],
    seq_column: str,
    tokenizer: Tokenizer,
    batch_size: int,
    seq_len: int,
    world_size,
    rank,
    infinite: bool = True,
    collate_fn=None,
    flow_matching: bool = False,
):
    hf_ds = HuggingFaceProteinDataset(
        dataset_name, dataset_path, dataset_split, seq_column, tokenizer, seq_len, world_size, rank, infinite,
    )

    # Return the dataloader and the length of the dataset. Length is needed to estimate the number of steps per epoch
    return DPAwareDataLoader(rank, hf_ds, batch_size=batch_size, collate_fn=hf_ds.collate_fn), len(hf_ds._data)


def split_seq(seq):
    """
    Split the sequence into two parts: the amino acid sequence and the foldseek sequence.
    
    Parameters
    ----------
    data : dict
        Dictionary containing the sequence to be split.
    
    Returns
    -------
    dict
        Dictionary containing the split amino acid sequence and the foldseek sequence.
    """
    aa_seq = seq[::2]  # get characters at even indices
    foldseek_seq = seq[1::2]  # get characters at odd indices
    return {"aa_seq": aa_seq, "foldseek_seq": foldseek_seq}



def pad_sequences(sequences, constant_value=0, dtype=None):
	batch_size = len(sequences)
	shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

	if dtype is None:
		dtype = sequences[0].dtype

	if isinstance(sequences[0], np.ndarray):
		array = np.full(shape, constant_value, dtype=dtype)
	elif isinstance(sequences[0], torch.Tensor):
		device = sequences[0].device
		array = torch.full(shape, constant_value, dtype=dtype, device=device)

	for arr, seq in zip(array, sequences):
		arrslice = tuple(slice(dim) for dim in seq.shape)
		arr[arrslice] = seq

	return array



def corrupt_data(X, times, mask_token_id):
     assert times.shape[0] == X.shape[0]

     # Generate random numbers of size X.shape
     u = torch.rand(X.shape)
     # Generate a mask for the data to corrupt. Times close to 0 are more likely to be corrupted
     target_mask = u < (1-times)
     X[target_mask] = mask_token_id

     return X, target_mask