# %%
import sys
import os
import torch
import json
from functools import partial

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Constants import TOKENIZER
from Utils import partition_data, create_data_loaders, custom_collate_fn, set_device
from Dataset_Instruction import InstructionDataset

# %%
# Create a version of custom_collate_fn that saves the default device setting
customized_collate_fn = partial(
    custom_collate_fn,
    device=set_device("cuda"),
    allowed_max_length=1024,
)

# %%
# File management
file_path = "instruction-data.json"

torch.manual_seed(123)
num_workers = 0
batch_size = 8

# %%
# Load full data
with open(file_path, "r") as file:
    data = json.load(file)

# %%
# Partition full data
train_data, test_data, validate_data = partition_data(data, 0.85, 0.10)

# %% 
# Create datasets
train_dataset = InstructionDataset(train_data, TOKENIZER)
test_dataset = InstructionDataset(test_data, TOKENIZER)
validate_dataset = InstructionDataset(validate_data, TOKENIZER)

# %%
# Create data loaders
train_loader, test_loader, validate_loader = create_data_loaders(
    train_dataset, test_dataset, validate_dataset, 
    batch_size=batch_size, 
    num_workers=num_workers,
    train_shuffle=True,
    test_shuffle=False,
    validate_shuffle=False,
    collate_fn=customized_collate_fn
    )
