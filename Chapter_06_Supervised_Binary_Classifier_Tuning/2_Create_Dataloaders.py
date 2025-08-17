import sys
import os
import pandas as pd
import torch
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the child directory to sys.path
os.chdir(os.path.dirname(__file__))

from Constants import TOKENIZER
from Utils import create_data_loaders, partition_data
from Dataset_Spam import SpamDataset

# File management
torch.manual_seed(123)
num_workers = 0
batch_size = 8

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = "sms_spam_collection.zip"
EXTRACTED_PATH = "sms_spam_collection"
DATA_FILE_PATH = Path(EXTRACTED_PATH) / "sms_spam_collection.tsv"

# Specify helper function(s)
def create_balanced_dataset(df: pd.DataFrame, seed=123):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=seed
    )
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

# Create balanced data
data = pd.read_csv(DATA_FILE_PATH, sep="\t", header=None, names=["Label", "Text"])
balanced_data = create_balanced_dataset(data)
balanced_data["Label"] = balanced_data["Label"].map({"ham": 0, "spam": 1}) 

# Partition data 
train_data, test_data, validate_data = partition_data(balanced_data, 0.70, 0.20)

# Create datasets
train_dataset = SpamDataset(data=train_data, max_length=None, tokenizer=TOKENIZER)
test_dataset = SpamDataset(data=test_data, max_length=train_dataset.max_length, tokenizer=TOKENIZER)
validate_dataset = SpamDataset(data=validate_data, max_length=train_dataset.max_length, tokenizer=TOKENIZER)

# Create data loaders
train_loader, validate_loader, test_loader = create_data_loaders(train_dataset, validate_dataset, test_dataset, batch_size, num_workers)
