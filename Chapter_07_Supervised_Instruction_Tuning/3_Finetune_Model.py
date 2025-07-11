# %%
import sys
import os
import re
import time
import torch

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Change the working dir to current folder
os.chdir(os.path.dirname(__file__))

from Constants import TOKENIZER
from Utils import load_pretrained_model, train_model_simple, set_device, format_entry, plot_losses

# %% 
# File managemnet
which_model = "gpt2-medium (355M)"

# %%
# Add data loaders
DATA_LOADERS = __import__("2_Create_Dataloaders")
train_loader = DATA_LOADERS.train_loader
validate_loader = DATA_LOADERS.validate_loader
test_loader = DATA_LOADERS.test_loader

# %%
# Load pretrained model
device = set_device("cuda") # both the model and the data need to be on the same device

model, settings, params = load_pretrained_model(which_model=which_model, models_dir="../gpt2")
model = model.to(device) 

# %%
# Fine-tune model
star_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.0005,
    weight_decay=0.1
)
num_epochs = 2

train_losses, validate_losses, tokens_seen = train_model_simple(
    model, train_loader, validate_loader, optimizer, 
    device=device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context=format_entry(DATA_LOADERS.validate_data[0]), 
    tokenizer=TOKENIZER,
    task="instruction"
)
end_time = time.time()
execution_time_mins = (end_time - star_time) / 60
print(f"Training completed in {execution_time_mins:.2f} minutes.")

# %%
# Plot training losses over epoches
epoches_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epoches_tensor, tokens_seen, train_losses, validate_losses)

# %% 
# Save weights of the fine-tuned model 
# To same both structure and weights, use torch.save(model, FILE_NAME)
file_name = f"{re.sub(r'[ ()]', '', which_model) }-sft.pth"

torch.save(model.state_dict(), file_name)
print(f"Model weights saved as {file_name}")

# %%
