# %%
import time
import sys
import os
import torch

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Change the working dir to current folder
os.chdir(os.path.dirname(__file__))

from Utils import load_pretrained_model, train_classifier_simple, plot_values, calc_accuracy_loader, set_device
from Constants import BASE_CONFIG

# %%
# Get created data loaders
DATA_LOADERS = __import__("2_Create_Dataloaders")
train_loader = DATA_LOADERS.train_loader
validate_loader = DATA_LOADERS.validate_loader
test_loader = DATA_LOADERS.test_loader

# %% 
# Load pretrained model
device = set_device("cuda") # both the model and the data need to be on the same device
model, settings, params = load_pretrained_model(which_model="gpt2-small (124M)", models_dir="../gpt2")

# %% 
# Modify pretrained model
for param in model.parameters():
    param.requires_grad = False

num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
model.to(device)

for param in model.trf_blocks[-1].parameters(): # last transformer block
    param.requires_grad = True

for param in model.final_norm.parameters(): # last normalization layer
    param.requires_grad = True

# %% 
# Finetune model
start_time = time.time()
torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model,
    train_loader,
    validate_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=50,
    eval_iter=5,
    task="classification"
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# %% 
# Plot loss results
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# %% 
# Plot accuracy results
epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

# %% 
# Print accuracy values
train_acc = calc_accuracy_loader(train_loader, model, device)
validate_acc = calc_accuracy_loader(validate_loader, model, device)
test_acc = calc_accuracy_loader(test_loader, model, device)

print(f"Training accuracy: {train_acc*100:.2f}%")
print(f"Validation accuracy: {validate_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")

# %% 
# Save tuned model
torch.save(model, "spam_classifier.pth")
