# %%
import torch 
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Change the working dir to current folder
os.chdir(os.path.dirname(__file__))

from Utils import set_device
from Constants import END_OF_TEXT_ID, TOKENIZER


# %% 
# Load tuned model 
device = set_device("cuda")
model= torch.load("spam_classifier.pth", map_location=device)
model.eval()

# %% 
# Function to classify new test data
def classify_review(text, model, tokenizer, device, max_length=999, pad_token_id=END_OF_TEXT_ID):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    input_ids = input_ids[:min(
        max_length, supported_context_length
    )]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"

# %% Test 01 --> "spam"
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(
    text_1, model, TOKENIZER, device, max_length=120
))

# %% Test 02 --> "not spam"
text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(
    text_2, model, TOKENIZER, device, max_length=120
))

# %%
