# %%
import sys
import os
import json
from tqdm import tqdm
import torch

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Change the working dir to current folder
os.chdir(os.path.dirname(__file__))

from Constants import TOKENIZER, BASE_CONFIG, END_OF_TEXT_ID, BASE_CONFIG, MODEL_CONFIGS
from GPT_Model import GPTModel
from Utils import format_entry, generate, set_device, text_to_token_ids, token_ids_to_text

# %%
# File management
output_path = "instruction-data-with-response.json"
which_model = "gpt2-medium (355M)"
weights_path = "gpt2-medium355M-sft.pth"

BASE_CONFIG.update(MODEL_CONFIGS[which_model])

# %%
# Load test data
DATA_LOADERS = __import__("2_Create_Dataloaders")
test_data = DATA_LOADERS.test_data

# %%
# Create the base model
device = set_device("cuda")
model = GPTModel(BASE_CONFIG)
model = model.to(device)

# %%
# Load weights of tuned model
state_dict = torch.load(weights_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# %%
# Get model-generated response
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_entry(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, TOKENIZER).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=END_OF_TEXT_ID
    )

    generated_text = token_ids_to_text(token_ids, TOKENIZER)

    response_text = (
        generated_text[len(input_text):].replace("### Response:", "").strip()
    )
    test_data[i]["model_response"] = response_text

# %%
# Save test data + model-generated results
with open(output_path, "w") as file:
    json.dump(test_data, file, indent=4)

# %%
