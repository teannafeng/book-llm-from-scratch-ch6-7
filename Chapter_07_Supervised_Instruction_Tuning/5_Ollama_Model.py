# %%
import psutil
import json
from tqdm import tqdm
import urllib.request
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Change the working dir to current folder
os.chdir(os.path.dirname(__file__))

from Utils import format_entry

# %% 
# File management
file_path = "instruction-data-with-response.json"

# %%
# Load test data
DATA_LOADERS = __import__("2_Create_Dataloaders")
test_data = DATA_LOADERS.test_data

# %%
# Specify helper functions
def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

def check_if_ollama_running():
    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    
    print(f"Ollama running:", check_if_running("ollama"))

# query_model("What do Llamas eat?"), model
def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {
            "seed": 123,
            "tempterature": 0,
            "num_ctx": 2048,
        }
    }

    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]
    return response_data

def generate_model_scores(json_data, json_key, model="llama3", verbose=False):
    scores = []
    
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_entry(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
            )
        score = query_model(prompt, model)

        if verbose:
            print("\nDataset response:")
            print(">>", entry['output'])
            print("\nModel response:")
            print(">>", entry['model_response'])
            print("\nScore:")
            print(">>", query_model(prompt))
            print("\n-------------------------------")
        else:
            try:
                scores.append(int(score))
            except ValueError:
                print(f"Could not convert score: {score}")
                continue
    return scores

# %%
# ollama needs to be running to run the remaining code
check_if_ollama_running()

# %%
# Load test data
with open(file_path, "r") as file:
    test_data = json.load(file)

# %%
# Score model-generated responses
scores = generate_model_scores(test_data, "model_response")

# %%
# Print results
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")

# %%
