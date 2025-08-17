# %%
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Utils import download_and_load_file

# %% 
# File management
file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

# %%
# Download and save data
data = download_and_load_file(file_path, url)
print(f"Number of entries: {len(data)}")
