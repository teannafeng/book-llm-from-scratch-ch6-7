# %%
import urllib.request
import zipfile
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Constants import *

# %%
# File management
URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = "sms_spam_collection.zip"
EXTRACTED_PATH = "sms_spam_collection"
DATA_FILE_PATH = Path(EXTRACTED_PATH) / "sms_spam_collection.tsv"

# %%
# Function to download and save data
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return
    
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    os.remove(zip_path)

    original_file_path = Path(extracted_path) / "SMSSpamCollection" # name of the downloaded file
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

# %%
if __name__ == "__main__":
    # Download spam dataset
    download_and_unzip_spam_data(URL, ZIP_PATH, EXTRACTED_PATH, DATA_FILE_PATH)