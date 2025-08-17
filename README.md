This repository contains partly modified code based on the last two chapters of [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka.

## Notes
- This repository is intended as a proof-of-concept demo and can be further refined for pedagogical use.
- Credit for the original code and concepts goes to Sebastian Raschka.
- The file structure and some functions have been adapted for personal use; core content remains unchanged.
- Scripts in chapter folders are numbered for ordering and tracking.

## Folder structure
```text
example-llm-from-scratch/
│
├── Chapter_06_Supervised_Binary_Classifier_Tuning/    # Code and data from Chapter 6
├── Chapter_07_Supervised_Instruction_Tuning/          # Code and data from Chapter 7
├── gpt2/                                              # Pretrained model files based on GPT2
│
├── Dataset_Instruction.py                             # Torch Dataset setup for instruction tuning
├── Dataset_Spam.py                                    # Torch Dataset setup for binary classification (spam)
│
├── GPT_Download.py                                    # Script to download pretrained GPT model
├── GPT_Model.py                                       # GPT model definition
├── Utils.py                                           # Shared utility functions
├── Constants.py                                       # Global constants
│
├── .gitignore
└── .gitattributes
```

