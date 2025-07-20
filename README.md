This repository contains partly modified code based on the last two chapters of [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch) by Sebastian Raschka.

## Notes
- This repo is intended for my own learning and reference.
- All original credit goes to Sebastian Raschka.
- The file structure and some functions have been adapted for personal use; core content remains unchanged.
- Scripts in chapter folders are prefixed with numbers for ordering and tracking.

## Structure (subject to change)
```
.
├── Chapter_06_Supervised_Binary_Classifier_Tuning/    # Code and data from Chapter 6
├── Chapter_07_Supervised_Instruction_Tuning/          # Code and data from Chapter 7
├── gpt2/                                              # Pretrained model files based on GPT2
├── Constants.py                                       # Global constants
├── Dataset_Instruction.py                             # Torch Dataset setup for instruction tuning
├── Dataset_Spam.py                                    # Torch Dataset setup for binary classification (spam)
├── GPT_Download.py                                    # Script to download pretrained GPT model
├── GPT_Model.py                                       # GPT model definition
├── Utils.py                                           # Shared utility functions
├── .gitignore
└── .gitattributes
```

