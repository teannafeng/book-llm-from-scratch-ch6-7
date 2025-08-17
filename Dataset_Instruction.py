
from torch.utils.data import Dataset
from Utils import format_entry

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_and_input = format_entry(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_and_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)


