import tiktoken

TOKENIZER = tiktoken.get_encoding("gpt2")

END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_ID = TOKENIZER.encode(END_OF_TEXT, allowed_special={END_OF_TEXT})[0]

TRAIN_CSV_PATH = "train.csv"
VALIDATION_CSV_PATH = "validation.csv"
TEST_CSV_PATH = "test.csv"


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {
        "emb_dim": 768, 
        "n_layers": 12,
        "n_heads": 12,
    },
    "gpt2-medium (355M)": {
        "emb_dim": 1024, 
        "n_layers": 24,
        "n_heads": 16 ,
    },
    "gpt2-larger (744M)": {
        "emb_dim": 1280, 
        "n_layers": 36,
        "n_heads": 20,
    },
    "gpt2-xl (1558M)": {
        "emb_dim": 1600, 
        "n_layers": 48,
        "n_heads": 25,
    },
}