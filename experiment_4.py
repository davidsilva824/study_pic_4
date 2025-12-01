from transformers import AutoTokenizer

model_name = "phonemetransformers/GPT2-85M-BPE-PHON"  # or any other
tok = AutoTokenizer.from_pretrained(model_name)

vocab = tok.get_vocab()

# Show a sample of 200 tokens
sample = list(vocab.keys())[:200]
for t in sample:
    print(t)
