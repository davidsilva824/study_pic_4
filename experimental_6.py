from transformers import AutoTokenizer

model_name = "phonemetransformers/GPT2-85M-CHAR-PHON"
tok = AutoTokenizer.from_pretrained(model_name)

vocab = tok.get_vocab()
print("Vocab size:", len(vocab))

# Print all tokens (one per line)
for token, idx in sorted(vocab.items(), key=lambda x: x[1]):  # sort by id
    print(idx, repr(token))