from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("bbunzeck/phoneme-llama", use_fast=False)

# Try plain text
text = "this monster is a rat eater"
print("TEXT:", text)
print("TOKENS (text):")
for t in tok.tokenize(text):
    print(t)

print()

