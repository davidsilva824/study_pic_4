from g2p_plus import transcribe_utterances
from transformers import AutoTokenizer

# 1) Input text
text = "this monster is a rat eater"

# 2) Text -> IPA (same settings)
ipa_text = transcribe_utterances(
    [text],
    backend="phonemizer",
    language="en-us",
    keep_word_boundaries=True
)[0]

print("TEXT:", text)
print("IPA :", ipa_text)
print()

# 3) Load the tokenizer (same one)
tokenizer = AutoTokenizer.from_pretrained(
    "phonemetransformers/babble-tokenizers",
    subfolder="BABYLM-TOKENIZER-BPE-PHON"
)

# 4) Tokenize
tokens = tokenizer.tokenize(ipa_text)

print("TOKENS:")
for i, tok in enumerate(tokens):
    print(i, tok)