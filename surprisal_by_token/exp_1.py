from g2p_plus import transcribe_utterances
from transformers import AutoTokenizer

# 1) Input text
text = "this monster is a rat eater"

# 2) Text -> IPA (g2p-plus)
ipa_text = transcribe_utterances(
    [text],
    backend="phonemizer",
    language="en-us",
    keep_word_boundaries=True
)[0]

print("TEXT:", text)
print("IPA :", ipa_text)
print()

# 3) Load tokenizer (BPE-PHON)
tokenizer = AutoTokenizer.from_pretrained(
    "phonemetransformers/babble-tokenizers",
    subfolder="BABYLM-TOKENIZER-BPE-PHON"
)

# 4) Build dictionary: phoneme -> tokenizer symbol(s) WITHOUT Ġ
#    (WORD_BOUNDARY is ignored)
phoneme_to_utf = {}

for ph in ipa_text.split():
    if ph == "WORD_BOUNDARY":
        continue
    if ph in phoneme_to_utf:
        continue

    toks = tokenizer.tokenize(ph)  # tokenize the phoneme alone
    cleaned = [t.replace("Ġ", "") for t in toks]  # remove GPT-2 word marker
    phoneme_to_utf[ph] = cleaned

# 5) Print dictionary
print("PHONEME -> UTF tokenizer symbols")
for ph, utf_tokens in phoneme_to_utf.items():
    print(f"{ph!r}: {utf_tokens}")