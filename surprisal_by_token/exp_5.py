from g2p_plus import transcribe_utterances
import re

lines = [
    'The post office has purchased a state of the art device that can sort every piece of mail by various shapes. Conveniently enough, the machine is called the size device.'
]

line = lines[0]

print("FULL SENTENCE (repr):")
print(repr(line))
print()

# 1) Test full sentence
print("=== FULL SENTENCE TEST ===")
try:
    out = transcribe_utterances([line], "phonemizer", "en-us", keep_word_boundaries=True)
    print("OK")
    print(out)
except Exception as e:
    print("FAILED:", type(e).__name__, e)
print()

# 2) Split into words (keeps punctuation attached so we can detect if punctuation is the issue)
words = line.split()

print("=== WORD-BY-WORD TEST (raw split words) ===")
failed_raw = []
for i, w in enumerate(words):
    try:
        out = transcribe_utterances([w], "phonemizer", "en-us", keep_word_boundaries=True)
        print(f"[{i:02d}] OK    {repr(w)} -> {out[0]}")
    except Exception as e:
        print(f"[{i:02d}] FAIL  {repr(w)} -> {type(e).__name__}: {e}")
        failed_raw.append((i, w, e))
print()

# 3) Test cleaned words (remove punctuation) to see if punctuation is the cause
print("=== WORD-BY-WORD TEST (cleaned words, punctuation removed) ===")
failed_clean = []
for i, w in enumerate(words):
    clean = re.sub(r"^[^\w']+|[^\w']+$", "", w)  # remove punctuation at start/end
    if not clean:
        continue
    try:
        out = transcribe_utterances([clean], "phonemizer", "en-us", keep_word_boundaries=True)
        print(f"[{i:02d}] OK    raw={repr(w)} clean={repr(clean)} -> {out[0]}")
    except Exception as e:
        print(f"[{i:02d}] FAIL  raw={repr(w)} clean={repr(clean)} -> {type(e).__name__}: {e}")
        failed_clean.append((i, w, clean, e))
print()

# 4) Optional: test cumulative prefixes to find the exact point where the sentence starts failing
print("=== PREFIX TEST (find first failing point) ===")
for i in range(1, len(words) + 1):
    prefix = " ".join(words[:i])
    try:
        transcribe_utterances([prefix], "phonemizer", "en-us", keep_word_boundaries=True)
        print(f"[1..{i:02d}] OK   ends with {repr(words[i-1])}")
    except Exception as e:
        print(f"[1..{i:02d}] FAIL ends with {repr(words[i-1])} -> {type(e).__name__}: {e}")
        print("Failing prefix repr:")
        print(repr(prefix))
        break