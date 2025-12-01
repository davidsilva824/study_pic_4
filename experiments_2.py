import torch
from minicons import scorer

# -------------------------------
# Config
# -------------------------------
model_name = "phonemetransformers/GPT2-85M-BPE-TXT"
text = "this moster is a rat eater"
BOS = False  # same as your example

# -------------------------------
# 1. Get raw surprisal from minicons (no correction)
# -------------------------------
lm = scorer.IncrementalLMScorer(model_name, device="cpu")

raw = lm.token_score(
    text,
    bos_token=BOS,
    prob=False,
    surprisal=True,
    bow_correction=False,
    decode=False,
)[0]

tokens = [t for t, _ in raw]
raw_surps = [s for _, s in raw]

print("RAW (minicons, bow=False, surprisal):")
for t, s in raw:
    print(f"{t:10s} {s:.3f}")

# -------------------------------
# 2. Re-implement compute_stats + bow_correction exactly
# -------------------------------

# Step 2.1: prepare_text (same as token_score)
tokenized = lm.prepare_text([text], bos_token=BOS)
encoded, offsets = tokenized

if lm.device != "auto":
    encoded = encoded.to(lm.device)

# ids = list of non-padded ids, exactly as in compute_stats
ids = [
    [i for i, am in zip(instance, attention_mask) if am != 0]
    for instance, attention_mask in zip(
        encoded["input_ids"].tolist(), encoded["attention_mask"].tolist()
    )
]

# effective_ids = drop first token (exact line from compute_stats)
effective_ids = [id_seq[1:] for id_seq in ids]

with torch.no_grad():
    logits = lm.model(**encoded).logits.detach()

# split logits the same way
logits_split = logits.split([1] * len(offsets))

all_corrected_log_scores = []

for logit, idx, offset in zip(logits_split, effective_ids, offsets):
    # idx: list of token ids with first token removed
    length = len(idx)

    # exact lines from compute_stats:
    query_ids = idx[offset:]
    logit = logit.squeeze(0)
    logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)

    actual_logprob_distribution = logprob_distribution[
        torch.arange(offset, length),
    ]

    # score = log p(token | context) for tokens from offset onward
    score = actual_logprob_distribution[
        torch.arange(length - offset), query_ids
    ]

    # BOW logic flag (exact)
    bow_correction = True
    if not lm.is_bow_tokenizer:
        bow_correction = False

    if bow_correction:
        # EXACT bow_correction block from your file
        mask_forward = torch.zeros(length).to(lm.device)
        mask_current = torch.zeros(length).to(lm.device)

        for i in range(len(idx)):
            if i == len(idx) - 1:
                mask_forward[i] = 1
                if not lm.bow_subwords[idx[i]]:
                    mask_current[i] = 0
                else:
                    mask_current[i] = 1
                break
            elif lm.bow_subwords[idx[i + 1]]:
                mask_forward[i] = 1
                if not lm.bow_subwords[idx[i]]:
                    mask_current[i] = 0
                else:
                    mask_current[i] = 1
            else:
                mask_forward[i] = 0
                mask_current[i] = 1

        # apply offset exactly as in file
        mask_forward = mask_forward[offset:]
        mask_current = torch.roll(mask_forward, shifts=1)
        mask_current[0] = 0.0

        bow_subword_idx_tensor = torch.tensor(lm.bow_subword_idx).to(lm.device)

        forward_correction = (
            logprob_distribution[offset:][torch.arange(length - offset) + 1,]
            .index_select(-1, bow_subword_idx_tensor)
            .logsumexp(1)
        )

        current_correction = (
            actual_logprob_distribution[torch.arange(length - offset),]
            .index_select(-1, bow_subword_idx_tensor)
            .logsumexp(1)
        )

        score = (
            score
            + (forward_correction * mask_forward)
            - (current_correction * mask_current)
        )

    # store corrected log-scores for this sequence (only one here)
    all_corrected_log_scores.append(score)

# -------------------------------
# 3. Rebuild token-aligned surprisals like token_score
# -------------------------------
log_scores = all_corrected_log_scores[0]  # one sentence
log_scores_list = log_scores.tolist()     # length = length - offset (here 8)

# In token_score, they pad if len(tokens) > len(scores)
# (they add zeros at the front for the extra token, usually <s>)
if len(tokens) > len(log_scores_list):
    diff = len(tokens) - len(log_scores_list)
    padded_log_scores = [0.0] * diff + log_scores_list
else:
    padded_log_scores = log_scores_list

# Convert to surprisals (token_score does scores = -1.0 * s when surprisal=True)
corrected_surps = [-1.0 * s for s in padded_log_scores]

print("\nEXTERNAL bow-corrected surprisal (should match bow_correction=True):")
for t, s in zip(tokens, corrected_surps):
    print(f"{t:10s} {s:.3f}")
