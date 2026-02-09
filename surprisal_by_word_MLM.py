### For MLM models. 
# Returns the surpisal of full words.
# Uses 'within_word_l2r' metric, that corrects the surprisal of multi-token words. 

from minicons import scorer
from nltk.tokenize import TweetTokenizer

model_name = "BabyLM-community/babylm-baseline-100m-gpt-bert-causal-focus"
text = "cow register"

def get_masked_word_surprisal(model, text):

    token_scores = model.token_score(
        text, 
        surprisal=True, 
        base_two=True, 
        PLL_metric='within_word_l2r' 
    )[0]
    
    tokenizer = TweetTokenizer()
    target_words = tokenizer.tokenize(text)
    
    final_results = []
    token_idx = 0
    
    # Aggregation Loop
    for word in target_words:
        current_surprisal = 0.0
        reconstructed = ""
        
        # Add up model tokens until they form the word
        while token_idx < len(token_scores):
            tok_text, tok_score = token_scores[token_idx]
            
            # Cleans BERT artifacts
            clean_tok = tok_text.replace('##', '').strip()
            
            reconstructed += clean_tok
            current_surprisal += tok_score
            token_idx += 1
            
            if reconstructed == word:
                break
        
        final_results.append((word, current_surprisal))
        
    return final_results


print(f"Loading model: {model_name}...")
lm_bert = scorer.MaskedLMScorer(model_name, device="cpu",  trust_remote_code=True)

print(f"Analyzing sentence: '{text}'")
results = get_masked_word_surprisal(lm_bert, text)

print("\n" + "=" * 30)
print(f"{'WORD':<15} {'SURPRISAL(bits)':<15}")
print("-" * 30)
for w, s in results:
    print(f"{w:<15} {s:.3f}") #to adjust decimal units
print("=" * 30)
