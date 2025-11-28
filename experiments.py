from minicons import scorer
from nltk.tokenize import TweetTokenizer

def get_masked_word_surprisal(model, text):
    """
    Calculates word-level surprisal for Masked models (e.g., BERT, RoBERTa).
    
    CRITICAL: Uses 'within_word_l2r' metric. 
    This forces the model to predict sub-tokens (eat -> er) sequentially, 
    allowing us to validly sum them up without future leakage.
    """
    # 1. Get raw token scores with the CORRECT METRIC
    # PLL_metric='within_word_l2r' is mandatory here.
    token_scores = model.token_score(
        text, 
        surprisal=True, 
        base_two=True, 
        PLL_metric='within_word_l2r' 
    )[0]
    
    # 2. Setup the "Word Ruler" (Tokenizer)
    tokenizer = TweetTokenizer()
    target_words = tokenizer.tokenize(text)
    
    final_results = []
    token_idx = 0
    
    # 3. Manual Aggregation Loop
    for word in target_words:
        current_surprisal = 0.0
        reconstructed = ""
        
        # Add up model tokens until they form the word
        while token_idx < len(token_scores):
            tok_text, tok_score = token_scores[token_idx]
            
            # Clean BERT-style artifacts (## = suffix)
            clean_tok = tok_text.replace('##', '').strip()
            
            reconstructed += clean_tok
            current_surprisal += tok_score
            token_idx += 1
            
            if reconstructed == word:
                break
        
        final_results.append((word, current_surprisal))
        
    return final_results

# --- USAGE EXAMPLE ---
# lm_bert = scorer.MaskedLMScorer("bert-base-uncased", device="cpu")
# results = get_masked_word_surprisal(lm_bert, "this monster is a rat eater")
# print(results)