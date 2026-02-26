### Use this one for babble. 

from g2p_plus import transcribe_utterances

lines = ['The post office, has purchased a state.']

phonemized = transcribe_utterances(lines, "phonemizer", "en-us", keep_word_boundaries=True)

print()

print(phonemized)