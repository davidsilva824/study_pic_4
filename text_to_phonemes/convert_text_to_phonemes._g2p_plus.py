### Use this one for babble. 

from g2p_plus import transcribe_utterances

lines = ['aitch and ar']

phonemized = transcribe_utterances(lines, "phonemizer", "en-us", keep_word_boundaries=True)

print()

print(phonemized)