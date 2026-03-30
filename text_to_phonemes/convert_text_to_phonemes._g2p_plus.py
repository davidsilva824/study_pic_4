### Use this one for babble. 

from g2p_plus import transcribe_utterances

lines = ['labourer']

phonemized = transcribe_utterances(lines, "phonemizer", "en-us", keep_word_boundaries=True)

print()

print(phonemized)