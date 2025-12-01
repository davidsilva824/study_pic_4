from g2p_plus import transcribe_utterances

lines = ['This monster is a rat eater']

phonemized = transcribe_utterances(lines, "phonemizer", "en-gb", keep_word_boundaries=True)

print(phonemized)