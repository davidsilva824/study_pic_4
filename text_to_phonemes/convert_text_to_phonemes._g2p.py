### Use this one for phoneme llama. 

from g2p import make_g2p

lines = ['fedex']

# English text -> English IPA
transducer = make_g2p('eng', 'eng-ipa') # add   neural=True for the neural model. 

phonemized = [transducer(line).output_string for line in lines]

print()
print(phonemized)