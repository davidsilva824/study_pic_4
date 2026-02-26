from g2p import make_g2p

g2p = make_g2p("eng", "eng-ipa",  neural=True)

tests = [
    "well",
    "phases",
    "phases classifier",
    "this monster is a rat eater killer"
]

for text in tests:
    out = g2p(text)
    print("-" * 60)
    print("TEXT:      ", repr(text))
    print("RAW OUT:   ", repr(out))
    print("AS STRING: ", str(out))