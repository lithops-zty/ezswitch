import json
import random

with open(r'smsCorpus_en_2015.03.09_all.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

msgs = []
for x in data['smsCorpus']['message']:
    if isinstance(x['text']['$'], str):  # there are some messages that are integers
        msgs.append(x['text']['$'])
random.shuffle(msgs)

with open(r'parsed.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(msgs))