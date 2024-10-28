import pickle
import json


def read_file(file):
    with open(file) as f:
        return [line.strip() for line in f.readlines()]

def read_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
    
def write_json(file, data):
    with open(file, 'w+', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
def pprint_json(data):
    print(json.dumps(data, sort_keys=True, indent=4))