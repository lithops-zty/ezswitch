import argparse

from easygoogletranslate import EasyGoogleTranslate
from tqdm import tqdm

def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--src', type=str)
    arg_parser.add_argument('--tgt', type=str)
    arg_parser.add_argument('--input', type=str)
    arg_parser.add_argument('--output', type=str)
    
    args = arg_parser.parse_args()
    return args

def main(args):
    translator = EasyGoogleTranslate(
        source_language=args.src,
        target_language=args.tgt,
        timeout=10
    )
    
    print(f"reading input file {args.input}")
    # read input file
    with open(args.input) as f:
        raw_lines = [line.strip() for line in f.readlines()]
    
    print(f"translating from {args.src} to {args.tgt}")
    translated_lines = []
    for line in tqdm(raw_lines):
        result = translator.translate(line)
        translated_lines.append(result)
    
    print(f"writing output file {args.output}")
    with open(args.output, 'w+') as f:
        for line in translated_lines:
            f.write(f"{line}\n")
            
if __name__ == "__main__":
    args = init()
    
    main(args)