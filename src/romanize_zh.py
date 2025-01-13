import argparse
import pypinyin
import pandas as pd

import pypinyin

def transcribe_to_pinyin(series):
    pinyin_lines = []
    for line in series:
        pinyin_line = ' '.join(pypinyin.lazy_pinyin(line))
        pinyin_lines.append(pinyin_line)
    return pinyin_lines

def init():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str, required=True)
    argparser.add_argument('--output', type=str, required=True)
    argparser.add_argument('--cols', nargs='*', help='columns to romanize', default=[])
    argparser.add_argument('--exclude_cols', nargs='*', help='columns to exclude from romanization', default=[])
    
    args = argparser.parse_args()
    
    if len(args.cols) != 0 and len(args.exclude_cols) != 0:
        print("Error: cannot specify both --cols and --exclude_cols")
        exit(1)
        
    return args


def main(args):
    print(f"Romanizing input file {args.input}")
    
    df = pd.read_csv(args.input)
    
    if len(args.cols) != 0:
        cols_to_romanize = args.cols
    else:
        cols_to_romanize = [col for col in df.columns if col not in args.exclude_cols]
        
    for col in cols_to_romanize:
        if col not in df.columns:
            print(f"Column {col} not found in input file")
            continue
            
        print(f"Transcribing column {col}")
        pinyin_lines = transcribe_to_pinyin(df[col])
        df[col] = pinyin_lines

    print(f"Writing romanized lines to {args.output}")
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    args = init()

    main(args)
