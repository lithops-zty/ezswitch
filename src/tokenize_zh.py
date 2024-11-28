import argparse
import jieba

from utils import read_file


def init():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', type=str, required=True)
    argparser.add_argument('--output', type=str, required=True)
    args = argparser.parse_args()
    return args


def main(args):
    print(f"Tokenizing input file {args.input}")
    lines = read_file(args.input)
    tokenized_lines = [
        ' '.join(jieba.cut(line)) for line in lines
    ]

    print(f"Writing tokenized lines to {args.output}")
    with open(args.output, 'w') as f:
        f.write('\n'.join(tokenized_lines))


if __name__ == "__main__":
    args = init()

    main(args)
