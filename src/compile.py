import argparse
import os
import pandas as pd


def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--directory",
        type=str,
    )
    arg_parser.add_argument(
        "--output",
        type=str,
    )

    args = arg_parser.parse_args()
    return args

def main(args):
    directory = args.directory
    output = args.output

    df_dict = {
        "src": [],
        "tgt": [],
        "generated": [],
        "method": [],   
        "model": [],
        "direction": [],
    }
    
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            df = pd.read_csv(os.path.join(directory, filename))
            assert 'src' in df.columns
            assert 'tgt' in df.columns
            assert len(df.columns) > 2

            for column in df.columns:
                if column not in ['src', 'tgt', 'src_translated', 'tgt_translated']:
                    df_dict['src'].extend(df['src'])
                    df_dict['tgt'].extend(df['tgt'])
                    df_dict['generated'].extend(df[column])
                    df_dict['model'].extend([filename.split('.')[0]] * len(df))
                    df_dict['method'].extend([column.split('_')[0]] * len(df))
                    df_dict['direction'].extend([column.split('_')[1]] * len(df))
    
    df = pd.DataFrame(df_dict)
    df.to_csv(output, index=False)  
    print(f"Output saved to {output}")


if __name__ == "__main__":
    args = init()
    main(args)

