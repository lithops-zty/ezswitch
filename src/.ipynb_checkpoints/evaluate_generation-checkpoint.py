import argparse
import pandas as pd
import numpy as np

import evaluate

from utils import read_pickle, write_json, pprint_json

def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input_file', type=str)
    arg_parser.add_argument('--reference', type=str)
    arg_parser.add_argument('--output_file', type=str)
    arg_parser.add_argument('--write_csv', type=bool, default=False)
        
    args = arg_parser.parse_args()
    return args

def main(args):
    print(f"Evaluating {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    reference_dict = read_pickle(args.reference)
    
    rouge = evaluate.load('rouge')
    wer = evaluate.load('wer')
    bleu = evaluate.load("bleu")
    bertscore = evaluate.load("bertscore")
    chrf = evaluate.load("chrf")
    
    references = []
    references_single = []
    for i, row in df.iterrows():
        
        reference = reference_dict[row['src']+'\n']
        
        references.append(reference)
        references_single.append(reference[0])
    
    eval_dict = {}
    for col in df.columns:
        if col in ['src', 'tgt', 'src_translated', 'tgt_translated']:
            continue
        bs = bertscore.compute(predictions = df[col], references = references_single, model_type='bert-base-multilingual-cased')
        eval_dict[col] = {
            'rouge': rouge.compute(predictions = df[col], references = references)['rougeL'],
            'wer': wer.compute(predictions = df[col], references = references_single),
            'bleu': bleu.compute(predictions = df[col], references = references)['bleu'] * 100,
            'bertscore-f1': np.mean(bs['f1']),
            'bertscore-recall': np.mean(bs['recall']),
            'bertscore-precision': np.mean(bs['precision']),
            'chrf': chrf.compute(predictions = df[col], references = references_single)['score']
        }
        
        # eval_dict[col] = {k: round(v,5) for k, v in eval_dict[col].items()}

    pprint_json(eval_dict)
    
    print(f"Writing to {args.output_file}...")
    write_json(args.output_file, eval_dict)
    
    if args.write_csv:
        # For creating reports and plots in excel
        print("Write to temp.csv")
        df_dict = {'method': [], 'rouge': [], 'wer': [], 'bleu': []}
        for col in eval_dict:
            df_dict['method'].append(col)
            df_dict['rouge'].append(eval_dict[col]['rouge'])
            df_dict['wer'].append(eval_dict[col]['wer'])
            df_dict['bleu'].append(eval_dict[col]['bleu'])
            df_dict['chrf'].append(eval_dict[col]['chrf'])
        pd.DataFrame(df_dict).to_csv('output/temp.csv', index=False)
if __name__ == "__main__":
    args = init()
    main(args)