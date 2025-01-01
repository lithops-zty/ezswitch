import argparse
import json
import pandas as pd
import pickle
import pprint

import torch
import transformers

from utils import read_file, read_pickle

def init():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--lang1', type=str, default = 'en')
    arg_parser.add_argument('--lang2', type=str, default = 'hi')
    arg_parser.add_argument('--src', type=str, default = '')
    arg_parser.add_argument('--tgt', type=str, default = '')
    arg_parser.add_argument('--src_translated', type=str, default = '')
    arg_parser.add_argument('--tgt_translated', type=str, default = '')
    arg_parser.add_argument('--gold_align', type=str, default = '')
    arg_parser.add_argument('--silver_src_align', type=str, default = '')
    arg_parser.add_argument('--silver_tgt_align', type=str, default = '')
    
    arg_parser.add_argument('--human_reference', type=str, default = '')
    
    arg_parser.add_argument('--model_id', type=str)
    
    arg_parser.add_argument('--output', type=str)
    
    eot_token = {
        "meta-llama/Meta-Llama-3-8B-Instruct": "<|eot_id|>",
        "meta-llama/Llama-3.2-1B-Instruct": "<|eot_id|>",
        "aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct": "<|eot_id|>",
        "CohereForAI/aya-23-8B": "<|END_OF_TURN_TOKEN|>",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "<|eot_id|>",
        "bigscience/bloomz-7b1": None,
        "bigscience/mt0-xxl": None,
        "aisingapore/sea-lion-7b-instruct": None,
        "SeaLLMs/SeaLLMs-v3-7B-Chat": None,
    }
    args = arg_parser.parse_args()
    args.eot_token = eot_token[args.model_id]
    
    return args

    
def create_baseline(src, tgt, lang1, lang2, example_l1, example_l2, example_cs, example_words_l1, example_words_l2):
    SYSTEM_SRC_PROMPT = (f"You are a Bilingual {lang1} {lang2} speaker, "
                         f"you will help translate these {lang1} sentence "
                         f"into a code mixed sentence with romanized {lang2} "
                         f"and {lang1}"
                        )
    SYSTEM_TGT_PROMPT = (f"You are a Bilingual {lang1} {lang2} speaker, "
                         f"you will help translate these {lang2} sentence "
                         f"into a code mixed sentence with romanized {lang2} "
                         f"and {lang1}"
                        )
    src_prompts = []
    tgt_prompts = []
    if src is not None:
        for src_sent in src:
            src_prompts.append([
                {"role": "system", "content": SYSTEM_SRC_PROMPT},
                {"role": "user", "content": example_l1},
                {"role": "assistant", "content": example_cs},
                {"role": "user", "content": src_sent},
            ])
    if tgt is not None:
        for tgt_sent in tgt:
            tgt_prompts.append([
                {"role": "system", "content": SYSTEM_TGT_PROMPT},
                {"role": "user", "content": example_l2},
                {"role": "assistant", "content": example_cs},
                {"role": "user", "content": tgt_sent},
            ])
    
    return src_prompts if len(src_prompts) else None, tgt_prompts if len(tgt_prompts) else None

def get_valid_switching_point(alignment_pairs):
    valid_pairs = []
    for i in range(len(alignment_pairs)):
        valid = True
        for j in range(len(alignment_pairs)):
            ai, bi = alignment_pairs[i]
            aj, bj = alignment_pairs[j]
            if (ai < aj and bi > bj) or (ai > aj and bi < bj):
                valid = False
                break
        if valid:
            valid_pairs.append(alignment_pairs[i])
    return valid_pairs


def create_alignment(src, tgt, alignment, lang1, lang2, example_l1, example_l2, example_cs, example_words_l1, example_words_l2):
    SYSTEM_SRC_PROMPT = (f"You are a Bilingual {lang1} {lang2} speaker, "
                         f"you will help translate these {lang1} sentence "
                         f"into a code mixed sentence with romanized {lang2} "
                         f"and {lang1} with specific key words that we want to appear"
                        )
    SYSTEM_TGT_PROMPT = (f"You are a Bilingual {lang1} {lang2} speaker, "
                         f"you will help translate these {lang2} sentence "
                         f"into a code mixed sentence with romanized {lang2} "
                         f"and {lang1} with specific key words that we want to appear"
                        )
    src_prompts = []
    tgt_prompts = []
    for src_sent, tgt_sent, align_text in zip(src,tgt,alignment):
        align_pairs = [tuple(map(int, pair.split('-'))) for pair in align_text.strip().split()]
        switching_locations = get_valid_switching_point(align_pairs)
        src_words = src_sent.replace('\n', ' ').strip().lower().split()
        tgt_words = tgt_sent.replace('\n', ' ').strip().lower().split()
        valid_src_words = []
        valid_tgt_words = []
        for src_point, tgt_point in switching_locations:
            try:
                valid_tgt_words.append(tgt_words[tgt_point]), valid_src_words.append(src_words[src_point])
            except:
                continue
        src_prompts.append(
            [
                {"role": "system", "content": SYSTEM_SRC_PROMPT},
                {"role": "user", "content": f"{example_l1}\n words wanted: {example_words_l1}"},
                {"role": "assistant", "content": example_cs},
                {"role": "user", "content": f"{src_sent}\n words wanted: {list(set(valid_src_words))}"},
            ]
        )
        tgt_prompts.append(
            [
                {"role": "system", "content": SYSTEM_TGT_PROMPT},
                {"role": "user", "content": f"{example_l2}\n words wanted: {example_words_l2}"},
                {"role": "assistant", "content": example_cs},
                {"role": "user", "content": f"{tgt_sent}\n words wanted: {list(set(valid_tgt_words))}"},
            ]
        )
    return src_prompts, tgt_prompts

def get_constraint(en, eval_set):
    normalize = lambda sent: sent.lower().replace(',','').replace('.','').strip().split()
    normalized_eval = [normalize(sent) for sent in eval_set]
    normalized_en = normalize(en)
    constraints = [set(sent).intersection(set(normalized_en)) for sent in normalized_eval]
    min_constraint = constraints[0]
    min_val = len(min_constraint)
    for c in constraints:
        if len(c) < min_val:
            min_constraint = c
            min_val = len(c)
    return list(min_constraint)

def create_data_leak(src, tgt, human_reference, lang1, lang2):
    SYSTEM_SRC_PROMPT = (f"You are a Bilingual {lang1} {lang2} speaker, "
                         f"you will help translate these {lang1} sentence "
                         f"into a code mixed sentence with romanized {lang2} "
                         f"and {lang1} with specific key words that we want to appear"
                        )
    SYSTEM_TGT_PROMPT = (f"You are a Bilingual {lang1} {lang2} speaker, "
                         f"you will help translate these {lang2} sentence "
                         f"into a code mixed sentence with romanized {lang2} "
                         f"and {lang1} with specific key words that we want to appear"
                        )
    src_prompts = []
    tgt_prompts = []
    for src_sent, tgt_sent in zip(src, tgt):
        eval_set = human_reference[src_sent+'\n']
        constraint = get_constraint(src_sent, eval_set)
        src_prompts.append(
            [
                {"role": "system", "content": SYSTEM_SRC_PROMPT},
                {"role": "user", "content": "The reward of goodness shall be nothing but goodness\n words wanted: ['reward', 'goodness']"},
                {"role": "assistant", "content": "goodness ka reward keval goodness hee hoga."},
                {"role": "user", "content": f"{src_sent}\n words wanted: {constraint}"},
            ]
        )
        tgt_prompts.append(
            [
                {"role": "system", "content": SYSTEM_TGT_PROMPT},
                {"role": "user", "content": "अच्छाई का पुरस्कार अच्छाई के अलावा कुछ नहीं होगा\n words wanted: ['होगा']"},
                {"role": "assistant", "content": "goodness ka reward keval goodness hee hoga."},
                {"role": "user", "content": f"{tgt_sent}\n words wanted: {constraint}"},
            ]
        )
    return src_prompts, tgt_prompts
    
def get_outputs(input_list, pipeline, terminators):
    print(f"Generating... input_list={pprint.pprint(input_list)}")
    result = []
    outputs_list = pipeline(
        input_list,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    for i, outputs in enumerate(outputs_list):
        result.append(outputs[0]["generated_text"][-1]['content'])
    return result

def main(args):
    
    lang_dict = {
        'en': 'English',
        'hi': 'Hindi',
        'ml': 'Malayalam',
        'ta': 'Tamil',
        'id': 'Indonesian',
        'zh': 'Chinese',
        'sge': 'Singlish',
    }
    example_dict = {
        'en': 'The reward of goodness shall be nothing but goodness',
        'hi': 'अच्छाई का पुरस्कार अच्छाई के अलावा कुछ नहीं होग',
        'ml': 'നന്മയുടെ പ്രതിഫലം നന്മയല്ലാതെ മറ്റൊന്നുമല്ലा',
        'ta': 'நன்மையின் வெகுமதி நன்மையைத் தவிர வேறில்லை',
        'id': 'Pahala kebaikan tak lain hanyalah kebaikan',
        'zh': '善的回报只能是善',
        'sge': 'The reward for doing good is nothing but more good',
    }
    example_cs_dict = {
        'en': 'The reward of goodness shall be nothing but goodness',
        'hi': 'goodness ka reward keval goodness hee hoga.',
        'ml': 'goodnessinte reward mattoru goodness mathramayirikkum',
        'ta': 'The reward of goodness yadharthamaana nalladhu dhaan irukkum.',
        'id': 'The reward dari kebaikan shall be nothing but kebaikan.',
        'zh': '善的回报shall be nothing but善',
        'sge': 'The reward for doing good is nothing but more good',
    }
    example_words_dict = {
        'en': ['reward'],
        'hi': ['होगा'],
        'ml': ['മറ്റൊ'],
        'ta': ['நல்லது'],
        'id': ['kebaikan'],
        'zh': ['只能是'],
        'sge': ['reward'],
    }
    
    lang1 = args.lang1
    lang2 = args.lang2
    example_l1 = example_dict[lang1]
    example_l2 = example_dict[lang2]
    example_cs = example_cs_dict[lang2]
    example_words_l1 = example_words_dict[lang1]
    example_words_l2 = example_words_dict[lang2]
    lang1 = lang_dict[lang1]
    lang2 = lang_dict[lang2]
    
    model_id = args.model_id

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
    ]
    if args.eot_token:
        terminators.append(
            pipeline.tokenizer.convert_tokens_to_ids(args.eot_token) 
        )
    
    if args.src != '':
        src = read_file(args.src)
    else:
        src = None
        
    if args.tgt != '':
        tgt = read_file(args.tgt)
    else:
        tgt = None
        
    if args.src_translated != '':
        src_translated = read_file(args.src_translated)
    else:
        src_translated = None
        
    if args.tgt_translated != '':
        tgt_translated = read_file(args.tgt_translated)
    else:
        tgt_translated = None
    
    if args.gold_align != '':
        gold_align = read_file(args.gold_align)
    else:
        gold_align = None
    
    if args.silver_src_align != '':
        silver_src_align = read_file(args.silver_src_align)
    else:
        silver_src_align = None
        
    if args.silver_tgt_align != '':
        silver_tgt_align = read_file(args.silver_tgt_align)
    else:
        silver_tgt_align = None
        
    if args.human_reference != '':
        human_reference = read_pickle(args.human_reference)
    else:
        human_reference = None
    
    # No Parallel Data Needed
    baseline_input_src, baseline_input_tgt = create_baseline(src, tgt, 
                                                             lang1, lang2, 
                                                             example_l1, example_l2,
                                                             example_cs, 
                                                             example_words_l1, example_words_l2)
    
    # Silver Parallel Data Needed
    if src is not None and tgt_translated is not None and silver_src_align is not None:
        alignment_silver_src_translated, _ = create_alignment(src, tgt_translated, silver_src_align,
                                                              lang1, lang2, 
                                                              example_l1, example_l2,
                                                              example_cs, 
                                                              example_words_l1, example_words_l2)
    else:
        alignment_silver_src_translated = None
        
    if src_translated is not None and tgt is not None and silver_tgt_align is not None:
        _, alignment_silver_tgt_translated = create_alignment(src_translated, tgt, silver_tgt_align,
                                                              lang1, lang2, 
                                                              example_l1, example_l2,
                                                              example_cs, 
                                                              example_words_l1, example_words_l2)
    else:
        alignment_silver_tgt_translated = None
    
    if src is not None and tgt is not None:
    # Gold Parallel Data Needed
        alignment_gold_src, alignment_gold_tgt = create_alignment(src, tgt, gold_align,
                                                                  lang1, lang2, 
                                                                  example_l1, example_l2,
                                                                  example_cs, 
                                                                  example_words_l1, example_words_l2)
    else:
        alignment_gold_src = None
        alignment_gold_tgt = None
        
    
    if src is not None and tgt is not None and human_reference is not None:
    # Data Leak (peek at evaluation set)
        ground_truth_src, ground_truth_tgt = create_data_leak(src, tgt, human_reference,
                                                              lang1, lang2)
    else:
        ground_truth_src = None
        ground_truth_tgt = None
    
    data = {
        'src': src, 
        'tgt': tgt, 
        'src_translated': src_translated,
        'tgt_translated': tgt_translated,
    }
    for key in list(data.keys()):
        if not data[key]:
            del data[key]
    
    print("Baseline")
    if baseline_input_src:
        baseline_src = get_outputs(baseline_input_src, pipeline, terminators)
        # baseline_src = None
        data['baseline_src'] = baseline_src
    else:
        baseline_src = None
    pd.DataFrame(data).to_csv(args.output, index=False)
        
    if baseline_input_tgt:
        baseline_tgt = get_outputs(baseline_input_tgt, pipeline, terminators)
        # baseline_tgt = None
        data['baseline_tgt'] = baseline_tgt
    else:
        baseline_tgt = None
    pd.DataFrame(data).to_csv(args.output, index=False)
    
    print("Silver")
    if alignment_silver_src_translated:
        silver_src = get_outputs(alignment_silver_src_translated, pipeline, terminators)
        data['silver_src'] = silver_src
    else:
        silver_src = None
    pd.DataFrame(data).to_csv(args.output, index=False)
        
    if alignment_silver_tgt_translated:
        silver_tgt = get_outputs(alignment_silver_tgt_translated, pipeline, terminators)
        data['silver_tgt'] = silver_tgt
    else:
        silver_tgt = None
    pd.DataFrame(data).to_csv(args.output, index=False)

    print("Gold")
    if alignment_gold_src:
        gold_src = get_outputs(alignment_gold_src, pipeline, terminators)
        data['gold_src'] = gold_src
    else:
        gold_src = None
    pd.DataFrame(data).to_csv(args.output, index=False)
        
    if alignment_gold_tgt:
        gold_tgt = get_outputs(alignment_gold_tgt, pipeline, terminators)
        data['gold_tgt'] = gold_tgt
    else:
        gold_tgt = None
    pd.DataFrame(data).to_csv(args.output, index=False)
    
    print("Peek Eval")
    if ground_truth_src:
        gt_src = get_outputs(ground_truth_src, pipeline, terminators)
        data['gt_src'] = gt_src
    else:
        gt_src = None
    pd.DataFrame(data).to_csv(args.output, index=False)
        
    if ground_truth_tgt:
        gt_tgt = get_outputs(ground_truth_tgt, pipeline, terminators)
        data['gt_tgt'] = gt_tgt
    else:
        gt_tgt = None
        
    pd.DataFrame(data).to_csv(args.output, index=False)
    
    
    
    
if __name__ == "__main__":
    args = init()
    main(args)
