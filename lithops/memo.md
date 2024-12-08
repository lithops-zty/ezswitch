EZSwitch Usage
---

### Notation

| Notation         | Description                                 | Remarks                                       |
|------------------|---------------------------------------------|-----------------------------------------------|
| `src_lang`       | Source language                             | two-letter language code                      |
| `tgt_lang`       | Target language                             | two-letter language code                      |
| `src_input`      | Input sentences in `src_lang`               |                                               |
| `tgt_input`      | `src_input` human-translated  to `tgt_lang` | specified only when using gold translation?   |
| `src_translated` | `tgt_input` LLM-translated  to `src_lang`   | specified only when using SILVER translation? |
| `tgt_translated` | `src_input` LLM-translated to `tgt_lang`    | specified only when using SILVER translation? |


### 1. Translation Stage

#### SILVER Translation

##### Command
```shell
python src/translate.py <options>
```
| Options    | Description          | Default | Required |
|------------|----------------------|---------|----------|
| `--src`    | Source language Code | N/A     | ?        |
| `--tgt`    | Target language Code | N/A     | ?        |
| `--input`  | Path to input file   | N/A     | ?        |
| `--output` | Path to output file  | N/A     | ?        |

| File       | Description                                                   | format                         | I/O   |
|------------|---------------------------------------------------------------|--------------------------------|-------|
| `--input`  | File containing sentences in source language to be translated | Sentences separated by newline | Read  |
| `--output` | File containing translated sentences in target language       | Sentences separated by newline | Write |

##### Example
```shell
## English to Hindi
python src/translate.py --src en --tgt hi --input data/hinge/train.en --output data/hinge/train.translated.hi
python src/translate.py --src en --tgt hi --input data/hinge/valid.en --output data/hinge/valid.translated.hi

## Hindi to English
python src/translate.py --src hi --tgt en --input data/hinge/train.hi --output data/hinge/train.translated.en
python src/translate.py --src hi --tgt en --input data/hinge/valid.hi --output data/hinge/valid.translated.en
```

### 2. Alignment Stage

#### Using SILVER Translation

##### Command
```shell
python alignment/giza-py/giza.py <options>
```

| Options        | Description                               | Default | Required |
|----------------|-------------------------------------------|---------|----------|
| `--source`     | Path to source language file              | N/A     | ?        |
| `--target`     | Path to target language file              | N/A     | ?        |
| `--alignments` | Path to output file containing alignments | N/A     | ?        |

| File           | Description                                                    | format                          | I/O   |
|----------------|----------------------------------------------------------------|---------------------------------|-------|
| `--source`     | File containing sentences in in source language, to be aligned | Sentences separated by newline  | Read  |
| `--target`     | File containing sentences in in target language, to be aligned | Sentences separated by newline  | Read  |
| `--alignments` | File containing alignments between source and target sentences | Alignments separated by newline | Write |

##### Example
```shell
python alignment/giza-py/giza.py --source data/hinge/train.en --target data/hinge/train.translated.hi --alignments output/train.en-translated-hi.align
python alignment/giza-py/giza.py --source data/hinge/valid.en --target data/hinge/valid.translated.hi --alignments output/valid.en-translated-hi.align
```

### 3. Generation Stage

##### Command
```shell
python src/generate.py <options>
```

------------------------------------------
| Options              | Description                             | Default | Required |
|----------------------|-----------------------------------------|---------|----------|
| `--lang1`            | Source language Code                    | 'en'    | ?        |
| `--lang2`            | Target language Code                    | 'hi'    | ?        |
| `--src`              | Path to source language file            | `''`    | ?        |
| `--tgt`              | Path to target language file            | `''`    | ?        |
| `--src_translated`   | Path to source language translated file | `''`    | ?        |
| `--tgt_translated`   | Path to target language translated file | `''`    | ?        |
| `--gold_align`       | Path to gold alignments file            | `''`    | ?        |
| `--silver_src_align` | Path to silver source alignments file   | `''`    | ?        |
| `--silver_tgt_align` | Path to silver target alignments file   | `''`    | ?        |
| `--human_reference`  | Path to human generated reference file  | `''`    | ?        |
| `--model_id`         | Model ID for generation                 | N/A     | ?        |
| `--output`           | Path to output file                     | N/A     | ?        |

| File                 | Description                                                            | format                          | I/O   |
|----------------------|------------------------------------------------------------------------|---------------------------------|-------|
| `--src`              | File containing `src_input`                                            | Sentences separated by newline  | Read  |
| `--tgt`              | File containing `src_output`                                           | Sentences separated by newline  | Read  |
| `--src_translated`   | File containing `src_translated`                                       | Sentences separated by newline  | Read  |
| `--tgt_translated`   | File containing `tgt_translated`                                       | Sentences separated by newline  | Read  |
| `--gold_align`       | File containing alignments between ``                                  | Alignments separated by newline | Read  |
| `--silver_src_align` | File containing silver alignments from `src_input` to `tgt_translated` | Alignments separated by newline | Read  |
| `--silver_tgt_align` | File containing silver alignments from `src_translated` to `tgt_input` | Alignments separated by newline | Read  |
| `--human_reference`  | ??                                                                     | Pickle file                     | Read  |
| `--output`           | File containing full input and generated sentences                     | Sentences separated by newline  | Write |
##### Example
```shell
python src/inference.py \
    --src data/hinge/train.en \
    --tgt data/hinge/train.hi \
    --src_translated data/hinge/train.translated.en \
    --tgt_translated data/hinge/train.translated.hi \
    --gold_align output/train.en-hi.align \
    --silver_src_align output/train.en-translated-hi.align \
    --silver_tgt_align output/train.translated-en-hi.align \
    --human_reference data/hinge/train_human_generated.pkl \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --output output/hindi/full_llama3_1.csv
```