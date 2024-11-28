python src/translate.py \
    --src sge \
    --tgt zh \
    --input lithops/singlish_data/train.sge \
    --output lithops/singlish_data/train.translated.zh

# Tokenize Chinese with jieba
python src/tokenize_zh.py \
    --input lithops/singlish_data/train.translated.zh \
    --output lithops/singlish_data/train.translated.tokenized.zh

# Generate Alignment Files from silver translations
python alignment/giza-py/giza.py \
    --source lithops/singlish_data/train.sge \
    --target lithops/singlish_data/train.translated.tokenized.zh \
    --alignments lithops/singlish_data/train.sge-translated-zh.align

python src/inference.py \
    --lang1 sge \
    --lang2 zh \
    --src lithops/singlish_data/train.sge \
    --tgt_translated lithops/singlish_data/train.translated.tokenized.zh \
    --silver_src_align lithops/singlish_data/train.sge-translated-zh.align \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --output lithops/singlish_data/output/full_llama3_2.csv