echo -e "\033[1;38;5;22mStage 1: Translation\033[0m"

python src/translate.py \
    --src sge \
    --tgt zh \
    --input lithops/singlish_data/train.sge \
    --output lithops/singlish_data/train.translated.zh

echo -e "\033[1;38;5;22mStage 1.1: Chinese Tokenization\033[0m"
# Tokenize Chinese with jieba
python src/tokenize_zh.py \
    --input lithops/singlish_data/train.translated.zh \
    --output lithops/singlish_data/train.translated.tokenized.zh

echo -e "\033[1;38;5;22mStage 2: Alignment\033[0m"
# Generate Alignment Files from silver translations
python alignment/giza-py/giza.py \
    --source lithops/singlish_data/train.sge \
    --target lithops/singlish_data/train.translated.tokenized.zh \
    --alignments lithops/singlish_data/train.sge-translated-zh.align

echo -e "\033[1;38;5;22mStage 1: CSW Generation\033[0m"
python src/inference.py \
    --lang1 sge \
    --lang2 zh \
    --src lithops/singlish_data/train.sge \
    --tgt_translated lithops/singlish_data/train.translated.tokenized.zh \
    --silver_src_align lithops/singlish_data/train.sge-translated-zh.align \
    --model_id "meta-llama/Llama-3.2-1B-Instruct" \
    --output lithops/singlish_data/output/full_llama3_2.csv