echo -e "\033[1;38;5;22mStage 1: Translation\033[0m"

# python src/translate.py \
#     --src sge \
#     --tgt zh \
#     --input lithops/singlish_data/train.sge \
#     --output lithops/singlish_data/train.translated.zh \
#     --model_id aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct

echo -e "\033[1;38;5;22mStage 1.1: Chinese Tokenization\033[0m"
# Tokenize Chinese with jieba
# python src/tokenize_zh.py \
#     --input lithops/singlish_data/train.translated.zh \
#     --output lithops/singlish_data/train.translated.tokenized.zh

echo -e "\033[1;38;5;22mStage 2: Alignment\033[0m"
# Generate Alignment Files from silver translations
# python alignment/giza-py/giza.py \
#     --source lithops/singlish_data/train.sge \
#     --target lithops/singlish_data/train.translated.tokenized.zh \
#     --alignments lithops/singlish_data/train.sge-translated-zh.align

echo -e "\033[1;38;5;22mStage 3: CSW Generation\033[0m"
python src/inference.py \
    --lang1 sge \
    --lang2 zh \
    --src lithops/singlish_data/train.sge \
    --tgt_translated lithops/singlish_data/train.translated.tokenized.zh \
    --silver_src_align lithops/singlish_data/train.sge-translated-zh.align \
    --model_id "SeaLLMs/SeaLLMs-v3-7B-Chat" \
    --output lithops/singlish_data/output/full_SeaLLMs_SeaLLMs-v3-7B-Chat.csv
    # --model_id "aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct" \
    # --output lithops/singlish_data/output/full_sea-lionv2-1.csv
    # --model_id "aisingapore/sea-lion-7b-instruct" \
    # --output lithops/singlish_data/output/full_aisingapore_sea-lion-7b-instruct.csv