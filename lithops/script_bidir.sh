STAGE_1=0
STAGE_1_1=0
STAGE_2=0
STAGE_3=1
STAGE_3_1=0
STAGE_4=0
STAGE_5=0

DATA_DIR=lithops/singlish_data

# Stage 1: TRanslation
if [ "$STAGE_1" -eq 1 ]; then
    echo -e "\033[1;38;5;22mStage 1: Translation\033[0m"
    # SgE to Chinese
    python src/translate.py \
        --src sge \
        --tgt zh \
        --input $DATA_DIR/train.sge \
        --output $DATA_DIR/train.translated.zh \
        --model_id aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct
    # Chinese to SgE
    python src/translate.py \
        --src zh \
        --tgt sge \
        --input $DATA_DIR/train.zh \
        --output $DATA_DIR/train.translated.sge \
        --model_id aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct   
fi

# Stage 1.1: Chinese Tokenization
if [ "$STAGE_1_1" -eq 1 ]; then
    echo -e "\033[1;38;5;22mStage 1.1: Chinese Tokenization\033[0m"
    #Tokenize Chinese with jieba
    python src/tokenize_zh.py \
        --input $DATA_DIR/train.zh \
        --output $DATA_DIR/train.tokenized.zh
    
    #Tokenize translated Chinese with jieba
    python src/tokenize_zh.py \
        --input $DATA_DIR/train.translated.zh \
        --output $DATA_DIR/train.translated.tokenized.zh
fi

# Stage 2: Alignment
if [ "$STAGE_2" -eq 1 ]; then
    echo -e "\033[1;38;5;22mStage 2: Alignment\033[0m"
    #Generate Alignment Files from gold translations
    python alignment/giza-py/giza.py \
        --source $DATA_DIR/train.sge \
        --target $DATA_DIR/train.tokenized.zh \
        --alignments $DATA_DIR/train.sge-zh.align
        
    #Generate Alignment Files from silver translations
    python alignment/giza-py/giza.py \
        --source $DATA_DIR/train.sge \
        --target $DATA_DIR/train.translated.tokenized.zh \
        --alignments $DATA_DIR/train.sge-translated-zh.align
    python alignment/giza-py/giza.py \
        --source $DATA_DIR/train.translated.sge \
        --target $DATA_DIR/train.tokenized.zh \
        --alignments $DATA_DIR/train.translated-sge-zh.align
fi

# Stage 3: CSW Generation
if [ "$STAGE_3" -eq 1 ]; then
    echo -e "\033[1;38;5;22mStage 3: CSW Generation\033[0m"
    python src/inference.py \
        --lang1 sge \
        --lang2 zh \
        --src $DATA_DIR/train.sge \
        --tgt $DATA_DIR/train.tokenized.zh \
        --src_translated $DATA_DIR/train.translated.sge \
        --tgt_translated $DATA_DIR/train.translated.tokenized.zh \
        --gold_align $DATA_DIR/train.sge-zh.align \
        --silver_src_align $DATA_DIR/train.sge-translated-zh.align \
        --silver_tgt_align $DATA_DIR/train.translated-sge-zh.align \
        --model_id "SeaLLMs/SeaLLMs-v3-7B" \
        --output $DATA_DIR/output/full_SeaLLMs_SeaLLMs-v3-7B-non-romanized.csv
        # --model_id "SeaLLMs/SeaLLM-7B-v2.5" \
        # --output $DATA_DIR/output/full_SeaLLMs_SeaLLM-7B-v2.5.csv
        # --model_id "aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct" \
        # --output $DATA_DIR/output/full_sea-lionv2-1.csv
        # --model_id "aisingapore/sea-lion-7b-instruct" \
        # --output $DATA_DIR/output/full_aisingapore_sea-lion-7b-instruct.csv
fi

# Stage 3.1: Romanize Chinese
if [ "$STAGE_3_1" -eq 1 ]; then
    echo -e "\033[1;38;5;22mStage 3.1: Romanize Chinese\033[0m"
    python src/romanize_zh.py \
    --input $DATA_DIR/output/full_SeaLLMs_SeaLLMs-v3-7B.csv \
    --output $DATA_DIR/output/full_SeaLLMs_SeaLLMs-v3-7B_tmp.csv \
    --exclude_cols src tgt src_translated tgt_translated
fi

# Stage 4: Compile Output
if [ "$STAGE_4" -eq 1 ]; then
    echo -e "\033[1;38;5;22mStage 4: Compile Output\033[0m"
    python src/compile.py \
        --directory $DATA_DIR/output \
        --output $DATA_DIR/output/compile_sge-zh.csv
fi

# Stage 5: Evaluation
if [ "$STAGE_5" -eq 1 ]; then
    echo -e "\033[1;38;5;22mStage 5: Evaluation\033[0m"
    python src/evaluate_generation.py \
        --input_file $DATA_DIR/output/compile_sge-zh.csv \
        --lang en \
        --output_file $DATA_DIR/output/evaluation_sge-zh.csv \
        --openai_key "$OPENAI_API_KEY" \
        --openai_org "$OPENAI_ORG_KEY" \
        --batch
fi
