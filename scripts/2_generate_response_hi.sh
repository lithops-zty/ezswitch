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

# --model_id "meta-llama/Meta-Llama-3-8B-Instruct"
# --model_id "CohereForAI/aya-23-8B"
