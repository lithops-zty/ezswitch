python src/inference.py \
    --src data/hinge/train.en \
    --tgt data/hinge/train.hi \
    --src_translated data/hinge/train.translated_llama.en \
    --tgt_translated data/hinge/train.translated_llama.hi \
    --gold_align output/train.en-hi.align \
    --silver_src_align output/train.en-translated_llama-hi.align \
    --silver_tgt_align output/train.translated_llama-en-hi.align \
    --human_reference data/hinge/train_human_generated.pkl \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --output output/hindi/full_llama3_1.csv

# --model_id "meta-llama/Meta-Llama-3-8B-Instruct"
# --model_id "CohereForAI/aya-23-8B"
