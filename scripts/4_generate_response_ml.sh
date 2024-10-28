python src/inference.py \
    --lang1 en \
    --lang2 ml \
    --src data/samanantar_full/en-ml.en \
    --tgt data/samanantar_full/en-ml.ml \
    --src_translated data/samanantar_full/en-ml.translated_llama.en \
    --tgt_translated data/samanantar_full/en-ml.translated_llama.ml\
    --gold_align output/samanantar_full.en-ml.align \
    --silver_src_align output/samanantar_full.en-translated_ml.align \
    --silver_tgt_align output/samanantar_full.translated_en-ml.align \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --output output/malayalam/full_llama31.csv
    
# --model_id "meta-llama/Meta-Llama-3-8B-Instruct"
# --model_id "CohereForAI/aya-23-8B"