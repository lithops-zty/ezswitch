python src/inference.py \
    --lang1 en \
    --lang2 ta \
    --src data/samanantar_full/en-ta.en \
    --tgt data/samanantar_full/en-ta.ta \
    --src_translated data/samanantar_full/en-ta.translated_llama.en \
    --tgt_translated data/samanantar_full/en-ta.translated_llama.ta\
    --gold_align output/samanantar_full.en-ta.align \
    --silver_src_align output/samanantar_full.en-translated_ta.align \
    --silver_tgt_align output/samanantar_full.translated_en-ta.align \
    --model_id "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --output output/tamil/full_llama31.csv
    
# --model_id "meta-llama/Meta-Llama-3-8B-Instruct"
# --model_id "CohereForAI/aya-23-8B"