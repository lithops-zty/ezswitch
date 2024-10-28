python src/evaluate_generation.py \
    --generation_file output/compile_hindi.csv \
    --lang hi \
    --output_file output/evaluation_hindi.csv \
    --openai_key $OPENAI_API_KEY \
    --openai_org $OPENAI_ORG_KEY

python src/evaluate_generation.py \
    --generation_file output/compile_malayalam.csv \
    --lang ml \
    --output_file output/evaluation_malayalam.csv \
    --openai_key $OPENAI_API_KEY \
    --openai_org $OPENAI_ORG_KEY

python src/evaluate_generation.py \
    --generation_file output/compile_tamil.csv \
    --lang ta \
    --output_file output/evaluation_tamil.csv \
    --openai_key $OPENAI_API_KEY \
    --openai_org $OPENAI_ORG_KEY
