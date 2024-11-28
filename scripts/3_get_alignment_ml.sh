# Generate Alignment Files for full dataset
python alignment/giza-py/giza.py --source data/samanantar_full/en-ml.en --target data/samanantar_full/en-ml.ml --alignments output/samanantar_full.en-ml.align
python alignment/giza-py/giza.py --source data/samanantar_full/en-ml.en --target data/samanantar_full/en-ml.translated_llama.ml --alignments output/samanantar_full.en-translated_ml.align
python alignment/giza-py/giza.py --source data/samanantar_full/en-ml.translated_llama.en --target data/samanantar_full/en-ml.ml --alignments output/samanantar_full.translated_en-ml.align

# Generate Alignment Files for subset dataset
python alignment/giza-py/giza.py --source data/samanantar/en-ml.en --target data/samanantar/en-ml.ml --alignments output/samanantar.en-ml.align
python alignment/giza-py/giza.py --source data/samanantar/en-ml.en --target data/samanantar/en-ml.translated_llama.ml --alignments output/samanantar.en-translated_ml.align
python alignment/giza-py/giza.py --source data/samanantar/en-ml.translated_llama.en --target data/samanantar/en-ml.ml --alignments output/samanantar.translated_en-ml.align