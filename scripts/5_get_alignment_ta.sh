# Generate Alignment Files from full dataset
python alignment/giza-py/giza.py --source data/samanantar_full/en-ta.en --target data/samanantar_full/en-ta.ta --alignments output/samanantar_full.en-ta.align
python alignment/giza-py/giza.py --source data/samanantar_full/en-ta.en --target data/samanantar_full/en-ta.translated_llama.ta --alignments output/samanantar_full.en-translated_ta.align
python alignment/giza-py/giza.py --source data/samanantar_full/en-ta.translated_llama.en --target data/samanantar_full/en-ta.ta --alignments output/samanantar_full.translated_en-ta.align

# Generate Alignment Files from subset dataset
python alignment/giza-py/giza.py --source data/samanantar/en-ta.en --target data/samanantar/en-ta.ta --alignments output/samanantar.en-ta.align
python alignment/giza-py/giza.py --source data/samanantar/en-ta.en --target data/samanantar/en-ta.translated_llama.ta --alignments output/samanantar.en-translated_ta.align
python alignment/giza-py/giza.py --source data/samanantar/en-ta.translated_llama.en --target data/samanantar/en-ta.ta --alignments output/samanantar.translated_en-ta.align