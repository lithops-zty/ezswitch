# Generate Alignment Files from gold translations
python alignment/giza-py/giza.py --source data/hinge/train.en --target data/hinge/train.hi --alignments output/train.en-hi.align
python alignment/giza-py/giza.py --source data/hinge/valid.en --target data/hinge/valid.hi --alignments output/valid.en-hi.align

# Generate Alignment Files from silver translations
python alignment/giza-py/giza.py --source data/hinge/train.en --target data/hinge/train.translated.hi --alignments output/train.en-translated-hi.align
python alignment/giza-py/giza.py --source data/hinge/valid.en --target data/hinge/valid.translated.hi --alignments output/valid.en-translated-hi.align

python alignment/giza-py/giza.py --source data/hinge/train.translated.en --target data/hinge/train.hi --alignments output/train.translated-en-hi.align
python alignment/giza-py/giza.py --source data/hinge/valid.translated.en --target data/hinge/valid.hi --alignments output/valid.translated-en-hi.align

# Generate Alignment Files from silver translations # LLAMA
python alignment/giza-py/giza.py --source data/hinge/valid.en --target data/hinge/valid.translated_llama.hi --alignments output/valid.en-translated_llama-hi.align

python alignment/giza-py/giza.py --source data/hinge/valid.translated_llama.en --target data/hinge/valid.hi --alignments output/valid.translated_llama-en-hi.align
