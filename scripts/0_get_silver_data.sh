# Translate using LLAMA3
## English to Hindi
python src/translate.py --src en --tgt hi --input data/hinge/train.en --output data/hinge/train.translated.hi
python src/translate.py --src en --tgt hi --input data/hinge/valid.en --output data/hinge/valid.translated.hi

## Hindi to English
python src/translate.py --src hi --tgt en --input data/hinge/train.hi --output data/hinge/train.translated.en
python src/translate.py --src hi --tgt en --input data/hinge/valid.hi --output data/hinge/valid.translated.en

## English to Tamil
python src/translate.py --src en --tgt ta --input data/samanantar_full/en-ta.en --output data/samanantar_full/en-ta.translated_llama.ta
python src/translate.py --src en --tgt ta --input data/samanantar/en-ta.en --output data/samanantar/en-ta.translated_llama.ta

## Tamil to English
python src/translate.py --src ta --tgt en --input data/samanantar_full/en-ta.ta --output data/samanantar_full/en-ta.translated_llama.en
python src/translate.py --src ta --tgt en --input data/samanantar/en-ta.ta --output data/samanantar/en-ta.translated_llama.en


## English to Malayalam
python src/translate.py --src en --tgt ml --input data/samanantar_full/en-ml.en --output data/samanantar_full/en-ml.translated_llama.ml
python src/translate.py --src en --tgt ml --input data/samanantar/en-ml.en --output data/samanantar/en-ml.translated_llama.ml

## Malayalam to English
python src/translate.py --src ml --tgt en --input data/samanantar_full/en-ml.ml --output data/samanantar_full/en-ml.translated_llama.en
python src/translate.py --src ml --tgt en --input data/samanantar/en-ml.ml --output data/samanantar/en-ml.translated_llama.en