CATEGORY=jiadian  # fushi xiangbao
DATA_DIR=xxx

# Tokenize
python multiprocessing_bpe_encoder.py \
  --vocab-bpe data/vocab.jd.txt \
  --inputs  test.source \
  --outputs bpe/test.source \
  --workers 60
