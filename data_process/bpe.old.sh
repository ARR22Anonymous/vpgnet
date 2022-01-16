DATA_DIR=data

idx=$1
echo ${idx}

# Tokenize
for split in train dev test; do
  for lang in source_kb_val target; do
    python multiprocessing_bpe_encoder.py \
      --vocab-bpe ${DATA_DIR}/vocab.jd.txt \
      --inputs  ${DATA_DIR}/result${idx}/${split}.${lang} \
      --outputs ${DATA_DIR}/bpe_src_kb${idx}/${split}.${lang} \
      --workers 60
  done
done

