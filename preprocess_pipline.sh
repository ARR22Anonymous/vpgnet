#!/bin/bash

set -u
set -b

VOCAB_FILE=${1-/data/xxx/jdsum/vocab/vocab.jd.txt}
RAW_DIR=${2-/data/xxx/jddc_v2/raw/}
BPE_DIR=${3-/data/xxx/jddc_v2/bpe_mg_batch/}
BIN_DIR=${4-/data/xxx/jddc_v2/bin_mg_batch/}
SPLIT=${5-valid}

doMakeDir() {
  dir_path=$1
  if [ ! -d $dir_path ]; then
    mkdir -p $dir_path
  fi
}

doMakeDir ${BPE_DIR}

# Tokenize
begin=$(date +%s)
echo "[Tokenize Dataset Starts]"
split=${SPLIT}
for lang in src tgt; do
  python data_process/multiprocessing_bpe_encoder.py \
    --vocab-bpe ${VOCAB_FILE} \
    --inputs ${RAW_DIR}/${split}.${lang} \
    --outputs ${BPE_DIR}/${split}.${lang} \
    --workers 60
  cp -r ${RAW_DIR}/${split}.sku ${BPE_DIR}/${split}.sku
done
end=$(date +%s)
spend=$(expr $end - $begin)
echo "[Tokenize Dataset Ends] Consumes $spend S"

# Binarized data
begin=$(date +%s)
echo "[Binarized Dataset Starts]"

if [ ${split} == "train" ]; then
  destdir=${BIN_DIR}

  fairseq-preprocess \
  --user-dir model \
  --task translation \
  --bertdict \
  --source-lang src \
  --target-lang tgt \
  --srcdict ${VOCAB_FILE} \
  --tgtdict ${VOCAB_FILE} \
  --trainpref ${BPE_DIR}/${split} \
  --destdir  ${destdir} \
  --workers 20


  rm -rf ${BIN_DIR}/valid.*
  ln -s ${BIN_DIR}/../valid/valid.src-tgt.src.bin ${BIN_DIR}/valid.src-tgt.src.bin
  ln -s ${BIN_DIR}/../valid/valid.src-tgt.src.idx ${BIN_DIR}/valid.src-tgt.src.idx
  ln -s ${BIN_DIR}/../valid/valid.src-tgt.tgt.bin ${BIN_DIR}/valid.src-tgt.tgt.bin
  ln -s ${BIN_DIR}/../valid/valid.src-tgt.tgt.idx ${BIN_DIR}/valid.src-tgt.tgt.idx

elif [ ${split} == "test" ]; then
  destdir=${BIN_DIR}/${split}

  fairseq-preprocess \
  --user-dir model \
  --task translation \
  --bertdict \
  --source-lang src \
  --target-lang tgt \
  --srcdict ${VOCAB_FILE} \
  --tgtdict ${VOCAB_FILE} \
  --testpref ${BPE_DIR}/${split} \
  --destdir  ${destdir} \
  --workers 20


else
  destdir=${BIN_DIR}/${split}

  fairseq-preprocess \
  --user-dir model \
  --task translation \
  --bertdict \
  --source-lang src \
  --target-lang tgt \
  --srcdict ${VOCAB_FILE} \
  --tgtdict ${VOCAB_FILE} \
  --validpref ${BPE_DIR}/${split} \
  --destdir  ${destdir} \
  --workers 20

fi


echo "[Binarized Dataset Ends] Consumes $spend S"
