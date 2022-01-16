#!/bin/bash

set -u
set -e

gpu_ids=${1-1}
cate=${2-home_appliances}
task_type=${3-vpg}
task_tag=${4-basic}
root_dir=${5-/data/xxx/jdsum/}
model_name=${6-checkpoint_best}


export CUDA_VISIBLE_DEVICES=${gpu_ids}

CATEGORY=${cate}  # fushi xiangbao
TASK_TYPE=${task_type} # matchgo kplug  tpg  vpg
ROOT_DATA_DIR=${root_dir}

DATA_DIR=${ROOT_DATA_DIR}/${CATEGORY}/bin_mg
MODEL_DIR=${ROOT_DATA_DIR}/${CATEGORY}/checkpoints_${CATEGORY}_${TASK_TYPE}_${task_tag}/
INFER_MODEL=${MODEL_DIR}/${model_name}.pt

nohup fairseq-generate ${DATA_DIR} \
    --path ${INFER_MODEL} \
    --user-dir model \
    --task matchgo \
    --bertdict \
    --sku2vec-path ${DATA_DIR}/../raw/ \
    --img2ids-path ${DATA_DIR}/../raw/ \
    --img2vec-path ${DATA_DIR}/../image_patch_vectors_img_merge.npy \
    --max-source-positions 512 --max-target-positions 512 \
    --batch-size 64 \
    --beam 5 \
    --min-len 50 \
    --truncate-source \
    --no-repeat-ngram-size 3 \
    --task_type ${TASK_TYPE} \
    > ${ROOT_DATA_DIR}/${CATEGORY}/${CATEGORY}_${TASK_TYPE}_${task_tag}_${model_name}.output.res 2>&1 &

tail -f ${ROOT_DATA_DIR}/${CATEGORY}/${CATEGORY}_${TASK_TYPE}_${task_tag}_${model_name}.output.res