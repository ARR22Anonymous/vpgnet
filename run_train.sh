#!/bin/bash

set -u
set -e

gpu_ids=${1-2}
cate=${2-home_appliances}
task_type=${3-vpg}
task_tag=${4-basic}
root_dir=${5-/data/xxx/jdsum/}
restore_pretrain=${6-kplug}
lr=${7-0.0005}
add_rel_margin=${8-0}
rel_margin=${9-1.0}
validate_interval=${10-1}
patch_num=${11-49}
patch_embed_size=${12-2048}
patience=${13-6}
max_tokens=${14-8192}
max_epoch=${15-240}
save_interval=${16-1}
keep_best_checkpoints=${17-3}

export CUDA_VISIBLE_DEVICES=${gpu_ids}

CATEGORY=${cate}  # fushi xiangbao
TASK_TYPE=${task_type} # matchgo kplug  tpg  vpg
ROOT_DATA_DIR=${root_dir}

DATA_DIR=${ROOT_DATA_DIR}/${CATEGORY}/bin_mg
MODEL_DIR=${ROOT_DATA_DIR}/${CATEGORY}/checkpoints_${CATEGORY}_${TASK_TYPE}_${task_tag}/
if [ "$restore_pretrain" = "kplug" ]; then
  RESTORE_MODEL=${ROOT_DATA_DIR}/model_pretrain/checkpoint72.pt
else
  RESTORE_MODEL=${MODEL_DIR}/checkpoint_last.pt
fi


nohup fairseq-train ${DATA_DIR} \
    --user-dir model \
    --task matchgo \
    --arch transformer_kplug_base \
    --bertdict \
    --sku2vec-path ${DATA_DIR}/../raw/ \
    --img2ids-path ${DATA_DIR}/../raw/ \
    --img2vec-path ${DATA_DIR}/../image_patch_vectors_img_merge.npy \
    --reset-optimizer --reset-dataloader --reset-meters \
    --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 \
    --lr ${lr} --stop-min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy_with_margin --label-smoothing 0.1 \
    --update-freq 8 --max-tokens ${max_tokens} \
    --ddp-backend=no_c10d --max-epoch ${max_epoch} \
    --max-source-positions 512 --max-target-positions 512 \
    --truncate-source \
    --restore-file ${RESTORE_MODEL} \
    --save-dir ${MODEL_DIR} \
    --task_type ${TASK_TYPE} \
    --patience ${patience} \
    --validate-interval ${validate_interval} \
    --save-interval	 ${save_interval} \
    --patch_num ${patch_num} \
    --patch_embed_size ${patch_embed_size} \
    --add_rel_margin ${add_rel_margin} \
    --rel_margin ${rel_margin} \
    --skip-invalid-size-inputs-valid-test	\
    --keep-best-checkpoints	${keep_best_checkpoints} \
    > ${ROOT_DATA_DIR}/${CATEGORY}/log_${CATEGORY}_${TASK_TYPE}_${task_tag}.train 2>&1 &

sleep 1s
tail -f ${ROOT_DATA_DIR}/${CATEGORY}/log_${CATEGORY}_${TASK_TYPE}_${task_tag}.train