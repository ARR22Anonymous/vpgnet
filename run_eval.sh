#!/bin/bash

set -u
set -e

out_dir=${1-res}
cate=${2-home_appliances}
task_type=${3-vpg}
task_tag=${4-basic}
model_name=${5-checkpoint_best}
ROOT_DATA_DIR=${6-/data/xxx/jdsum/}

CATEGORY=${cate}  # fushi xiangbao
TASK_TYPE=${task_type} # matchgo kplug  tpg  vpg

file_type=${CATEGORY}_${TASK_TYPE}_${task_tag}_${model_name}
data_dir=${ROOT_DATA_DIR}/${CATEGORY}

grep ^T ${data_dir}/${file_type}.output.res | cut -f2- | sed 's/ ##//g' > ${data_dir}/${out_dir}/${file_type}_tgt.txt
grep ^H ${data_dir}/${file_type}.output.res | cut -f3- | sed 's/ ##//g' > ${data_dir}/${out_dir}/${file_type}_hypo.txt

python ../tools/get_rouge.py ${data_dir}/${out_dir}/${file_type}_tgt.txt ${data_dir}/${out_dir}/${file_type}_hypo.txt | tee ${data_dir}/${out_dir}/${file_type}.res
