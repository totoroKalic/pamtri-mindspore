
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train.sh DATASET_PATH PRETRAIN_CKPT_PATH RANK_TABLE_FILE HEATMAP_SEGMENT"
echo "For example: bash run_distribute_train.sh ../data/ ./MultiTask_pretrained.ckpt ./rank_table_8pcs.json h"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATA_PATH=$(get_real_path $1)
PRETRAINED_PATH=$(get_real_path $2)
RANK_TABLE=$(get_real_path $3)
export DATA_PATH=${DATA_PATH}
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_TABLE

if [ "$4" == "h" ] || [ "$4" == "s" ];then
    if [ "$4" == "h" ];then
      need_heatmap=True
      need_segment=False
    else
      need_heatmap=False
      need_segment=True
    fi
else
    echo "heatmap_segment must be h or s"
    exit 1
fi

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    cp ../train.py ./
    cp -rf ../src ./
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    nohup python3.7 -u train.py --distribute True --pre_ckpt_path ${PRETRAINED_PATH} --root ${DATA_PATH} --heatmapaware ${need_heatmap} --segmentaware ${need_segment} > train$i.log 2>&1 &
    echo "$i finish"
    cd ../
done

rm -rf device0
mkdir device0
cd ./device0
cp ../train.py ./
cp -rf ../src ./

export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
nohup python3.7 -u train.py --distribute True --pre_ckpt_path ${PRETRAINED_PATH} --root ${DATA_PATH} --heatmapaware ${need_heatmap} --segmentaware ${need_segment} > train0.log 2>&1 &
echo "0 finish"
cd ../

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
