
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_single_train.sh DATA_PATH PRETRAINED_PATH DEVICE_ID"
echo "For example: bash run_single_train.sh /path/dataset /path/pretrained_path 0 "
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
PRE_CKPT=$(get_real_path $2)

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
export DEVICE_ID=$3
export RANK_SIZE=1
env > env.log
python3 train.py --cfg config.yaml --data_dir ${DATA_PATH} --distribute False --pre_ckpt_path ${PRE_CKPT} > train.log 2>&1

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
