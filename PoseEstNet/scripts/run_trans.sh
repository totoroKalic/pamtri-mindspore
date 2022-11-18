echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_trans.sh DATA_PATH CKPT_PATH DEVICE_ID"
echo "For example: bash run_eval.sh /path/dataset /path/ckpt 0"
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
CKPT_PATH=$(get_real_path $2)

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
export DEVICE_ID=$3
export RANK_SIZE=1
env > env.log
python3.7 trans.py --cfg config.yaml --data_dir ${DATA_PATH} --ckpt_path ${CKPT_PATH} > trans.log 2>&1

if [ $? -eq 0 ];then
    echo "trans success"
else
    echo "trans failed"
    exit 2
fi
echo "finish"
cd ../
