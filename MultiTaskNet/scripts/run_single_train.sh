
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_single_train.sh DATA_PATH PRETRAINED_PATH DEVICE_ID HEATMAP_SEGMENT"
echo "For example: bash run_single_train.sh /path/dataset /path/pretrained_path 0 s"
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
PRE_CKPT_PATH=$(get_real_path $2)

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
export DEVICE_ID=$3
export RANK_SIZE=1
env > env.log
python3 train.py --pre_ckpt_path  ${PRE_CKPT_PATH} --root ${DATA_PATH} --distribute False --heatmapaware ${need_heatmap} --segmentaware ${need_segment} > train.log 2>&1

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
