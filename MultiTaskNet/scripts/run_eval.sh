
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval.sh DATASET_NAME CKPT_PATH DEVICE_ID HEATMAP_SEGMENT"
echo "For example: bash run_eval.sh ./*.ckpt ./data/ 0 h"
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
export RANK_SIZE=1
env > env.log
python eval.py --root ${DATA_PATH} --ckpt_path ${CKPT_PATH} --device_id $3 --heatmapaware ${need_heatmap} --segmentaware ${need_segment} > eval.log 2>&1

if [ $? -eq 0 ];then
    echo "eval success"
else
    echo "eval failed"
    exit 2
fi
echo "finish"
cd ../
