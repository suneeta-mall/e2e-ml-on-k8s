#!/usr/bin/env bash

set -euox pipefail

function help_text {
    cat <<EOF
    Usage: $0 [ -o|--output OUTPUT_DIR ]
        OUTPUT_DIR    (optional) base output directory. If unspecified, CWD is assumed.
EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    arg=$1
    case ${arg} in
        -h|--help)
            help_text
        ;;
        -i|--output)
            export OUTPUT_DIR="$2"
            shift; shift
        ;;
        *)
            echo "ERROR: Unrecognised option: ${arg}"
            help_text
            exit 1
        ;;
    esac
done

if [[ -z "${OUTPUT_DIR+x}" ]]
    then
        echo "Output dir is not set, assuming current working directory ./pfs/ as as base output dir!"
        export OUTPUT_DIR="$PWD/pfs/"
fi

mkdir -p ${OUTPUT_DIR}
exec > >(tee -a -i ${OUTPUT_DIR}/output.log)
exec 2>&1

SRC_DIR=$(dirname "$0")


# Data downlad
python ${SRC_DIR}/app/download_petset.py --output ${OUTPUT_DIR}/warehouse
python ${SRC_DIR}/app/dataset_gen.py --input ${OUTPUT_DIR}/warehouse --output ${OUTPUT_DIR}/transform

python ${SRC_DIR}/app/train.py --model_arch MobileNetV2 --input ${OUTPUT_DIR}/transform \
   --output ${OUTPUT_DIR}/train \
   --checkpoint_path "${OUTPUT_DIR}/train/ckpts" \
   --tensorboard_path ${OUTPUT_DIR}/train

# python ${SRC_DIR}/app/viz_model.py --input ${OUTPUT_DIR}/transform --model_weight ${OUTPUT_DIR}/train/model.h5

python ${SRC_DIR}/app/tune.py --input ${OUTPUT_DIR}/transform --output ${OUTPUT_DIR}/tune --num_samples 2

python ${SRC_DIR}/app/train.py --model_arch MobileNetV2 --input ${OUTPUT_DIR}/transform --output ${OUTPUT_DIR}/model \
  --hyperparam_fn_path ${OUTPUT_DIR}/tune/optimal_hp.json \
  --checkpoint_path "${OUTPUT_DIR}/model/ckpts" \
  --tensorboard_path ${OUTPUT_DIR}/model

python ${SRC_DIR}/app/calibrate.py --input ${OUTPUT_DIR}/transform --model_weight ${OUTPUT_DIR}/model/model.h5 \
    --output ${OUTPUT_DIR}/calibrate

papermill evaluate.ipynb ${OUTPUT_DIR}/evaluate/Report.ipynb -p input_data_dir ${OUTPUT_DIR}/transform \
    -p model_weights "${OUTPUT_DIR}/calibrate/model.h5" \
    -p calibration_weights "${OUTPUT_DIR}/calibrate/calibration.weights"  \
    -p batch_size 64 -p hyperparameters ${OUTPUT_DIR}/tune/optimal_hp.json

#python ${SRC_DIR}/app/test.py --input ${OUTPUT_DIR}/transform/training/Abyssinian_2/image.jpg \
#   --output ${OUTPUT_DIR}/test/result.jpg

# These are still WIP
python ${SRC_DIR}/app/adversarial.py --model_weights "${OUTPUT_DIR}/model/model.h5"  --batch_size 64 \
    --input ${OUTPUT_DIR}/transform --output "${OUTPUT_DIR}/adversarial" --eps 0.000007

python ${SRC_DIR}/app/adversarial_train.py --input ${OUTPUT_DIR}/transform \
    --hyperparam_fn_path ${OUTPUT_DIR}/tune/optimal_hp.json \
    --model_weights "${OUTPUT_DIR}/model/model.h5" \
    --output "${OUTPUT_DIR}/adversarial_train" \
    --checkpoint_path "${OUTPUT_DIR}/adversarial_train/ckpts" \
    --tensorboard_path ${OUTPUT_DIR}/adversarial_train


MODEL_WEIGHTS="${OUTPUT_DIR}/model/model.h5"  \
    CALIB_WEIGHTS="${OUTPUT_DIR}/calibrate/calibration.weights" \
    seldon-core-microservice ${SRC_DIR}/app/PetSetModel --service-type MODEL --persistence 0 REST