# Classifier

Train pet classifier:
```bash
OUTPUT_DIR=`pwd`
python ${SRC_DIR}/app/download_petset.py --output ${OUTPUT_DIR}/warehouse
python ${SRC_DIR}/app/dataset_gen.py --input ${OUTPUT_DIR}/warehouse --output ${OUTPUT_DIR}/transform

python ${SRC_DIR}/app/classifer/train.py --model_arch MobileNetV2 --input ${OUTPUT_DIR}/transform \
   --output ${OUTPUT_DIR}/train \
   --checkpoint_path "${OUTPUT_DIR}/train/ckpts" \
   --tensorboard_path ${OUTPUT_DIR}/train


python ${SRC_DIR}/app/classifer/adversarial_nsl_train.py --model_arch VGG19 --input ${OUTPUT_DIR}/transform \
   --output ${OUTPUT_DIR}/train \
   --checkpoint_path "${OUTPUT_DIR}/train/ckpts" \
   --tensorboard_path ${OUTPUT_DIR}/train
```