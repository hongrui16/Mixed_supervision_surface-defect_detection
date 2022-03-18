# python -u train_net.py  \
#     --GPU=0 \
#     --DATASET=STEEL \
#     --RUN_NAME=train \
#     --DATASET_PATH=/comp_robot/hongrui/pot_pro/severstal-steel-defect-detection \
#     --RESULTS_PATH=runs \
#     --SAVE_IMAGES=True \
#     --DILATE=7 \
#     --EPOCHS=150 \
#     --LEARNING_RATE=1.0 \
#     --DELTA_CLS_LOSS=0.01 \
#     --BATCH_SIZE=1 \
#     --WEIGHTED_SEG_LOSS=True \
#     --WEIGHTED_SEG_LOSS_P=2 \
#     --WEIGHTED_SEG_LOSS_MAX=1 \
#     --DYN_BALANCED_LOSS=True \
#     --GRADIENT_ADJUSTMENT=True \
#     --FREQUENCY_SAMPLING=True \
#     --FOLD=0 \
#     --DEBUG=False \
#     --TRAIN_NUM=300 \
#     --NUM_SEGMENTED=150
#     # --TRAIN_NUM=3000 \
#     # --NUM_SEGMENTED=1500
#     # --TRAIN_NUM=300 \
#     # --NUM_SEGMENTED=150 


python -u train_net.py  \
    --GPU=7 \
    --DATASET=CustomPotSeg \
    --RUN_NAME=train \
    --DATASET_PATH=/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_0 \
    --RESULTS_PATH=runs \
    --SAVE_IMAGES=True \
    --DILATE=7 \
    --EPOCHS=150 \
    --LEARNING_RATE=1.0 \
    --DELTA_CLS_LOSS=0.01 \
    --BATCH_SIZE=1 \
    --WEIGHTED_SEG_LOSS=True \
    --WEIGHTED_SEG_LOSS_P=2 \
    --WEIGHTED_SEG_LOSS_MAX=1 \
    --DYN_BALANCED_LOSS=True \
    --GRADIENT_ADJUSTMENT=True \
    --FREQUENCY_SAMPLING=True \
    --FOLD=0 \
    --DEBUG=False \
    --TRAIN_NUM=300 \
    --NUM_SEGMENTED=150 \
    --testValTrain=4 \
    --base_size=480 \
    --crop_size=480 \
    --use_txtfile \
    --pot_train_mode=2 \
    --de_ignore_index
    # --reverse_distance_transform



