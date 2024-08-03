
mkdir -p outputs/Triangle-FastRcNN-RTMPose

# faster-r_cnn 模型配置
FASTER_R_CNN_CONFIG='/home/hehanlong/work/mmdetection/data/faster_r_cnn_triangle.py'
# faster-r_cnn checkpoint文件
FASTER_R_CNN_CKP='/home/hehanlong/work/mmdetection/checkpoint/faster_r_cnn_triangle_epoch_50-48ca4422.pth'
# rtmpose 模型配置
RTMPOSE_CONFIG='data/rtmpose-s-triangle.py'
# rtmpose checkpoint文件 
#RTMPOSE_CKP='checkpoint/rtmpose_s_triangle_300-955b26fe_20240803.pth'
RTMPOSE_CKP='checkpoint/rtmpose_s_triangle_best_PCK_epoch_40-1b600ed5_20240803.pth'
 
# 关键点0  30度
# 关键点1  60度
# 关键点2  90度
python demo/topdown_demo_with_mmdet.py \
        ${FASTER_R_CNN_CONFIG}\
        ${FASTER_R_CNN_CKP} \
        ${RTMPOSE_CONFIG} \
        ${RTMPOSE_CKP} \
        --input data/triangle_test/triangle_4.jpg \
        --output-root outputs/Triangle-FastRcNN-RTMPose \
        --device cuda:0 \
        --bbox-thr 0.5 \
        --kpt-thr 0.5 \
        --nms-thr 0.3 \
        --radius 36 \
        --thickness 30 \
        --draw-bbox \
        --draw-heatmap \
        --show-kpt-idx
