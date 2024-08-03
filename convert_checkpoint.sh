mkdir checkpoint

python tools/misc/publish_model.py \
        work_dirs/rtmpose-s-triangle/epoch_300.pth \
        checkpoint/rtmpose_s_triangle_300.pth


python tools/misc/publish_model.py \
        work_dirs/rtmpose-s-triangle/best_PCK_epoch_40.pth \
        checkpoint/rtmpose_s_triangle_best_PCK_epoch_40.pth