
# 下载配置文件 config 
# wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20220610-mmpose/triangle_dataset/rtmpose-s-triangle.py -P data/

# 需要安装  pip install albumentations 
#
# Albumentations支持所有常见的计算机视觉任务，如分类、语义分割、实例分割、目标检测和姿态估计。 
#
# 对图像的预处理的办法, 在相同的对图像的处理下， 比其他处理方式更快
# 
#（该库提供了一个简单统一的API，用于处理所有数据类型:图像(rbg图像、灰度图像、多光谱图像)、分割掩码、边界框和关键点
# 像素级转换包括：模糊，色彩抖动，图像压缩，高斯噪声，倒置，归一化，随机雨，随机亮度对比，锐化，色相饱和度值等等
# 空间级变换     将同时更改输入图像以及其他目标，例如蒙版，边界框和关键点

# https://www.bilibili.com/video/BV12a4y1u7sd config文件讲解

CUDA_VISIBLE_DEVICES=2 python tools/train.py data/rtmpose-s-triangle.py

# 配置文件 val_interval = 10 每10个epoch会跑一下训练集 测试集上精度更好的会替换到之前best的checkpoint文件

# 最好的是在40epoch
# Epoch(train)  [40][11/11]  base_lr: 4.000000e-03 lr: 4.000000e-03  eta: 2:54:10  time: 3.366752  data_time: 3.267923  memory: 718  loss: 0.146595  loss_kpt: 0.146595  acc_pose: 0.698718
# Epoch(val) [40][10/10]    coco/AP: 0.419052  coco/AP .5: 0.848477  coco/AP .75: 0.269792  coco/AP (M): -1.000000  coco/AP (L): 0.419052  coco/AR: 0.490000  coco/AR .5: 0.875000  coco/AR .75: 0.400000  coco/AR (M): -1.000000  coco/AR (L): 0.490000  PCK: 0.816667  AUC: 0.049167  NME: 0.030452  data_time: 0.450027  time: 0.478831
# 300epoch 测试集AP居然是0了? loss反而高了! (看数据 从105到106 106的loss就从0.38立刻飙到3.0了!) base_lr和lr都正常下降  
# Epoch(train) [300][11/11]  base_lr: 2.000034e-04 lr: 2.000034e-04  eta: 0:00:00  time: 3.402896  data_time: 3.303589  memory: 718  loss: 0.379717  loss_kpt: 0.379717  acc_pose: 0.000000
# Epoch(val) [300][10/10]    coco/AP: 0.000000  coco/AP .5: 0.000000  coco/AP .75: 0.000000  coco/AP (M): -1.000000  coco/AP (L): 0.000000  coco/AR: 0.000000  coco/AR .5: 0.000000  coco/AR .75: 0.000000  coco/AR (M): -1.000000  coco/AR (L): 0.000000  PCK: 0.050000  AUC: 0.003750  NME: 0.634052  data_time: 0.445603  time: 0.477683
