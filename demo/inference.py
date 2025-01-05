import torch
import numpy as np
from detectron2.utils.visualizer import Visualizer
# from demo import predictor
from detectron2.engine import DefaultPredictor
import cv2
image = cv2.imread("deepglobe/images/test/951_0_0.jpg")  # 读取测试图像
# 模型推理
outputs = DefaultPredictor(image)

# 获取语义分割结果
sem_seg = outputs["sem_seg"].argmax(0).to("cpu").numpy()  # 将结果转换为单通道类别图
print(sem_seg.shape)  # (H, W)

# 可视化结果
visualizer = Visualizer(image[:, :, ::-1])  # 转换为 RGB 图像
vis_output = visualizer.draw_sem_seg(sem_seg)
# cv2.imshow("Semantic Segmentation", vis_output.get_image()[:, :, ::-1])
# cv2.waitKey(0)

cv2.imwrite("result.jpg", vis_output.get_image()[:, :, ::-1])