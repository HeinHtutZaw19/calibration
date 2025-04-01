import cv2
import numpy as np

KERNEL_SIZE = 10

image = cv2.imread("../data/imgs/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927612460.jpg", cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
edges = cv2.Canny(image, threshold1=50, threshold2=150)
edge_mask = (edges > 0).astype(np.uint8) * 255
edge_mask = cv2.dilate(edge_mask, np.ones((10,10), np.uint8))

image = cv2.imread("../data/imgs/camera_front.jpg")
cv2.imwrite("../data/imgs/edge_mask.png", edge_mask)
