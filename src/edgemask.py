import cv2
import numpy as np

## Parameters

def generate_edge_mask(image_input, kernel_size=10, show=False, save_path=None):
    img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 1.5)
    edges = cv2.Canny(blurred, threshold1=0, threshold2=15)
    edge_mask = (edges > 0).astype(np.uint8) * 255
    edge_mask = cv2.dilate(edge_mask, np.ones((kernel_size, kernel_size), np.uint8))
    if show:
        cv2.imshow("Edge Mask", edge_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if save_path:
        cv2.imwrite(save_path, edge_mask)
    return edge_mask

KERNEL_SIZE = 10
FILENAME = "n015-2018-07-24-11-22-45+0800__CAM_BACK_LEFT__1532402927647423"
SAMPLE_ID = 0

image = cv2.imread(f"./calibration_data/imgs/sample{SAMPLE_ID}/depth_imgs/{FILENAME}.jpg", cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
edges = cv2.Canny(image, threshold1=0, threshold2=15)
edge_mask = (edges > 0).astype(np.uint8) * 255
edge_mask = cv2.dilate(edge_mask, np.ones((KERNEL_SIZE,KERNEL_SIZE), np.uint8))

image = cv2.imread(f"./calibration_data/imgs/sample{SAMPLE_ID}/depth_imgs/{FILENAME}.jpg")
cv2.imshow("Edge Mask", edge_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(f"./calibration_data/imgs/sample{SAMPLE_ID}/edgemasks/{FILENAME}.jpg", edge_mask)
