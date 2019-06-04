import cv2
import torch

from face_ssd_infer import SSD
from utils import vis_detections

device = torch.device("cpu")
conf_thresh = 0.3
target_size = (800, 800)


net = SSD("test")
net.load_state_dict(torch.load('weights/WIDERFace_DSFD_RES152.pth'))
net.to(device).eval();

img_path = './imgs/12_Group_Group_12_Group_Group_12_128.jpg'

img = cv2.imread(img_path, cv2.IMREAD_COLOR)

detections = net.detect_on_image(img, target_size, device, is_pad=False, keep_thresh=conf_thresh)
vis_detections(img, detections, conf_thresh, show_text=False)