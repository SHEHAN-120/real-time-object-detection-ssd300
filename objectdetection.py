import cv2
import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import functional as F

COCO_INSTANCE_CATEGORY_NAMES = [  
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

model=ssd300_vgg16(pretrained=True)
model.eval()
cap=cv2.VideoCapture("D:\Our aak model car Unboxing and testing #rccar.mp4")

while True:
    ret,frame=cap.read()
    if not ret:
        break

    img_tensor=F.to_tensor(frame).unsqueeze(0)
    with torch.no_grad():
        outputs=model(img_tensor)[0]
    for box,label_idx,score in zip(outputs['boxes'],outputs['labels'],outputs['scores']):
        if score > 0.5:
            x1,y1,x2,y2=box
            x1,y1,x2,y2=int(x1.item()),int(y1.item()),int(x2.item()),int(y2.item())
            label=COCO_INSTANCE_CATEGORY_NAMES[label_idx]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"{label}:{score:.2f}",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    cv2.imshow('SSD Video',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()