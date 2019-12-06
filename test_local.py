import argparse
from mtcnn.core.detect import MtcnnDetector,create_mtcnn_net
import cv2
import time
from  mtcnn.config import *

pnet, rnet, onet = create_mtcnn_net(p_model_path=PNET_MODEL_PATH,r_model_path=RNET_MODEL_PATH,
                                 o_model_path=ONET_MODEL_PATH,use_cuda=False)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=48)



cap = cv2.VideoCapture(0)
f = 0
stime = time.time()
while (True):
    ret, frame = cap.read()  # 读取一帧的图像
    frame = cv2.resize(frame,(480,360))
    # frame = cv2.imread('face.jpg')
    boxes, boxes_align = mtcnn_detector.detect_pnet(im=frame)
    rboxes, rboxes_align = mtcnn_detector.detect_rnet(im=frame, dets=boxes_align)
    # oboxes,olandmark = mtcnn_detector.detect_onet(im=frame,dets=rboxes_align)
    if rboxes_align is not None:
        for box in rboxes_align:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 128), 2)
    if f % 20 == 0:
        fps = int(20/(time.time()-stime))
        f = 0
        stime = time.time()
    cv2.putText(frame, '{:d}fps'.format(fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 255), 2)
    cv2.imshow('Face Recognition', frame)
    f += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 释放摄像头
cv2.destroyAllWindows()
