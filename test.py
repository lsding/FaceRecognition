import argparse
from mtcnn.core.detect import MtcnnDetector,create_mtcnn_net
import cv2

pnet, rnet, onet = create_mtcnn_net(p_model_path='pnet_epoch_15.pt',r_model_path='rnet_epoch_10.pt',
                                 o_model_path='onet_epoch.pt', use_cuda=False)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=48)



# cap = cv2.VideoCapture(0)
# while (True):
#     ret, frame = cap.read()  # 读取一帧的图像
#     frame = cv2.resize(frame,(640,360))
#     frame = cv2.imread('face.jpg')
#     boxes, boxes_align = mtcnn_detector.detect_pnet(im=frame)
#     rboxes, rboxes_align = mtcnn_detector.detect_rnet(im=frame, dets=boxes_align)
#     oboxes,olandmark = mtcnn_detector.detect_onet(im=frame,dets=rboxes_align)
#     print(oboxes)
#     for box in oboxes:
#         cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 128), 2)
#     cv2.imshow('Face Recognition', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()  # 释放摄像头
# cv2.destroyAllWindows()

frame = cv2.imread('face.jpg')
boxes, boxes_align = mtcnn_detector.detect_pnet(im=frame)
rboxes, rboxes_align = mtcnn_detector.detect_rnet(im=frame, dets=boxes_align)
oboxes,olandmark = mtcnn_detector.detect_onet(im=frame,dets=rboxes_align)
print(oboxes)
for box in oboxes:
    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 128), 2)
cv2.imshow('Face Recognition', frame)
cv2.waitKey()