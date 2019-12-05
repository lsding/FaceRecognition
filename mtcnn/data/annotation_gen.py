import os
from scipy.io import loadmat
import time

class DATA:
    def __init__(self, image_name, bboxes):
        self.image_name = image_name
        self.bboxes = bboxes


class WIDER(object):
    def __init__(self, file_to_label, path_to_image=None):
        self.file_to_label = file_to_label
        self.path_to_image = path_to_image

        self.f = loadmat(file_to_label)
        self.event_list = self.f['event_list']
        self.file_list = self.f['file_list']
        self.face_bbx_list = self.f['face_bbx_list']

    def next(self):
        for event_idx, event in enumerate(self.event_list):
            e = event[0][0]
            for file, bbx in zip(self.file_list[event_idx][0],
                                 self.face_bbx_list[event_idx][0]):
                f = file[0][0]
                path_of_image = os.path.join(self.path_to_image, e, f) + ".jpg"

                bboxes = []
                bbx0 = bbx[0]
                for i in range(bbx0.shape[0]):
                    # xmin, ymin, xmax, ymax = bbx0[i]
                    xmin, ymin, w, h = bbx0[i]

                    bboxes.append((int(xmin), int(ymin), int(xmin)+int(w), int(ymin)+int(h)))
                yield DATA(path_of_image, bboxes)


#wider face original images path
path_to_image = '/web2/ding/face/WIDER_train/images'

#matlab file path
file_to_label = '/web2/ding/face/wider_face_split/wider_face_train.mat'

#target file path
target_file = '/web2/ding/face/WIDER_train/anno.txt'

wider = WIDER(file_to_label, path_to_image)


line_count = 0
box_count = 0

print('start transforming....')
t = time.time()

with open(target_file, 'w+') as f:
    # press ctrl-C to stop the process
    for data in wider.next():
        line = []
        line.append(str(data.image_name))
        line_count += 1
        for i,box in enumerate(data.bboxes):
            box_count += 1
            for j,bvalue in enumerate(box):
                line.append(str(bvalue))
        line.append('\n')

        line_str = ' '.join(line)
        f.write(line_str)

st = time.time()-t
print('end transforming')

print('spend time:%ld'%st)
print('total line(images):%d'%line_count)
print('total boxes(faces):%d'%box_count)
