import argparse
import sys
from mtcnn.core.imagedb import ImageDB
import mtcnn.train_net.train as train
import mtcnn.config as config
import os
from mtcnn.core.image_reader import TrainImageReader



def train_net(annotation_file, model_store_path,
                end_epoch=16, frequent=200, lr=0.01, batch_size=128, use_cuda=False):

    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb()
    # gt_imdb = imagedb.append_flipped_images(gt_imdb)
    train_data = TrainImageReader(gt_imdb, 24, 16, shuffle=False)
    accuracy_list = []
    cls_loss_list = []
    bbox_loss_list = []
    for batch_idx, (image, (gt_label, gt_bbox, gt_landmark)) in enumerate(train_data):
        print(len(image))
        print(gt_bbox)
        break

def parse_args():
    parser = argparse.ArgumentParser(description='Train  RNet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    parser.add_argument('--anno_file', dest='annotation_file',
                        default=os.path.join(config.ANNO_STORE_DIR,config.RNET_TRAIN_IMGLIST_FILENAME), help='training data annotation file', type=str)
    parser.add_argument('--model_path', dest='model_store_path', help='training model store directory',
                        default=config.MODEL_STORE_DIR, type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=config.END_EPOCH, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--lr', dest='lr', help='learning rate',
                        default=config.TRAIN_LR, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
                        default=config.TRAIN_BATCH_SIZE, type=int)
    parser.add_argument('--gpu', dest='use_cuda', help='train with gpu',
                        default=config.USE_CUDA, type=bool)
    parser.add_argument('--prefix_path', dest='', help='training data annotation images prefix root path', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print('train Rnet argument:')
    print(args)


    train_net(annotation_file=args.annotation_file, model_store_path=args.model_store_path,
                end_epoch=args.end_epoch, frequent=args.frequent, lr=args.lr, batch_size=args.batch_size, use_cuda=args.use_cuda)
