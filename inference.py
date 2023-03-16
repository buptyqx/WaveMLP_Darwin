#coding:utf8
import torch
import numpy as np
from network import C3D_model
from network import R3D_model
from network import R2Plus1D_model
import cv2
import sys
import os
import argparse

## 输入模型权重，输入视频，结果图片路径，结果视频路径，帧率
parser = argparse.ArgumentParser('inference images')
parser.add_argument('--inputvideo', dest='inputvideo', help='input video', type=str, default='videos/太极拳1.mp4')
parser.add_argument('--outputvideo', dest='outputvideo', help='output video', type=str, default='videos/result_videos/太极拳1.mp4')
parser.add_argument('--outputimagesdir', dest='outputimagesdir', help='output images', type=str, default='videos/太极拳1')
parser.add_argument('--samplefrequency', dest='samplefrequency', help='video samplefrequency', type=int, default=4)
parser.add_argument('--model', dest='model', help='model', type=str, default='R3D')
parser.add_argument('--weightspath', dest='weightspath', help='input directory for modelweights', type=str, default='checkpoints/UCF-101/R3D/models/R3D-ucf101_epoch-49.pth.tar')
args = parser.parse_args()

torch.backends.cudnn.benchmark = True

def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    with open('./dataloaders/ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    ## 初始化模型
    if args.model == "R3D":
        model = R3D_model.R3DClassifier(num_classes=101,layer_sizes=(2, 2, 2, 2))
    elif args.model == "C3D":
        model = C3D_model.C3D(num_classes=101)
    elif args.model == "R2Plus1D":
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=101, layer_sizes=(2, 2, 2, 2))

    checkpoint = torch.load(args.weightspath, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    torch.no_grad()

    ## 输出图像文件夹
    imageresult = args.outputimagesdir
    if not os.path.exists(imageresult):
        os.mkdir(imageresult)

    ## 读取视频
    capin = cv2.VideoCapture(args.inputvideo)
    if capin.isOpened()== False:
        print('bad video')
    else:
        width  = capin.get(cv2.CAP_PROP_FRAME_WIDTH)  
        height = capin.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = int(capin.get(cv2.CAP_PROP_FPS)) #获得帧率
        print('video width='+str(width))
        print('video height='+str(height))
        print('video fps='+str(fps))

    retaining = True

    ## 输出视频结果
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoout = cv2.VideoWriter(args.outputvideo, fourcc, fps, (int(width),int(height)))
    clip = [] ##缓冲图像序列
    frequency = args.samplefrequency ##采样频率
    i = 0
    while retaining:
        retaining, frame = capin.read()
        if not retaining and frame is None:
            continue
        i = i + 1
        if frame.shape[1] > 2000.0:
            font = 4.0
        if frame.shape[1] > 1500.0:
            font = 2.0
        elif frame.shape[1] > 1000.0:
            font = 1.0
        else:
            font = 0.6
        if i % frequency != 0:
            continue
        tmp_ = cv2.resize(frame, (171, 128))
        tmp_ = center_crop(tmp_)
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)

        ## 每次取16帧进行推理
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
            
            cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (50, int(50*font)),
                        cv2.FONT_HERSHEY_SIMPLEX, font,
                        (0, 255, 255), 2)
            cv2.putText(frame, "prob: %.4f" % probs[0][label], (50, int(100*font)),
                        cv2.FONT_HERSHEY_SIMPLEX, font,
                        (0, 255, 255), 2)
            cv2.imwrite(os.path.join(imageresult,str(i)+'.png'),frame)
            videoout.write(frame)
            clip.pop(0)

        #cv2.imshow('result', frame)
        #k = cv2.waitKey(1)
        #if k == ord('q'):
            #break

    capin.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()









