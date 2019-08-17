# Darknet 

Darknet is an open source neural network framework written in C and CUDA. It is fast, easy to install, and supports CPU and GPU computation.

This repository is borrowed heavily from https://github.com/pjreddie/darknet and https://github.com/AlexeyAB/darknet

# new features
 - [shufflenetv2](https://arxiv.org/abs/1807.11164):
[channel_shuffle layer](https://github.com/gmayday1997/darknet.CG/blob/master/src/channel_shuffle.c) and 
[channel_slice layer](https://github.com/gmayday1997/darknet.CG/blob/master/src/channel_slice.c) are added in this repo.

![img1](https://user-images.githubusercontent.com/16068384/39479361-9f1345c0-4d97-11e8-8201-4a45ac4a6c7e.png)

- [yolov3 slimming](https://arxiv.org/abs/1708.06519):
[prune.cpp](https://github.com/gmayday1997/darknet.CG/blob/master/src/prune.cpp) is added.

 ![img2](https://user-images.githubusercontent.com/8370623/29604272-d56a73f4-879b-11e7-80ea-0702de6bd584.jpg)

## How to use

- shufflenev2: 
  for example basic unit
  
| ![basicunit](https://img3.doubanio.com/view/status/raw/public/e99ac6d308ca60e.jpg) | ![darknetcfg](https://img3.doubanio.com/view/status/raw/public/2928419c25e8e21.jpg) |
|---|---|

- yolov3 slimming 
```
  ./darknet prune ./cfg/yolov3.cfg ./cfg/yolov3.weights -rate 0.3
```
```
pruning threshold is: 0.878803
network slimming starting
layer_0:convolutional count of kernel is 32  count of pruned kernel is 2
layer_1:convolutional count of kernel is 64  count of pruned kernel is 0
layer_2:convolutional count of kernel is 32  count of pruned kernel is 8
layer_3:convolutional count of kernel is 64  count of pruned kernel is 0
layer_4:shortcut
layer_5:convolutional count of kernel is 128  count of pruned kernel is 0
layer_6:convolutional count of kernel is 64  count of pruned kernel is 50
layer_7:convolutional count of kernel is 128  count of pruned kernel is 0
layer_8:shortcut
layer_9:convolutional count of kernel is 64  count of pruned kernel is 11
layer_10:convolutional count of kernel is 128  count of pruned kernel is 0
layer_11:shortcut
layer_12:convolutional count of kernel is 256  count of pruned kernel is 0
layer_13:convolutional count of kernel is 128  count of pruned kernel is 92
layer_14:convolutional count of kernel is 256  count of pruned kernel is 0
layer_15:shortcut
layer_16:convolutional count of kernel is 128  count of pruned kernel is 59
layer_17:convolutional count of kernel is 256  count of pruned kernel is 0
layer_18:shortcut
layer_19:convolutional count of kernel is 128  count of pruned kernel is 66
layer_20:convolutional count of kernel is 256  count of pruned kernel is 0
layer_21:shortcut
layer_22:convolutional count of kernel is 128  count of pruned kernel is 56
layer_23:convolutional count of kernel is 256  count of pruned kernel is 0
layer_24:shortcut
layer_25:convolutional count of kernel is 128  count of pruned kernel is 90
layer_26:convolutional count of kernel is 256  count of pruned kernel is 0
layer_27:shortcut
layer_28:convolutional count of kernel is 128  count of pruned kernel is 84
layer_29:convolutional count of kernel is 256  count of pruned kernel is 0
layer_30:shortcut
layer_31:convolutional count of kernel is 128  count of pruned kernel is 90
layer_32:convolutional count of kernel is 256  count of pruned kernel is 0
layer_33:shortcut
layer_34:convolutional count of kernel is 128  count of pruned kernel is 93
layer_35:convolutional count of kernel is 256  count of pruned kernel is 0
layer_36:shortcut
layer_37:convolutional count of kernel is 512  count of pruned kernel is 0
layer_38:convolutional count of kernel is 256  count of pruned kernel is 91
layer_39:convolutional count of kernel is 512  count of pruned kernel is 0
layer_40:shortcut
layer_41:convolutional count of kernel is 256  count of pruned kernel is 132
layer_42:convolutional count of kernel is 512  count of pruned kernel is 0
layer_43:shortcut
layer_44:convolutional count of kernel is 256  count of pruned kernel is 227
layer_45:convolutional count of kernel is 512  count of pruned kernel is 0
layer_46:shortcut
layer_47:convolutional count of kernel is 256  count of pruned kernel is 216
layer_48:convolutional count of kernel is 512  count of pruned kernel is 0
layer_49:shortcut
layer_50:convolutional count of kernel is 256  count of pruned kernel is 233
layer_51:convolutional count of kernel is 512  count of pruned kernel is 0
layer_52:shortcut
layer_53:convolutional count of kernel is 256  count of pruned kernel is 249
layer_54:convolutional count of kernel is 512  count of pruned kernel is 0
layer_55:shortcut
layer_56:convolutional count of kernel is 256  count of pruned kernel is 201
layer_57:convolutional count of kernel is 512  count of pruned kernel is 0
layer_58:shortcut
layer_59:convolutional count of kernel is 256  count of pruned kernel is 242
layer_60:convolutional count of kernel is 512  count of pruned kernel is 0
layer_61:shortcut
layer_62:convolutional count of kernel is 1024  count of pruned kernel is 0
layer_63:convolutional count of kernel is 512  count of pruned kernel is 337
layer_64:convolutional count of kernel is 1024  count of pruned kernel is 0
layer_65:shortcut
layer_66:convolutional count of kernel is 512  count of pruned kernel is 422
layer_67:convolutional count of kernel is 1024  count of pruned kernel is 0
layer_68:shortcut
layer_69:convolutional count of kernel is 512  count of pruned kernel is 432
layer_70:convolutional count of kernel is 1024  count of pruned kernel is 0
layer_71:shortcut
layer_72:convolutional count of kernel is 512  count of pruned kernel is 348
layer_73:convolutional count of kernel is 1024  count of pruned kernel is 0
layer_74:shortcut
layer_75:convolutional count of kernel is 512  count of pruned kernel is 6
layer_76:convolutional count of kernel is 1024  count of pruned kernel is 0
layer_77:convolutional count of kernel is 512  count of pruned kernel is 5
layer_78:convolutional count of kernel is 1024  count of pruned kernel is 0
layer_79:convolutional count of kernel is 512  count of pruned kernel is 9
layer_80:convolutional count of kernel is 1024  count of pruned kernel is 3
layer_81:convolutional count of kernel is 255  count of pruned kernel is 0
layer_82:none
layer_83:route
layer_84:convolutional count of kernel is 256  count of pruned kernel is 0
layer_85:none
layer_86:route
layer_87:convolutional count of kernel is 256  count of pruned kernel is 25
layer_88:convolutional count of kernel is 512  count of pruned kernel is 18
layer_89:convolutional count of kernel is 256  count of pruned kernel is 22
layer_90:convolutional count of kernel is 512  count of pruned kernel is 19
layer_91:convolutional count of kernel is 256  count of pruned kernel is 19
layer_92:convolutional count of kernel is 512  count of pruned kernel is 10
layer_93:convolutional count of kernel is 255  count of pruned kernel is 0
layer_94:none
layer_95:route
layer_96:convolutional count of kernel is 128  count of pruned kernel is 0
layer_97:none
layer_98:route
layer_99:convolutional count of kernel is 128  count of pruned kernel is 31
layer_100:convolutional count of kernel is 256  count of pruned kernel is 42
layer_101:convolutional count of kernel is 128  count of pruned kernel is 21
layer_102:convolutional count of kernel is 256  count of pruned kernel is 27
layer_103:convolutional count of kernel is 128  count of pruned kernel is 12
layer_104:convolutional count of kernel is 256  count of pruned kernel is 29
layer_105:convolutional count of kernel is 255  count of pruned kernel is 0
layer_106:none
start to write cfg file
save pruned cfg file to: ./cfg/yolov3_prune.cfg
start to write weights
save pruned weights file to: yolov3_prune.weights
```
 <!-- ![img3](https://img1.doubanio.com/view/status/raw/public/0d1e2ae81cea1fc.jpg) -->

 the pruned cfg/weights are saved as ./cfg/yolov3_prune.cfg  ./cfg/yolov3_prune.weights
