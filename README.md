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

  Support: yolov3-tiny, yolov3, yolov3-spp
```
  ./darknet prune ./cfg/yolov3.cfg ./cfg/yolov3.weights -rate 0.3
```
 the pruned cfg/weights are saved as ./cfg/yolov3_prune.cfg  ./cfg/yolov3_prune.weights
 
## Results

 - shufflenetv2
 
 Top1: 0.52 Top5: 0.75
 
 shuffle_imagenet.cfg : [shuffle_imagenet.cfg.txt](https://github.com/AlexeyAB/darknet/files/3619916/shuffle_imagenet.cfg.txt)
 shuffle_imagenet.weights: [google driver](https://drive.google.com/file/d/1e1e1v2E659B3wxzJYHwlabjXfFwmgCmn/view?usp=sharing) OR [baidu  pan](https://pan.baidu.com/s/1RdDIsuc5tAgbq39naqQ_xA) (2eyp)

 - yolov3 slimming(coco)
 
|         yolov3       | volume(MB) | FLOPS |Map(coco_val5k @0.5)|  finetuning iters |   parameters |
|:--------------------:|:----------:|:-----:|:------------------:|:-----------------:|:------------:|
|    before pruned     |     246    |   65  |        54.65       |       500k        |      1x      |
|pruned @prune_rate=0.3|     122    |  36.3 |        49.7        |        80k        |     0.5x     |
|pruned @prune_rate=0.5|     60.5   |  16   |        49.2        |       160k        |    0.25x     |
|pruned @prune_rate=0.7|     31     |  8.2  |        46          |       200k        |    0.125x    |
| tiny-yolov3(official)|     36     |  5.5  |        17.3        |                   |              |
|    tiny-yolov3_3l    |     38     |  9.1  |        32          |                   |              |

- speed test(experiment on 1080Ti)

|         yolov3       | volume(MB) | FLOPS |    FPS(352x 288)   |  FPS(960 x 540)   |   FPS(1960 x 1080) |
|:--------------------:|:----------:|:-----:|:------------------:|:-----------------:|:------------------:|
|    before pruned     |     246    |   65  |         60         |         57        |          53        |
|pruned @prune_rate=0.3|     122    |  36.3 |         82         |         78        |          76        |
|pruned @prune_rate=0.5|     60.5   |  16   |        107         |         105       |          97        |
|pruned @prune_rate=0.7|     31     |  8.2  |        130         |         123       |         120        |

- speed test(experiment on XiaoMI laptop (Intel(R)Core(TM)i5-8250U CPU@1.6GHZ))

|         yolov3       | volume(MB) | FLOPS |  Inference Time |
|:--------------------:|:----------:|:-----:|:---------------:|
|    before pruned     |     246    |   65  |      436ms      |
|pruned @prune_rate=0.3|     122    |  36.3 |      230ms      |
|pruned @prune_rate=0.5|     60.5   |  16   |      125ms      |
|pruned @prune_rate=0.7|     31     |  8.2  |      70ms       |

 - download links to pruned cfgs/models
 
  pruned @prune_rate=0.3: [cfg(google driver)](https://drive.google.com/file/d/1eQdmLB4aJtScBicjOdC2L9IUxoRqi6p-/view?usp=sharing),[weight(google driver)](https://drive.google.com/file/d/1-WIkaWsvK61_B2NuEGdMtNWxgGemUCyQ/view?usp=sharing) OR [cfg(baidupan)]( https://pan.baidu.com/s/19OIjfVvOTfAw-5Y9Fp5HuQ)(s846),[weight(baidupan)](https://pan.baidu.com/s/1XNx6Bfc42C2tSSYkG8iUhQ)(eswd) 
  
 pruned @prune_rate=0.5: [cfg(google driver)](https://drive.google.com/file/d/1MLKcYBFDLmhW4fwgY7dbcMAkALDt_b0B/view?usp=sharing),[weight(google driver)](https://drive.google.com/file/d/1F_EEqekMqdo9nc0x126gcyVsChhnh6zU/view?usp=sharing) OR
  [cfg(baidupan)](https://pan.baidu.com/s/1wMhOae8B6ey_nIPfWKWQow)(y9gk), [weight(baidupan)](https://pan.baidu.com/s/1ONDkFCcFsKyH3ccpIh2CuQ)(5eqt)
  
   pruned @prune_rate=0.7: [cfg(google driver)](https://drive.google.com/file/d/1m6OAdFqH6frGvbyyk_3k-K-w0bugUS_6/view?usp=sharing),[weight(google driver)](https://drive.google.com/file/d/1FlXLbvo-0Rzf2d6amHlD2cHkO973_Eqc/view?usp=sharing) OR [cfg(baidupan)](https://pan.baidu.com/s/1hGc0Kzh3Tq0JHzxy0CzH1Q)(xh1d), [weight(baidupan)](https://pan.baidu.com/s/1KrwG7xxI8XQImWSc8PflJg)(ump7)
   
 <!-- ![img3](https://img1.doubanio.com/view/status/raw/public/0d1e2ae81cea1fc.jpg) -->

