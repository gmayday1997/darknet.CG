#ifndef PRUNE_H
#define PRUNE_H
#include "darknet.h"

#ifdef __cplusplus
extern "C"{
#endif
struct layer_info{

    LAYER_TYPE type;
    int old_index;
    int old_channel_number;
    int remain_channel_number;
    int old_kernel_number;
    int remain_kernel_number;
    int *input_layers;
    float *channel_remain_mask;
    float *kernel_remain_mask;
};

extern void prune_yolov3(char *cfgfile, char *weightfile, float prune_ratio);
extern void run_prune(int argc, char **argv);
#ifdef __cplusplus
}
#endif
//void run_prune(int argc, char **argv);
#endif // PRUNE_H

