#ifndef CHANNEL_SHUFFLE_H
#define CHANNEL_SHUFFLE_H

#include "layer.h"
#include "network.h"
#include "dark_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif
layer make_channel_shuffle_layer(int batch,int w,int h, int c,int groups);
void resize_channel_shuffle_layer(layer *l,int h, int w);
void forward_channel_shuffle_layer(const layer l,network_state state);
void backward_channel_shuffle_layer(const layer l,network_state state);
#ifdef GPU
void forward_channel_shuffle_layer_gpu(layer l,network_state net);
void backward_channel_shuffle_layer_gpu(const layer l,network_state state);
#endif 

#ifdef __cplusplus
}
#endif
#endif // CHANNEL_SHUFFLE_H
