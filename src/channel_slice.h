#ifndef CHANNEL_SLICE_H
#define CHANNEL_SLICE_H

#include "layer.h"
#include "network.h"
#include "dark_cuda.h"

typedef layer channel_slice_layer;
#ifdef __cplusplus
extern "C"{
#endif

channel_slice_layer make_channel_slice_layer(int batch, int w, int h, int c, \
                                             int begin_slice_point, int end_slice_point, int axis,\
                                             int n,int *input_layers,int *input_sizes);

void resize_channel_slice_layer(channel_slice_layer *l,int h,int w);
void forward_channel_slice_layer(const layer l, network_state net);
void backward_channel_slice_layer(const layer l, network_state state);
#ifdef GPU
void forward_channel_slice_layer_gpu(channel_slice_layer l,network_state net);
void backward_channel_slice_layer_gpu(channel_slice_layer l,network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif //CHANNEL_SLICE_H

