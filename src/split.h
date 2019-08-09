#ifndef SPLIT_H
#define SPLIT_H

#include "layer.h"
#include "network.h"
#include "darknet.h"

#ifdef __cplusplus
extern "C"{
#endif
layer make_split_layer(int batch, int w, int h, int c, int index);
void reszie_split_layer(layer *l,int h, int w);
void forward_split_layer(const layer l, network_state net);
void backward_split_layer(const layer l,network_state state);

#ifdef GPU
void forward_split_layer_gpu(const layer l, network_state net);
void backward_split_layer_gpu(const layer l, network_state state);
#endif

#ifdef __cplusplus
}
#endif
#endif // SPLIT_H
