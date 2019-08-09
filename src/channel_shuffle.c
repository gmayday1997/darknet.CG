#include "channel_shuffle.h"
#include "darknet.h"
#include "dark_cuda.h"
#include "blas.h"
#include <assert.h>

layer make_channel_shuffle_layer(int batch,int w,int h, int c,int groups)
{
    layer l = {(LAYER_TYPE)0};
    l.type = CHANNEL_SHUFFLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.groups = groups;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;
    int output_size = l.outputs * batch;
    l.delta = (float*)calloc(output_size, sizeof(float));
    l.output = (float*)calloc(output_size, sizeof(float));
    l.forward = forward_channel_shuffle_layer;
    l.backward = backward_channel_shuffle_layer;
    #ifdef GPU
    l.forward_gpu = forward_channel_shuffle_layer_gpu;
    l.backward_gpu = backward_channel_shuffle_layer_gpu;
    l.delta_gpu = cuda_make_array(l.output,output_size);
    l.output_gpu = cuda_make_array(l.delta,output_size);
    #endif
    fprintf(stderr, "channel_shuffle              %4d x%4d x%4d   ->  %4d x%4d x%4d \n", w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void resize_channel_shuffle_layer(layer *l,int h, int w)
{
    l->h = h;
    l->w = w;
    l->out_h = h;
    l->out_w = w;
    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->h * l->w * l->c;
    l->delta = (float*)realloc(l->delta,l->outputs * l->batch * sizeof(float));
    l->output = (float*)realloc(l->output,l->outputs * l->batch * sizeof(float));
    int output_size = l->outputs * l->batch;
#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
#endif
}

void channel_shuffle_op(float *output,float *input, int group_row,int group_colomn,int len)
{
    for(int i =0; i < group_row; i++)
    {
        for(int j =0; j < group_colomn; j++){
            float *p_i = input + (i*group_colomn + j) * len;
            float *p_o = output + (j*group_row + i) * len;
            copy_cpu(len,p_i,1,p_o,1);
        }
    }
}

void forward_channel_shuffle_layer(const layer l,network_state net)
{
    int channel = l.c;
    int group_row = l.groups;
    int batch_size = l.batch;
    int spatial_size = l.w * l.h;
    int feature_map_size = spatial_size * channel;
    int group_column = (int)(channel / group_row);
    for(int n = 0; n < batch_size;++n){
        channel_shuffle_op(l.output + n* feature_map_size, net.input+n * feature_map_size,group_row,group_column,spatial_size);
    }
}

void backward_channel_shuffle_layer(const layer l,network_state state)
{
    int channel = l.c;
    int group_row = l.groups;
    int batch_size = l.batch;
    int spatial_size = l.w * l.h;
    int feature_map_size = spatial_size * channel;
    int group_column = (int)(channel / group_row);
    for(int n = 0; n < batch_size;++n)
    {
        channel_shuffle_op(state.delta + n* feature_map_size,l.delta + n* feature_map_size,\
                           group_row,group_column,spatial_size);
    }
}

#ifdef GPU
void forward_channel_shuffle_layer_gpu(layer l,network_state net)
{
    int channel = l.c;
    int group_row = l.groups;
    int batch_size = l.batch;
    int spatial_size = l.w * l.h;
    int feature_map_size = spatial_size * channel;
    int group_column = (int)(channel / group_row);
    int count = batch_size * group_column * group_row * spatial_size;
    channel_shuffle_ongpu(count,l.output_gpu,net.input,group_row, group_column,feature_map_size, spatial_size);
}

void backward_channel_shuffle_layer_gpu(const layer l,network_state state)
{
    int channel = l.c;
    int group_row = l.groups;
    int batch_size = l.batch;
    int spatial_size = l.w * l.h;
    int feature_map_size = spatial_size * channel;
    int group_column = (int)(channel / group_row);
    int count = batch_size * group_column * group_row * spatial_size;
    channel_shuffle_ongpu(count,state.delta,l.delta_gpu,group_row, group_column,feature_map_size,spatial_size);
}
#endif
