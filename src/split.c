#include "split.h"
#include "darknet.h"
#include "dark_cuda.h"
#include "gemm.h"
#include "blas.h"

layer make_split_layer(int batch, int w, int h, int c, int index)
{
    layer l = {(LAYER_TYPE)0};
    l.type = SPLIT;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = l.out_w * l.out_h * l.out_c;
    l.inputs = l.w * l.h * l.c;
    l.index = index;
    int output_size = l.outputs * batch;
    l.delta = (float*)calloc(output_size,sizeof(float));
    l.output = (float*)calloc(output_size,sizeof(float));
    l.forward = forward_split_layer;
    l.backward = backward_split_layer;
    #ifdef GPU
    l.forward_gpu = forward_split_layer_gpu;
    l.backward_gpu = backward_split_layer_gpu;
    l.delta_gpu = cuda_make_array(l.delta,output_size);
    l.output_gpu = cuda_make_array(l.output,output_size);
    #endif
    fprintf(stderr, "split         %4d x%4d x%4d   ->  %4d x%4d x%4d \n", w, h, c, l.out_w, l.out_h, l.out_c);
    return l;
}

void reszie_split_layer(layer *l,int h, int w)
{
    l->h = h;
    l->w = w;
    l->out_h = h;
    l->out_w = w;
    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->h * l->w * l->c;
    l->delta = (float*)realloc(l->delta,l->outputs * l->batch * sizeof(float));
    l->output = (float*)realloc(l->output,l->outputs * l->batch * sizeof(float));
}

void forward_split_layer(const layer l,network_state net)
{
    int batch_size = l.batch;
    int nsize = l.h * l.w * l.c;
    int index = l.index;
    float *input = net.net.layers[index].output;
    #pragma omp parallel for
    for(int j =0; j< batch_size;++j){
        float *p_i = input + j * nsize;
        float *p_o = l.output + j * nsize;
        copy_cpu(nsize,p_i,1,p_o,1);
    }
}

void backward_split_layer(const layer l,network_state state)
{
    int batch_size = l.batch;
    int nsize = l.w * l.h * l.c;
    float *output = state.net.layers[l.index].delta;
    int i;
    #pragma omp parallel for
    for(i=0; i<batch_size;++i){
        axpy_cpu(nsize,1,l.delta + i * l.outputs,1, output + i * l.outputs,1 );
    }
}

#ifdef GPU
void forward_split_layer_gpu(const layer l,network_state net)
{
    int batch_size = l.batch;
    int nsize = l.h * l.w * l.c;
    int index = l.index;
    float *input = net.net.layers[index].output_gpu;
    for(int j =0; j< batch_size;++j){
        simple_copy_ongpu(nsize,input + j * nsize,l.output_gpu + j * nsize);
    }
}

void backward_split_layer_gpu(const layer l,network_state state)
{
    int batch_size = l.batch;
    int nsize = l.w * l.h * l.c;
    float *output = state.net.layers[l.index].delta_gpu;
    int i;
    for(i=0; i<batch_size;++i){
        axpy_ongpu(nsize, 1, l.delta_gpu + i * l.outputs, 1, output + i * l.outputs,1);
    }
}

#endif

