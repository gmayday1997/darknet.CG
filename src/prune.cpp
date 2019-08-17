#include "prune.h"
#include "darknet.h"
#include "parser.h"
#include "option_list.h"
#include "list.h"
#include "layer.h"
#include "network.h"
#include "blas.h"
#include "version.h"
#include "parser.h"
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include "utils.h"
using namespace std;

// pruning yolov3 based on batch normalization slimming algrothim
template<typename Dtype>
void print_array(Dtype *array,int sz){
    for(int i =0; i<sz;i++){
        cout << "the " << i << " index of array is " << array[i] << endl;
    }
}

template<typename Dtype>
void print_array_weights(Dtype *array,int sz,FILE *fp){
    for(int i =0; i<sz;i++){
        //cout << "the " << i << " index of array is " << array[i] << endl;
        char buff[100] = {0};
        sprintf(buff,"the %d index of array is %f\n",i,array[i]);
        fprintf(fp,buff);
    }
}

vector<int> parse_remain_layers(network net)
{
    int n = net.n;
    vector<int> remain_indexs;
    for(int i= 0; i < n; ++i){
        if(net.layers[i].type == SHORTCUT){
            //cout << "index of layer:" << i << endl;
            remain_indexs.push_back(i-1);
            if(net.layers[net.layers[i].index].type != SHORTCUT){
                 remain_indexs.push_back(net.layers[i].index);
            }
        }
        if(net.layers[i].type == YOLO){
            remain_indexs.push_back(i-1);
        }
    }
    sort(remain_indexs.begin(),remain_indexs.end());
    return remain_indexs;
}

int find_exist(vector<int> vec,int value)
{
    vector<int>::iterator ret;
    ret = find(vec.begin(),vec.end(),value);
    if(ret == vec.end())return 0;
    else return 1;
}

void stat_batch_norm_count(network net,vector<int> remain_layer_indexs,vector<int> *prune_layer_indexs, int *count)
{
    int n = net.n;
    for(int i =0;i < n; ++i){
        int bn = net.layers[i].batch_normalize;
        if(bn){
            if(!find_exist(remain_layer_indexs,i)){
                 prune_layer_indexs->push_back(i);
                 *count += net.layers[i].out_c;
            }
        }
    }
}

void copy_bn_scales(float *bn_values,network net,vector<int> prune_layer_indexs){

    vector<int> ::iterator it;
    for(it = prune_layer_indexs.begin();it!=prune_layer_indexs.end();it++)
    {
        layer l = net.layers[*it];
        float *bn_scale = l.scales;
        int sz = l.out_c;
        cout << "size of scale at layer " << *it << " is: " << sz << endl;
        convert_abs(sz,bn_scale,1,bn_values,1);
        bn_values += sz;
    }
}

void compare_array_with_thresh(float* mask, float *weight,float thresh,int N, int *mask_sum){

    vector<int> mask_idx(0);
    for(int i =0;i < N;++i){
        if(abs(weight[i]) > thresh) {
            mask[i]=1.0;
            *mask_sum +=1;
            mask_idx.push_back(i);
        }
        else mask[i]=0.0;
    }
}

void free_section_(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void write_cfg(char *filename, list *sections,network *net,vector<layer_info> prune_vec)
{
    ofstream in;
    in.open(filename,ios::trunc);
    node *n = sections->front;
    section *section_ = (section *)n->val;
    list *options = section_->options;
    node *option_cont = options->front;
    in << "[net]" << "\n";
    while(option_cont){
        kvp *p = (kvp *)option_cont->val;
        auto key = p->key;
        auto value = p->val;
        in << key << "=" << value << "\n";
        option_cont = option_cont->next;
    }
    in << "\n";
    free_section_(section_);
    n = n->next;
    int count = 0;
    while(n){
        layer_info lt = prune_vec[count];
        section_ = (section *)n->val;
        options = section_->options;
        char *type_ = section_->type;
        in << "#" << count << "\n";
        in <<  type_ << "\n";
        node *option_cont = options->front;
        while(option_cont){
            kvp *p = (kvp *)option_cont->val;
            auto key = p->key;
            if(!strcmp(type_,"[convolutional]") && !strcmp(key,"filters"))
            {
                in << key << "=" << lt.remain_kernel_number << "\n";
            }else{
                in << key << "=" << p->val << "\n";
            }
            option_cont = option_cont->next;
        }
        in << "\n";
        n = n->next;
        count +=1;
    }
}

void copy_cpu_(float *x, float *y, float *mask,int channel_len,int kernel_size)
{
    int i=0,j=0;
    int count = 0;
    while(j<channel_len){
        if(mask[j]){
            float *x_i = x + j * kernel_size;
            float *y_o = y + i * kernel_size;
            copy_cpu(kernel_size,x_i,1,y_o,1);
            i ++;
            j ++;
            count ++;
        }
        else{
            j ++;
        }
    }
}


void save_prune_conv_weights(layer l,layer_info *per_layer)
{
    // first step: prune kernel
    float *prune_kernel_mask = per_layer->kernel_remain_mask;
    int remain_kernel = per_layer->remain_kernel_number;
    int ori_len = l.c * l.size * l.size;
    int remain_len = per_layer->remain_channel_number * l.size * l.size;
    int nweights = remain_kernel * remain_len;
    float *ori_weights = l.weights;
    float *ori_bias = l.biases;
    float *ori_scale = l.scales;
    float *ori_mean = l.rolling_mean;
    float *ori_variance = l.rolling_variance;

    float *prune_kernel_scale = (float*)calloc(remain_kernel,sizeof(float));
    float *prune_kernel_mean = (float*)calloc(remain_kernel,sizeof(float));
    float *prune_kernel_variance = (float*)calloc(remain_kernel,sizeof(float));
    float *prune_kernel_weights = (float*)calloc(nweights,sizeof(float));
    float *prune_kernel_bias = (float*)calloc(remain_kernel,sizeof(float));
    int i=0,j=0;
    while(j < l.n){
            if(prune_kernel_mask[j]){
                float *weight_i = ori_weights + j * ori_len;
                float *weight_o = prune_kernel_weights + i * remain_len;
                int kernel_size = l.size * l.size;
                copy_cpu_(weight_i,weight_o,per_layer->channel_remain_mask,l.c,kernel_size);
                float *bias_i = ori_bias + j;
                float *bias_o = prune_kernel_bias + i;
                copy_cpu(1,bias_i,1,bias_o,1);
                if(l.batch_normalize)
                {
                    float *scale_i = ori_scale + j;
                    float *scale_o = prune_kernel_scale + i;
                    copy_cpu(1,scale_i,1,scale_o,1);
                    float *mean_i = ori_mean + j;
                    float *mean_o = prune_kernel_mean + i;
                    copy_cpu(1,mean_i,1,mean_o,1);
                    float *variance_i = ori_variance + j;
                    float *variance_o = prune_kernel_variance + i;
                    copy_cpu(1,variance_i,1,variance_o,1);
                }
                i++;
                j++;
            }else{
                j++;}
     }
    per_layer->prune_weights = prune_kernel_weights;
    per_layer->prune_bias = prune_kernel_bias;
    if(l.batch_normalize){
        per_layer->prune_scales = prune_kernel_scale;
        per_layer->prune_means = prune_kernel_mean;
        per_layer->prune_variance = prune_kernel_variance;
    }
}

int *fullelem(int num)
{
    int *elem = (int*)calloc(num,sizeof(int));
    for(int i =0; i< num;i++){
        elem[i] = i;}
    return elem;
}

float* reduce_sum_spatial(float *kernel,int kernel_num,int channel_num,int kernel_size)
{
    int channels = kernel_num * channel_num;
    float *reduce_sum = (float*)calloc(channels, sizeof(float));
    int kernel_sp = kernel_size * kernel_size;
    int i;
    for(i=0;i<channels;++i){
        float *kernel_start = kernel + i * kernel_sp;
        float sum_ = sum_array(kernel_start,kernel_sp);
        reduce_sum[i]=sum_;
    }
    return reduce_sum;
}

float *abandoned_bias(float *remain_mask,float *bias,int bias_num){

    float *abandoned_biases = (float*)calloc(bias_num,sizeof(float));
    int i;
    for(i=0;i < bias_num;++i){
        float bias_ = (1-remain_mask[i]) * bias[i];
        abandoned_biases[i] = bias_ * (bias_ > 0);
    }
    return abandoned_biases;
}

float *transform_mm(float* absorbed_bias,float *reduce_sum,int kernel_num,int channel_num){

    float *result_mm = (float*)calloc(kernel_num,sizeof(float));
    int i;
    for(i=0;i< kernel_num;++i){
        float *sum = reduce_sum + i * channel_num;
        float dot = dot_cpu(channel_num,sum,1,absorbed_bias,1);
        result_mm[i]=dot;
    }
    return result_mm;
}

void write_head2weight(FILE *fp,network net)
{
    uint64_t seen[] = {0};
    int major = MAJOR_VERSION;
    int minor = MINOR_VERSION;
    int revision = PATCH_VERSION;
    fwrite(&major, sizeof(int), 1, fp);
    fwrite(&minor, sizeof(int), 1, fp);
    fwrite(&revision, sizeof(int), 1, fp);
    fwrite(seen, sizeof(uint64_t), 1, fp);
}

void write_weights(FILE *fp,network net,vector<layer_info> prune_layers_vec)
{
    int n = prune_layers_vec.size();
    int i;
    for(i=0; i< n;++i){
        layer l = net.layers[i];
        layer_info layer_ = prune_layers_vec[i];
        LAYER_TYPE lt = layer_.type;
        if(lt == CONVOLUTIONAL){
            int prune_nweights = layer_.remain_kernel_number * layer_.remain_channel_number * l.size * l.size;
            fwrite(layer_.prune_bias,sizeof(float),layer_.remain_kernel_number,fp);
            if(l.batch_normalize){
                fwrite(layer_.prune_scales,sizeof(float),layer_.remain_kernel_number,fp);
                fwrite(layer_.prune_means,sizeof(float),layer_.remain_kernel_number,fp);
                fwrite(layer_.prune_variance,sizeof(float),layer_.remain_kernel_number,fp);
            }
            fwrite(layer_.prune_weights,sizeof(float),prune_nweights,fp);
        }
    }
}

void init_value(layer_info *li){

    li->remain_kernel_number=0;
    li->remain_channel_number=0;
    li->old_channel_number=0;
    li->old_kernel_number=0;
}

void prune_yolov3(char *cfgfile, char *weightfile,float prune_ratio)
{
    char prune_cfg[100] ={0}, prune_weights[100]={0};
    find_replace(cfgfile,".cfg","_prune.cfg",prune_cfg);
    find_replace(weightfile,".weights","_prune.weights",prune_weights);
    list *section = read_cfg(cfgfile);
    network net = parse_network_cfg_custom(cfgfile,1,1);
    if(weightfile){
        load_weights(&net,weightfile);
    }else{
        cout << "weight cannot found, pretrain weight is needed" << endl;
        return;
    }
    vector<int> remain_layer_indexs;
    vector<int> prune_layer_indexs;
    remain_layer_indexs =  parse_remain_layers(net);
    int bn_total_count=0;
    cout << "starting to compute prune threshold" << endl;
    stat_batch_norm_count(net,remain_layer_indexs,&prune_layer_indexs, &bn_total_count);
    float *bn_values =(float*)calloc(bn_total_count,sizeof(float));
    copy_bn_scales(bn_values,net,prune_layer_indexs);
    float *bn_values_copy = (float*)calloc(bn_total_count,sizeof(float));
    copy_cpu(bn_total_count,bn_values,1,bn_values_copy,1);
    sort(bn_values_copy,bn_values_copy + bn_total_count);
    int thresh_index = int(bn_total_count * prune_ratio);
    float thresh = bn_values_copy[thresh_index];
    cout << "pruning threshold is: " << thresh << endl;
    vector<layer_info> prune_layer_list;
    cout << "network slimming starting" << endl;
    float *prev_mask;
    int prev_channel_number;
    for(int i=0; i < net.n;++i)
    {
        if(!find_exist(remain_layer_indexs,i))
        {
            if (net.layers[i].batch_normalize){
                layer_info per_layer;
                per_layer.old_index = i;
                per_layer.type = net.layers[i].type;
                int kernel_mask_sum = 0;
                per_layer.old_channel_number = net.layers[i].c;
                int total_kernel = net.layers[i].out_c;
                float *prune_mask = (float*)calloc(total_kernel,sizeof(float));
                compare_array_with_thresh(prune_mask,net.layers[i].scales,thresh,total_kernel, &kernel_mask_sum);
                if(i ==0)
                {
                    float const_mask[] = {1,1,1};
                    int const_channel_number = 3;
                    per_layer.channel_remain_mask = const_mask;
                    per_layer.remain_channel_number = const_channel_number;

                }else{
                    per_layer.channel_remain_mask = prev_mask;
                    per_layer.remain_channel_number = prev_channel_number;
                }
                per_layer.kernel_remain_mask = prune_mask;
                per_layer.old_kernel_number = total_kernel;
                per_layer.remain_kernel_number = kernel_mask_sum;
                save_prune_conv_weights(net.layers[i],&per_layer);
                prune_layer_list.push_back(per_layer);
                prev_mask = prune_mask;
                prev_channel_number = kernel_mask_sum;
                layer nextlayer = net.layers[i+1];
                if(nextlayer.type == CONVOLUTIONAL){
                    if(nextlayer.batch_normalize){
                        float * reduce_sum = reduce_sum_spatial(nextlayer.weights,nextlayer.out_c,nextlayer.c,nextlayer.size);
                        float * absorbed_bias = abandoned_bias(per_layer.kernel_remain_mask,net.layers[i].biases,per_layer.old_kernel_number);
                        float * result = transform_mm(absorbed_bias,reduce_sum,nextlayer.out_c,nextlayer.c);
                        float * run_mean = net.layers[i+1].rolling_mean;
                        axpy_cpu(nextlayer.out_c,-1,result,1,run_mean,1);
                        copy_cpu(nextlayer.out_c,run_mean,1,net.layers[i+1].rolling_mean,1);
                    }else{
                        float * reduce_sum = reduce_sum_spatial(nextlayer.weights,nextlayer.out_c,nextlayer.c,nextlayer.size);
                        float * absorbed_bias = abandoned_bias(per_layer.kernel_remain_mask,net.layers[i].biases,per_layer.old_kernel_number);
                        float * result = transform_mm(absorbed_bias,reduce_sum,nextlayer.out_c,nextlayer.c);
                        float * bias = net.layers[i+1].biases;
                        axpy_cpu(nextlayer.out_c,1,result,1,bias,1);
                        copy_cpu(nextlayer.out_c,bias,1,net.layers[i+1].biases,1);
                    }
                }else if(nextlayer.type == MAXPOOL || nextlayer.type == AVGPOOL){
                    layer nnlayer = net.layers[i+2];
                    float * reduce_sum = reduce_sum_spatial(nnlayer.weights,nnlayer.out_c,nnlayer.c,nnlayer.size);
                    float * absorbed_bias = abandoned_bias(per_layer.kernel_remain_mask,net.layers[i].biases,per_layer.old_kernel_number);
                    float * result = transform_mm(absorbed_bias,reduce_sum,nnlayer.out_c,nnlayer.c);
                    if(nnlayer.batch_normalize){
                        float * run_mean = net.layers[i+2].rolling_mean;
                        axpy_cpu(nextlayer.out_c,-1,result,1,run_mean,1);
                        copy_cpu(nextlayer.out_c,run_mean,1,net.layers[i+2].rolling_mean,1);
                    }else{
                        float * bias = net.layers[i+2].biases;
                        axpy_cpu(nextlayer.out_c,1,result,1,bias,1);
                        copy_cpu(nextlayer.out_c,bias,1,net.layers[i+2].biases,1);
                    }
                }
                cout<< "layer_" << i << ":" << get_layer_string(per_layer.type) << " count of kernel is " << per_layer.old_kernel_number \
                   << "  count of pruned kernel is " << per_layer.old_kernel_number - kernel_mask_sum << endl;
            }
            else{
                layer_info per_layer;
                per_layer.old_index = i;
                per_layer.type = net.layers[i].type;
                if(per_layer.type == ROUTE){
                    init_value(&per_layer);
                    per_layer.input_layers = net.layers[i].input_layers;
                    int k;
                    cout << "count is " << net.layers[i].n << endl;
                    for(k= 0;k < net.layers[i].n;k++){
                        per_layer.remain_kernel_number += prune_layer_list[per_layer.input_layers[k]].remain_kernel_number;
                        per_layer.remain_channel_number += prune_layer_list[per_layer.input_layers[k]].remain_channel_number;
                        per_layer.old_channel_number += prune_layer_list[per_layer.input_layers[k]].old_channel_number;
                        per_layer.old_kernel_number += prune_layer_list[per_layer.input_layers[k]].old_kernel_number;
                    }
                    float *prune_kernel_mask = (float*)calloc(per_layer.old_kernel_number,sizeof(float));
                    int offset = 0, len = 0;
                    for(int k=0; k < net.layers[i].n;k++){
                        len = prune_layer_list[per_layer.input_layers[k]].old_kernel_number;
                        copy_cpu(len,prune_layer_list[per_layer.input_layers[k]].kernel_remain_mask,1,prune_kernel_mask + offset,1);
                        offset = len;
                    }
                    per_layer.kernel_remain_mask = prune_kernel_mask;
                    prune_layer_list.push_back(per_layer);
                    prev_channel_number = per_layer.remain_kernel_number;
                    prev_mask = per_layer.kernel_remain_mask;
                    cout<< "layer_" << i << ":" << get_layer_string(per_layer.type) << endl;
                }else if(per_layer.type == SHORTCUT){
                    int index = net.layers[i].index;
                    per_layer.old_kernel_number = prune_layer_list[index].old_kernel_number;
                    per_layer.remain_kernel_number = prune_layer_list[index].remain_kernel_number;
                    per_layer.old_channel_number = prune_layer_list[index].old_channel_number;
                    per_layer.remain_channel_number = prune_layer_list[index].remain_channel_number;
                    per_layer.kernel_remain_mask = prune_layer_list[index].kernel_remain_mask;
                    prune_layer_list.push_back(per_layer);
                    prev_channel_number = per_layer.remain_kernel_number;
                    prev_mask = per_layer.kernel_remain_mask;
                    cout<< "layer_" << i << ":" << get_layer_string(per_layer.type) << endl;
                }else if(per_layer.type == YOLO){
                    prune_layer_list.push_back(per_layer);
                    cout<< "layer_" << i << ":" << get_layer_string(per_layer.type) << endl;
                }else if(per_layer.type == MAXPOOL || per_layer.type == AVGPOOL || per_layer.type == UPSAMPLE){
                    per_layer.channel_remain_mask = prev_mask;
                    per_layer.remain_channel_number = prev_channel_number;
                    per_layer.old_kernel_number = net.layers[i].out_c;
                    per_layer.old_channel_number = net.layers[i].c;
                    per_layer.remain_kernel_number = prev_channel_number;
                    per_layer.kernel_remain_mask = prev_mask;
                    prune_layer_list.push_back(per_layer);
                    cout<< "layer_" << i << ":" << get_layer_string(per_layer.type) << endl;
                }else{
                    prune_layer_list.push_back(per_layer);
                    cout << "this layer is not supported yet" << endl;
                }
             }
        }
        else{
            layer_info per_layer;
            layer l = net.layers[i];
            per_layer.type = l.type;
            per_layer.old_index = i;
            if(per_layer.type == CONVOLUTIONAL){
                per_layer.channel_remain_mask = prev_mask;
                per_layer.remain_channel_number = prev_channel_number;
                per_layer.old_kernel_number = net.layers[i].out_c;
                per_layer.old_channel_number = net.layers[i].c;
                per_layer.remain_kernel_number = net.layers[i].out_c;
                float *prune_kernel_mask = (float*)calloc(per_layer.remain_kernel_number,sizeof(float));
                const_cpu(per_layer.remain_kernel_number,1.0,prune_kernel_mask,1);
                per_layer.kernel_remain_mask = prune_kernel_mask;
                save_prune_conv_weights(net.layers[i],&per_layer);
                prune_layer_list.push_back(per_layer);
                prev_mask = prune_kernel_mask;
                prev_channel_number = per_layer.remain_kernel_number;
            }
            cout<< "layer_" << i << ":" << get_layer_string(per_layer.type) << " count of kernel is " << per_layer.old_kernel_number \
               << "  count of pruned kernel is " << 0 << endl;
        }
    }
    cout << "start to write cfg file" << endl;
    write_cfg(prune_cfg, section,&net,prune_layer_list);
    cout << "save pruned cfg file to: " << prune_cfg << endl;
    cout << "start to write weights" << endl;
    FILE *fp = fopen(prune_weights,"wr");
    write_head2weight(fp,net);
    write_weights(fp,net, prune_layer_list);
    fclose(fp);
    cout << "save pruned weights file to: " << prune_weights << endl;
}

void run_prune(int argc, char **argv)
{
    if(argc < 2){
        fprintf(stderr, "usage: %s %s [cfg] [weights] [prune_rate]\n", argv[0], argv[1]);
        return;
    }
    char *cfg = argv[2];
    char *weights = argv[3];
    float prune_rate = find_float_arg(argc, argv, "-rate", .3);
    prune_yolov3(cfg,weights,prune_rate);
}
