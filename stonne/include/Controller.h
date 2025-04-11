
#ifndef __CONTROLLER__H
#define __CONTROLLER__H

#include "dramsim3.h"
#include "DRAMModel.h"
#include "STONNEModel.h"


class Controller{
public:

    Config stonne_cfg;
    std::vector<layer_topology> layers;

    unsigned int n_cycles;
    unsigned int n_conv;
    unsigned int n_pooling;
    unsigned int n_fc;

    unsigned long time_ms;
    unsigned long time_as;
    unsigned long time_mem;
    unsigned long time_update;
    unsigned long time_pooling;

    // 定义权重位宽
    unsigned int weight_width=4; 
    int min_weight = -(1 << (weight_width-1));
    int max_weight = (1 << (weight_width-1))-1;

    std::string layer_name;
    bool pooling_enabled;
    int Timestamp;
    
    // 定义各类数据在DRAM中的内存分配大小
    unsigned int input_dram_size = 2;  // MB
    unsigned int weight_dram_size = 3;  
    unsigned int output_dram_size = 2;
    // unsigned int neuron_state_dram_size  这个值的大小和神经元阈值的大小有关，暂不设置 

    // 片上buffer的大小
    unsigned int input_buffer_size;
    unsigned int weight_buffer_size;
    unsigned int output_buffer_size;
    unsigned int neuron_state_buffer_size;
    // 片上可存储的各参数的个数
    unsigned int num_input;
    unsigned int num_weight;
    unsigned int num_output;
    unsigned int num_neuron_state;
    // 建模片上buffer
    int* input_buffer;
    int* weight_buffer;
    int* output_buffer;
    int* neuron_state_buffer;
    int* output_buffer_cpu;
    int* neuron_state_buffer_cpu;

    // 建模输入数据重排序之后的内存，建模为多bank
    int numBanks; // bank个数，等于脉动阵列的行数
    int bankSize; // bank大小，等于filter size ： R*S*C
    std::vector<std::vector<int>> input_arranged;
    int * spikes; // 用于存储得到的input_arranegd数据，送到计算模块


    // 建模片外存储,存储完整的输入数据和权重数据，以及存储计算得到的中间数据和输出
    int* ifmap;
    int* filter;
    int* ofmap;
    int* ofmap_cpu;
    int* nfmap;
    int* nfmap_cpu;

    // 模拟器和DRAM的接口
    Fifo* read_request_fifo;
    Fifo* write_request_fifo;

    Dram* dram_instance;


    Controller(Config stonne_cfg, std::vector<layer_topology> layers);
    ~Controller();

    //void runDenseGEMMComand(Config stonne_cfg, std::string layer_namee, unsigned int MM, unsigned int NN, unsigned int KK, unsigned int TT_M, unsigned int TT_N, unsigned int TT_K);

    void traverse();

    void run();

    static void read_callback(uint64_t addr);
    static void write_callback(uint64_t addr);

    static int completed_reads;
    static int completed_writes;

};



#endif