
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
    int weight_width;
    int min_weight;
    int max_weight;

    std::string layer_name;
    bool pooling_enabled;
    int Timestamp;

    // // 定义各类数据在DRAM中的内存分配大小
    // unsigned int input_dram_size;  // MB
    // unsigned int weight_dram_size;  
    // unsigned int output_dram_size;
    // // unsigned int neuron_state_dram_size  这个值的大小和神经元阈值的大小有关，暂不设置 

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

    // 乒乓buffer，首先向input_buffer_0中写入数据，此时input_buffer_1是空的，因此没有操作。
    // 当向input_buffer_0写入完毕，通过逻辑操作使得接下来向input_buffer_1写入数据，与此同时其他模块可以从input_buffer_0中读出已经写入的数据。
    // 待input_buffer_1中写完，再次转换，重新向input_buffer_0中写入数据，同时其他模块从input_buffer_1中读出数据。
    PingPong_Buffer* ppbuf_input; 
    int* input_buffer_0;
    int* input_buffer_1;

    PingPong_Buffer* ppbuf_weight;
    int* weight_buffer_0;
    int* weight_buffer_1;

    // 建模双buffer
    PingPong_Buffer* ppbuf_input_arranged;
    int* input_arranged_buffer_0;
    int* input_arranged_buffer_1;

    // 建模片上buffer
    int* input_buffer;
    int* weight_buffer;
    int* output_buffer;
    int* neuron_state_buffer;
    int* output_buffer_cpu;
    int* neuron_state_buffer_cpu;

    int* on_chip_sram;
    int* output_regfile_pooling;
    int* output_regfile_pooling_cpu;

    // 建模一行卷积输出结果的存储
    int* output_regfile_conv;
    int* output_regfile_conv_cpu;
    int* neuron_state_regfile_conv;
    int* neuron_state_regfile_conv_cpu;

    // 建模输入数据重排序之后的内存，建模为多bank
    int numBanks; // bank个数，等于脉动阵列的行数
    int bankSize; // bank大小，等于filter size ： R*S*C
    std::vector<int> im2col_bank;
    std::vector<std::vector<int>> input_arranged;
    int* input_arranged_buffer;
    int* spikes; // 用于存储得到的input_arranegd数据，送到计算模块


    // 用于加padding情况下，判断当下取数据地址
    int num_retrieve_input_data;
    std::vector<int> skip_list;

    // 用于记录从片外DRAM取数据的信息
    std::vector<record> records;

    // 建模片外存储,存储完整的输入数据和权重数据，以及存储计算得到的中间数据和输出
    int* ifmap;
    int* filter;
    int* ofmap;
    int* ofmap_cpu;
    int* nfmap;
    int* nfmap_cpu;

    // DRAM中各类型数据存储的基地址
    uint64_t input_offset;
    uint64_t weight_offset;
    uint64_t output_offset;
    uint64_t neuron_state_offset;
    uint64_t addr_offset; // 一次可以从DRAM中读取多少个字节数据

    // 模拟器和DRAM的接口
    Fifo* read_request_fifo;
    Fifo* write_request_fifo;

    Dram* dram_instance;
    int length_bits; // DRAM突发一次的数据量，单位为bit


    Controller(Config stonne_cfg, std::vector<layer_topology> layers);
    ~Controller();

    // 初始化乒乓buffer
    void PingPongBuffer_Init(PingPong_Buffer* ppbuf, int* buffer_0, int* buffer_1);
    // 交换乒乓buffer
    void PingPongBuffer_Switch(PingPong_Buffer* ppbuf, int* buffer_0, int* buffer_1);

    std::tuple<int*, int*, int*, int*> runConv(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters);
    std::tuple<int*, int*, int*, int*> runConvandPooling(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters);
    std::tuple<int*, int*, int*, int*> runFC_0(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters);
    std::tuple<int*, int*, int*, int*> runFC_1(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters);
    std::tuple<int*, int*, int*, int*> runFC_2(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters);  // 双buffer

    std::tuple<int*, int*, int*, int*> runConv_DataFlow_0(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters); // 单buffer
    std::tuple<int*, int*, int*, int*> runConv_DataFlow_1(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters); // 乒乓buffer
    std::tuple<int*, int*, int*, int*> runConv_DataFlow_2(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters); 
    std::tuple<int*, int*, int*, int*> runConv_DataFlow_3(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters); 
    std::tuple<int*, int*, int*, int*> runConvandPooling_DataFlow_0(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters);
    std::tuple<int*, int*, int*, int*> runConvandPooling_DataFlow_1(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters);
    std::tuple<int*, int*, int*, int*> runConvandPooling_DataFlow_2(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters);
    //std::tuple<int*, int*, int*, int*> runFC_DataFlow(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters);

    void traverse();

    void run();

    // 与DRAM交互的函数，返回值是所用的周期
    int load_weight_data_ppbuffer(int* filter, Dram* dram_instance, int num_weight_obtained, int num_weight_data);
    int load_weight_data(int* filter, Dram* dram_instance, int num_weight_obtained, int num_weight_read_request);
    int load_input_data_step2_ppbuffer(int* ifmap, Dram* dram_instance, int j, layer_topology layer_parameters);
    int load_input_data_step1_ppbuffer(int layer_id, int* ifmap, Dram* dram_instance, int j, layer_topology layer_parameters);
    int load_input_data_step1_onebuffer(int* ifmap, Dram* dram_instance, int j, layer_topology layer_parameters);
    int store_output_and_neuronstate_data(int* ofmap, int* nfmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters); // conv
    int store_neuron_state(int* nfmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters); // conv_and_pooling
    int store_output(int* ofmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters);  // conv_and_pooling

    int load_input_data_fc(int* ifmap, Dram* dram_instance, int num_input_obtained, int num_input_data);
    int load_weight_data_fc(int* filter, Dram* dram_instance, int i, int j, layer_topology layer_parameters);
    int store_output_and_neuronstate_data_fc(int* ofmap, int* nfmap, Dram* dram_instance, int i, layer_topology layer_parameters);

    // 解耦函数
    int im2col(int start, int num, layer_topology layer_parameters);
    int im2col_0(int start, int num, layer_topology layer_parameters);
    int im2col_ppbuffer(int start, int num, layer_topology layer_parameters);
    int process_conv(int i, int j, int cols, layer_topology layer_parameters);
    int process_conv_1(int i, int j, int cols, layer_topology layer_parameters);
    int process_conv_2(int i, int j, int cols, layer_topology layer_parameters);
    int process_conv_3(int layer_id, int i, int j, int cols, layer_topology layer_parameters);
    int process_conv_and_pooling(int layer_id, int i, int j, int cols, int count_rows, layer_topology layer_parameters);
    int process_pooling(int i, int j, int cols, layer_topology layer_parameters);
    int process_fc(int i, int j, layer_topology layer_parameters);

    static void read_callback(uint64_t addr);
    static void write_callback(uint64_t addr);

    static int completed_reads;
    static int completed_writes;

};



#endif