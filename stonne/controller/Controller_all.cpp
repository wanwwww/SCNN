
#include "Controller.h"
#include "STONNEModel.h"
#include <math.h>
#include "testbench.h"
#include <cstring>
#include "types.h"
#include <vector>
#include <algorithm>
#include "MYPOOL.h"
#include "RequestPackage.h"

Controller::Controller(Config stonne_cfg, std::vector<layer_topology> layers){
    this->stonne_cfg = stonne_cfg;
    this->layers = layers;
    this->n_cycles = 0;
    this->n_conv = 0;
    this->n_pooling = 0;
    this->n_fc = 0;

    this->Timestamp = this->stonne_cfg.Timestamp;
    this->pooling_enabled = false;

    // DRAM中各类参数存储的空间
    this->input_dram_size = this->stonne_cfg.m_DRAMCfg.input_dram_size;
    this->weight_dram_size = this->stonne_cfg.m_DRAMCfg.weight_dram_size;
    this->output_dram_size = this->stonne_cfg.m_DRAMCfg.output_dram_size;

    // DRAM中一次存放：输入、权重、输出和神经元状态
    this->input_offset = 0x0000; // 输入数据在DRAM中的起始地址 （单位是字节）
    this->weight_offset = this->input_dram_size*1024*1024;
    this->output_offset = (this->input_dram_size + this->weight_dram_size) * 1024 *1024;
    this->neuron_state_offset = (this->input_offset + this->weight_dram_size + this->output_offset) * 1024 * 1024;
    this->addr_offset = 8; // 一次可以从DRAM中读取8个字节的数据

    // 实例化数组当作片上buffer
    this->input_buffer_size = this->stonne_cfg.m_BufferCfg.input_buffer_size;
    this->weight_buffer_size = this->stonne_cfg.m_BufferCfg.weight_buffer_size;
    this->output_buffer_size = this->stonne_cfg.m_BufferCfg.output_buffer_size;
    this->neuron_state_buffer_size = this->stonne_cfg.m_BufferCfg.neuron_state_buffer_size;

    // 片上buffer能够存储的各种数据的个数，但实际不需要存储这么多
    this->num_input = this->input_buffer_size*1024*8;  // 存储输入脉冲的个数
    this->num_weight = this->weight_buffer_size*1024*8/this->weight_width;  // 存储权重的个数
    this->num_output = this->output_buffer_size*1024*8;  // 存储输入脉冲的个数
    this->num_neuron_state = this->neuron_state_buffer_size*1024*8/(std::ceil(std::log2(this->stonne_cfg.V_th)));  // 存储膜电位的个数

    // this->input_buffer = new int[num_input];
    // this->weight_buffer = new int[num_weight];
    // this->output_buffer = new int[num_output];
    // this->neuron_state_buffer = new int[num_neuron_state];

    this->read_request_fifo = new Fifo(1);
    this->write_request_fifo = new Fifo(1);

}

Controller::~Controller(){
    delete read_request_fifo;
    delete write_request_fifo;
}

int Controller::completed_reads = 0;
int Controller::completed_writes = 0;

// 运行FC层时，从片外DRAM加载数据到片上
int Controller::load_input_data_fc(int* ifmap, Dram* dram_instance, int num_input_obtained, int num_input_data){
    int local_cycles = 0;

    int num_input_read_request = std::ceil(num_input_data / (float)dram_instance->dram->GetBusBits());
    int num_input_read_request_copy = num_input_read_request;
    int addr = this->input_offset + num_input_obtained/8; // 字节地址
    while(true){
        if(num_input_read_request != 0){
            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  // 读请求
            //std::cout<<"read weight addr is : "<<addr<<std::endl;
            this->read_request_fifo->push(read_request);
            addr += this->addr_offset;
            num_input_read_request--;
        }

        if(completed_reads == num_input_read_request_copy){  // 读请求处理完毕
            completed_reads = 0;
            // 模拟将片外数据写入片上buffer
            for(int i=0; i<num_input_data; i++){
                //this->input_buffer[i] = ifmap[num_input_obtained+i];  // 权重数据的读取顺序在DRAM中是连续读取的
                this->ppbuf_input->next_buffer[i] = ifmap[num_input_obtained+i];
            }
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}

int Controller::load_weight_data_fc(int* filter, Dram* dram_instance, int i, int j, layer_topology layer_parameters){
    int local_cycles = 0;

    // 输入输出层神经元个数
    int input_layer = layer_parameters.input_neuron;
    int output_layer = layer_parameters.output_neuron;

    int start_row = i*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    int end_row = std::min<int>(start_row+this->stonne_cfg.m_MSNetworkCfg.ms_rows, output_layer);
    int rows = end_row - start_row;

    int start_col = j*this->num_input;
    int end_col = std::min<int>(start_col+this->num_input, input_layer);
    int cols = end_col - start_col;

    int num_total_read_request = 0;
    bool only = true;
    while(true){
        while(only){
            for(int r=0; r<rows; r++){  // 遍历每一行，每一行内的数据在DRAM内是连续存储的
                int current_num_read_request = std::ceil(cols*this->weight_width /(float)dram_instance->dram->GetBusBits());
                num_total_read_request += current_num_read_request;

                int addr = this->weight_offset + ((start_row+r)*input_layer + start_col)/8;
                while(current_num_read_request!=0){
                    std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  
                    this->read_request_fifo->push(read_request);
                    current_num_read_request--;
                    addr += this->addr_offset;
                    
                    dram_instance->run();
                    local_cycles++;
                }

                // 模拟从DRAM中读取数据到片上buffer
                for(int num=0; num<cols; num++){
                    // this->weight_buffer[r*cols + num] = filter[(start_row+r)*input_layer + start_col + num];
                    this->ppbuf_weight->next_buffer[r*cols + num] = filter[(start_row+r)*input_layer + start_col + num];
                }
            }
            only = false;
        }

        if(completed_reads == num_total_read_request) {
            completed_reads = 0; 
            break;
        }

        this->dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}

// 从DRAM中加载权重数据的函数，返回值是所用的时钟周期，单buffer
int Controller::load_weight_data(int* filter, Dram* dram_instance, int num_weight_obtained, int num_weight_data){
    // num_weight_obtained ： 已经取出的数据数，用于计算读取地址
    // num_weight_data : 要加载的数据个数
    //std::cout<<"weight start_addr is : "<<this->weight_offset<<std::endl;
    int local_cycles = 0;
    int num_weight_read_request = std::ceil(num_weight_data*this->weight_width / (float)dram_instance->dram->GetBusBits());
    //std::cout<<"num_weight_read_request : "<<num_weight_read_request<<std::endl;
    int num_weight_read_request_copy = num_weight_read_request;
    int addr = this->weight_offset + (num_weight_obtained*this->weight_width)/8;  // 字节地址
    while(true){
        if(num_weight_read_request!=0){
            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  // 读请求
            //std::cout<<"read weight addr is : "<<addr<<std::endl;
            this->read_request_fifo->push(read_request);
            addr += this->addr_offset;
            num_weight_read_request--;
        } 

        if(completed_reads == num_weight_read_request_copy){  // 读请求处理完毕
            completed_reads = 0;
            // 模拟将片外数据写入片上buffer
            for(int j=0; j<num_weight_data; j++){
                this->weight_buffer[j] = filter[num_weight_obtained+j];  // 权重数据的读取顺序在DRAM中是连续读取的
            }
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}

// 双buffer
int Controller::load_weight_data_ppbuffer(int* filter, Dram* dram_instance, int num_weight_obtained, int num_weight_data){
    // num_weight_obtained ： 已经取出的数据数，用于计算读取地址
    // num_weight_data : 要加载的数据个数
    //std::cout<<"weight start_addr is : "<<this->weight_offset<<std::endl;
    int local_cycles = 0;
    int num_weight_read_request = std::ceil(num_weight_data*this->weight_width / (float)dram_instance->dram->GetBusBits());
    //std::cout<<"num_weight_read_request : "<<num_weight_read_request<<std::endl;
    int num_weight_read_request_copy = num_weight_read_request;
    int addr = this->weight_offset + (num_weight_obtained*this->weight_width)/8;  // 字节地址
    while(true){
        if(num_weight_read_request!=0){
            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  // 读请求
            //std::cout<<"read weight addr is : "<<addr<<std::endl;
            this->read_request_fifo->push(read_request);
            addr += this->addr_offset;
            num_weight_read_request--;
        } 

        if(completed_reads == num_weight_read_request_copy){  // 读请求处理完毕
            completed_reads = 0;
            // 模拟将片外数据写入片上buffer
            for(int j=0; j<num_weight_data; j++){
                this->ppbuf_weight->next_buffer[j] = filter[num_weight_obtained+j];  // 权重数据的读取顺序在DRAM中是连续读取的
            }
            break;
        }

        dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}

// 双buffer，buffer中都是真实的数据
int Controller::load_input_data_step2_ppbuffer(int* ifmap, Dram* dram_instance, int j, layer_topology layer_parameters){
    // j : 第几行卷积
    int local_cycles = 0;
    auto it = std::find(this->skip_list.begin(), this->skip_list.end(), j);
    if(it == this->skip_list.end()) { 
        this->num_retrieve_input_data++; // 记录取数据的位置，用于计算取数据的地址
    }

    // 一次取R*C*Y个输入数据，遍历每一个输出通道，因为每个输出通道内的数据是连续存储的
    int num_total_read_request = 0;
    bool only=true;
    while(true){
        while(only){
            for(int c=0; c<layer_parameters.C; c++){  // 遍历每个输入通道
                int current_num_read_request = std::ceil(layer_parameters.R*layer_parameters.Y / (float)dram_instance->dram->GetBusBits());
                num_total_read_request += current_num_read_request;
                
                int addr = (c*layer_parameters.X*layer_parameters.Y + this->num_retrieve_input_data*layer_parameters.Y) / 8;
                while(current_num_read_request!=0){
                    std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  
                    this->read_request_fifo->push(read_request);
                    current_num_read_request--;
                    addr += this->addr_offset;
                    
                    dram_instance->run();
                    local_cycles++;
                }

                // 模拟从DRAM中读取数据到片上buffer
                for(int num=0; num<layer_parameters.R*layer_parameters.Y; num++){
                    //this->input_buffer[c*layer_parameters.R*layer_parameters.Y + num] = ifmap[c*layer_parameters.X*layer_parameters.Y + this->num_retrieve_input_data*layer_parameters.Y + num];
                    this->ppbuf_input->next_buffer[c*layer_parameters.R*layer_parameters.Y + num] = ifmap[c*layer_parameters.X*layer_parameters.Y + this->num_retrieve_input_data*layer_parameters.Y + num];
                }
            }
            only = false;
        }

        if(completed_reads == num_total_read_request) {
            completed_reads = 0; 
            break;
        }

        this->dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}

// 单buffer，buffer中的数据是已经padding后的
int Controller::load_input_data_step1_onebuffer(int* ifmap, Dram* dram_instance, int j, layer_topology layer_parameters){
    // j : 第几行卷积
    int local_cycles = 0;
    auto it = std::find(this->skip_list.begin(), this->skip_list.end(), j);
    if(it == this->skip_list.end()) { 
        this->num_retrieve_input_data++; // 记录取数据的位置，用于计算取数据的地址
    }

    int num_rows = this->records[j].num_rows; // 取多少行数据
    int start_rows = this->records[j].start_rows; //起始行
    int add_0_above = this->records[j].add_0_above;
    int add_0_below = this->records[j].add_0_below;

    int Y_padded = layer_parameters.Y + 2*layer_parameters.P;

    // 一次取R*C*Y个输入数据，遍历每一个输出通道，因为每个输出通道内的数据是连续存储的
    int num_total_read_request = 0;
    bool only=true;
    while(true){
        while(only){
            for(int c=0; c<layer_parameters.C; c++){  // 遍历每个输入通道
                int current_num_read_request = std::ceil(num_rows*layer_parameters.Y / (float)dram_instance->dram->GetBusBits());
                num_total_read_request += current_num_read_request;
                
                int addr = (c*layer_parameters.X*layer_parameters.Y + start_rows*layer_parameters.Y) / 8;
                while(current_num_read_request!=0){
                    std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  
                    this->read_request_fifo->push(read_request);
                    current_num_read_request--;
                    addr += this->addr_offset;
                    
                    dram_instance->run();
                    local_cycles++;
                }

                // 模拟从DRAM中读取数据到片上buffer
                for(int num=0; num<add_0_above*Y_padded; num++){
                    this->input_buffer[c*layer_parameters.R*Y_padded + num] = 0;
                }

                for(int r=0; r<num_rows; r++){  // 遍历每一行，在每一行两端加P个0，中间是Y个真实数据
                    for(int num=0; num<layer_parameters.P; num++){
                        this->input_buffer[c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + num] = 0;
                    }
                    for(int num=0; num<layer_parameters.Y; num++){
                        this->input_buffer[c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + num] = ifmap[c*layer_parameters.X*layer_parameters.Y + (start_rows+r)*layer_parameters.Y +num]; 
                    }
                    for(int num=0; num<layer_parameters.P; num++){
                        this->input_buffer[c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + layer_parameters.Y + num] = 0;
                    }
                }

                for(int num=0; num<add_0_below*Y_padded; num++){
                    this->input_buffer[c*layer_parameters.R*Y_padded + (add_0_above+num_rows)*Y_padded + num] = 0;
                }
            }
            only = false;
        }

        if(completed_reads == num_total_read_request) {
            completed_reads = 0; 
            break;
        }

        this->dram_instance->run();
        local_cycles++;
    }

    return local_cycles;
}

// 双buffer，buffer中已经进行padding处理
int Controller::load_input_data_step1_ppbuffer(int layer_id, int* ifmap, Dram* dram_instance, int j, layer_topology layer_parameters){
    // j : 第几行卷积

    int local_cycles = 0;
    auto it = std::find(this->skip_list.begin(), this->skip_list.end(), j);
    if(it == this->skip_list.end()) { 
        this->num_retrieve_input_data++; // 记录取数据的位置，用于计算取数据的地址
    }

    int num_rows = this->records[j].num_rows; // 取多少行数据
    int start_rows = this->records[j].start_rows; //起始行
    int add_0_above = this->records[j].add_0_above;
    int add_0_below = this->records[j].add_0_below;

    if(layer_id == 8){
        std::cout<<"call load_input_data_step1_ppbuffer"<<std::endl;
        std::cout<<"num_rows : "<<num_rows<<std::endl;
        std::cout<<"start_rows : "<<start_rows<<std::endl;
        std::cout<<"add_0_above : "<<add_0_above<<std::endl;
        std::cout<<"add_0_below : "<<add_0_below<<std::endl;
    }

    int Y_padded = layer_parameters.Y + 2*layer_parameters.P;

    int input_buffer_size = layer_parameters.C*layer_parameters.R*Y_padded;
    int ifmap_size = layer_parameters.C * layer_parameters.X * layer_parameters.Y;

    // 一次取R*C*Y个输入数据，遍历每一个输出通道，因为每个输出通道内的数据是连续存储的
    int num_total_read_request = 0;
    bool only=true;
    while(true){
        while(only){
            for(int c=0; c<layer_parameters.C; c++){  // 遍历每个输入通道

                int current_num_read_request = std::ceil(num_rows*layer_parameters.Y / (float)dram_instance->dram->GetBusBits());
                num_total_read_request += current_num_read_request;
                
                int addr = (c*layer_parameters.X*layer_parameters.Y + start_rows*layer_parameters.Y) / 8;
                while(current_num_read_request!=0){
                    std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr,false);  
                    this->read_request_fifo->push(read_request);
                    current_num_read_request--;
                    addr += this->addr_offset;
                    
                    dram_instance->run();
                    local_cycles++;
                }

                // 模拟从DRAM中读取数据到片上buffer
                for(int num=0; num<add_0_above*Y_padded; num++){
                    this->ppbuf_input->next_buffer[c*layer_parameters.R*Y_padded + num] = 0;
                    assert((c*layer_parameters.R*Y_padded + num) < input_buffer_size);
                }

                for(int r=0; r<num_rows; r++){  // 遍历每一行，在每一行两端加P个0，中间是Y个真实数据
                    for(int num=0; num<layer_parameters.P; num++){
                        this->ppbuf_input->next_buffer[c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + num] = 0;
                        assert((c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + num) < input_buffer_size);
                    }
                    for(int num=0; num<layer_parameters.Y; num++){
                        this->ppbuf_input->next_buffer[c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + num] = ifmap[c*layer_parameters.X*layer_parameters.Y + (start_rows+r)*layer_parameters.Y +num]; 
                        assert((c*layer_parameters.X*layer_parameters.Y + (start_rows+r)*layer_parameters.Y +num)>=0 && (c*layer_parameters.X*layer_parameters.Y + (start_rows+r)*layer_parameters.Y +num) < ifmap_size);
                        assert((c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + num)>=0 && (c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + num) < input_buffer_size);
                        // std::cout<<"load data from in DRAM addr "<<(c*layer_parameters.X*layer_parameters.Y + (start_rows+r)*layer_parameters.Y +num)<<"  to input_buffer in addr "<<(c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + num)<<std::endl;
                    }
                    for(int num=0; num<layer_parameters.P; num++){
                        this->ppbuf_input->next_buffer[c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + layer_parameters.Y + num] = 0;
                        assert((c*layer_parameters.R*Y_padded + (add_0_above+r)*Y_padded + layer_parameters.P + layer_parameters.Y + num) < input_buffer_size);
                    }
                }

                for(int num=0; num<add_0_below*Y_padded; num++){
                    this->ppbuf_input->next_buffer[c*layer_parameters.R*Y_padded + (add_0_above+num_rows)*Y_padded + num] = 0;
                    assert((c*layer_parameters.R*Y_padded + (add_0_above+num_rows)*Y_padded + num) < input_buffer_size);
                }
            }
            only = false;

            if(layer_id == 5 && j==1){
                std::cout<<"************ Request sending ends ***************"<<std::endl;
                std::cout<<"   num_total_read_request : "<<num_total_read_request<<std::endl;
            }

        }

        // if(layer_id == 5 && j==1){
        //     std::cout<<"complete_reads : "<<completed_reads<<"      local_cycles : "<<local_cycles<<std::endl;
        // }


        if(completed_reads == num_total_read_request) {
            completed_reads = 0; 
            break;
        }

        // this->dram_instance->run_1(layer_id, j, completed_reads);
        this->dram_instance->run();

        // if(layer_id == 5 && j==1){
        //     std::cout<<"this->dram_instance->run()  end "<<std::endl;
        // }

        local_cycles++;
    }

    // if(layer_id == 5 && j==1){
    //     std::cout<<"************ load over ***************"<<std::endl;
    // }

    // std::cout<<"num_total_read_request : "<<num_total_read_request<<"================================================"<<std::endl;
    return local_cycles;
}

// 将神经元状态写入DRAM，在执行层convandpooling时调用
int Controller::store_neuron_state(int* nfmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters){
    // i : 权重循环
    // j ： 输入数据循环
    // cols : 输出通道数
    int local_cycles = 0;
    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    int num_total_write_request = 0;
    bool only = true;
    while(true){
        while(only){
            for(int p=0; p<cols; p++){
                int current_num_neuron_state_write_request = std::ceil(Y_*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                num_total_write_request += current_num_neuron_state_write_request;

                int write_addr_neuron_state = this->neuron_state_offset + ((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                while(current_num_neuron_state_write_request!=0){
                    std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                    this->write_request_fifo->push(write_request);
                    write_addr_neuron_state += this->addr_offset;

                    this->dram_instance->run();
                    local_cycles++;  // 一个周期发送一个DRAM写请求
                    current_num_neuron_state_write_request--;
                }
                // 模拟将片上buffer中的数据写入DRAM
                for(int q=0; q<Y_; q++){
                    nfmap[((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_) + q] = this->neuron_state_buffer[p*Y_ + q];
                }
            }
            only = false;
        }

        if(completed_writes == num_total_write_request) {
            completed_writes = 0; 
            break;
        }

        this->dram_instance->run();
        local_cycles++;
    }
    return local_cycles;
} 

// 将池化结果写入DRAM，在执行层convandpooling时调用
int Controller::store_output(int* ofmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters){
    // X_pool = (j-1)/2  当前输出行（池化结果）在输出特征图的第几行
    int local_cycles = 0;
    int X_pool = (j-1)/2;
    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    int num_total_write_request = 0;
    bool only = true;

    // std::cout<<"cols : "<<cols<<std::endl;
    // std::cout<<"Y_/2 : "<<Y_/2<<std::endl;

    while(true){
        while(only){
            for(int p=0; p<cols; p++){  // 遍历每个输出通道，将每个输出通道内的一行输出数据存储到DRAM中
                //std::cout<<"cols : "<<p<<std::endl;
                int current_num_output_write_request = std::ceil((Y_/2) / (float)dram_instance->dram->GetBusBits());
                num_total_write_request += current_num_output_write_request;
                
                // 往DRAM中写的基地址
                int write_addr_output = this->output_offset + (i*this->stonne_cfg.m_MSNetworkCfg.ms_cols*X_*Y_/4 + p*X_*Y_/4 + X_pool*Y_/2) / 8;
                while(current_num_output_write_request!=0){
                    std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                    this->write_request_fifo->push(write_request);
                    write_addr_output += this->addr_offset;
                    current_num_output_write_request--;

                    this->dram_instance->run();
                    local_cycles++; // 一个周期发送一个内存请求事务
                }
                // 模拟将output_regs中的数据写入DRAM，一行有Y_/2个数据
                for(int q=0; q<Y_/2; q++){
                    //std::cout<<q<<std::endl;
                    ofmap[(i*this->stonne_cfg.m_MSNetworkCfg.ms_cols*X_*Y_/4 + p*X_*Y_/4 + X_pool*Y_/2) + q] = this->output_buffer[p*Y_/2 + q];
                    // std::cout<<"write output_buffer addr   "<<(p*Y_/2 + q)<<"   to DRAM addr "<<((i*this->stonne_cfg.m_MSNetworkCfg.ms_cols*X_*Y_/4 + p*X_*Y_/4 + X_pool*Y_/2) + q)<<std::endl;
                }
            }
            only = false;
        }
        if(completed_writes == num_total_write_request) {
            completed_writes = 0; 
            break;
        }
        this->dram_instance->run();
        local_cycles++;
    }

    // std::cout<<"below is pooling result : "<<std::endl;
    // std::cout<<"X_pool : "<<X_pool<<std::endl;
    // for(int k=0; k<cols; k++){
    //     std::cout<<"output channel : "<<k<<std::endl;
    //     for(int p=0; p<Y_/2; p++){
    //         std::cout<<this->output_buffer[k*Y_/2 + p]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    return local_cycles;
}

// 将输出和神经元状态数据写入DRAM，在执行层conv时调用
int Controller::store_output_and_neuronstate_data(int* ofmap, int* nfmap, Dram* dram_instance, int i, int j, int cols, layer_topology layer_parameters){
    // i : 权重循环
    // j : 输入数据循环
    // cols : 输出通道数
    int local_cycles = 0;
    // 将当前行的输出结果写入DRAM
    // 即将this->output_buffer和this->neuron_state_buffer的数据写入DRAM
    
    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;

    int num_total_write_request = 0;
    bool only = true;
    while(true){
        while(only){
            for(int p=0; p<cols; p++){ // 遍历每一个输出通道，每个通道内有一行数据
                int current_num_output_write_request = std::ceil(Y_ / (float)dram_instance->dram->GetBusBits());
                int current_num_neuron_state_write_request = std::ceil(Y_*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                int current_num_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                num_total_write_request += current_num_total_write_request;

                int write_addr_output = this->output_offset + ((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_) / 8;
                int write_addr_neuron_state = this->neuron_state_offset + ((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                
                while(current_num_total_write_request!=0){
                    if(current_num_output_write_request!=0){
                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                        this->write_request_fifo->push(write_request);
                        write_addr_output += this->addr_offset;
                        current_num_output_write_request--;
                    } else if(current_num_neuron_state_write_request!=0){
                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                        this->write_request_fifo->push(write_request);
                        write_addr_neuron_state += this->addr_offset;
                        current_num_neuron_state_write_request--;
                    }
                    this->dram_instance->run();
                    local_cycles++;
                    current_num_total_write_request--;
                }
                // 模拟将片上buffer中的数据写入DRAM
                for(int q=0; q<Y_; q++){
                    ofmap[((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_) + q] = this->output_buffer[p*Y_ + q];
                    nfmap[((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_) + q] = this->neuron_state_buffer[p*Y_ + q];
                }
            }
            only = false;
        }

        if(completed_writes == num_total_write_request) {
            completed_writes = 0; 
            break;
        }

        this->dram_instance->run();
        local_cycles++;
    }
    return local_cycles;
}

// 将全连接层的输出和神经元状态写入DRAM
int Controller::store_output_and_neuronstate_data_fc(int* ofmap, int* nfmap, Dram* dram_instance, int i, layer_topology layer_parameters){
    // i : 权重循环
    int local_cycles = 0;

    int output_layer = layer_parameters.output_neuron;

    int start = i*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    int end = std::min<int>(start+this->stonne_cfg.m_MSNetworkCfg.ms_rows, output_layer);
    int num = end - start;

    int num_output_write_request = std::ceil(num / (float)dram_instance->dram->GetBusBits());
    int num_neuron_state_write_request = std::ceil(num*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
    int num_total_write_request = num_output_write_request + num_neuron_state_write_request;
    int num_total_write_request_copy = num_total_write_request;

    int addr_output = this->output_offset + start/8;
    int addr_neuron_state = this->neuron_state_offset + start*std::ceil(std::log2(this->stonne_cfg.V_th))/8;
    while(true){
        while(num_total_write_request != 0){
            if(num_output_write_request != 0){
                std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr_output,true);  // 写请求
                this->write_request_fifo->push(read_request);
                addr_output += this->addr_offset;
                num_output_write_request--;
            } else if(num_neuron_state_write_request != 0){
                std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(addr_neuron_state,true);  // 写请求
                this->write_request_fifo->push(read_request);
                addr_neuron_state += this->addr_offset;
                num_neuron_state_write_request--;
            }
            num_total_write_request--;
            this->dram_instance->run();
            local_cycles++;
        }

        if(completed_writes == num_total_write_request_copy) {
            completed_writes = 0; 
            break;
        }

        this->dram_instance->run();
        local_cycles++;
    }

    // 模拟将数据写入DRAM
    for(int k=0; k<num; k++){
        ofmap[start + k] = this->output_buffer[k];
        nfmap[start + k] = this->neuron_state_buffer[k];
    }

    // std::cout<<"write output[0] : "<<this->output_buffer[0]<<std::endl;
    // std::cout<<"write neuron[0] : "<<this->neuron_state_buffer[0]<<std::endl;

    return local_cycles;
}

// 执行全连接的计算
int Controller::process_fc(int i, int j, layer_topology layer_parameters){
    int local_cycles = 0;
    
    // 输入输出层神经元个数
    int input_layer = layer_parameters.input_neuron;
    int output_layer = layer_parameters.output_neuron;
    
    int start_row = i*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    int end_row = std::min<int>(start_row+this->stonne_cfg.m_MSNetworkCfg.ms_rows, output_layer);
    int rows = end_row - start_row;

    int start_col = j*this->num_input;
    int end_col = std::min<int>(start_col+this->num_input, input_layer);
    int cols = end_col - start_col;

    Stonne* stonne_instance = new Stonne(stonne_cfg);
    // matrixMultiply(rows,cols,1, this->weight_buffer, this->input_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
    matrixMultiply(rows,cols,1, this->ppbuf_weight->current_buffer, this->ppbuf_input->current_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
    
    // stonne_instance->loadDenseGEMM(layer_name, 1, cols, rows, this->weight_buffer, this->input_buffer, this->output_buffer, this->neuron_state_buffer, CNN_DATAFLOW);
    stonne_instance->loadDenseGEMM(layer_name, 1, cols, rows, this->ppbuf_weight->current_buffer, this->ppbuf_input->current_buffer, this->output_buffer, this->neuron_state_buffer, CNN_DATAFLOW);
    //std::cout<<"debug"<<std::endl;
    stonne_instance->loadGEMMTile(1, 1,rows);
    stonne_instance->run();

    local_cycles += stonne_instance->n_cycles;

    // 对当前tile的计算结果进行验证
    for(int num=0; num<rows; num++){
        float difference = fabs(this->output_buffer[num] - this->output_buffer_cpu[num]) + fabs(this->neuron_state_buffer[num] - this->neuron_state_buffer_cpu[num]);
        if(difference>0){
            std::cout << "ERROR position " << num <<  ": Value ofmap simulator: " << this->output_buffer[num] << ". Value ofmap CPU: " << this->output_buffer_cpu[num] << std::endl;
            std::cout << "ERROR position " << num <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[num] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[num] << std::endl;
            std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
            assert(false);
        }
    }

    // std::cout<<"output[0] : "<<this->output_buffer[0]<<std::endl;
    // std::cout<<"neuron_state[0] : "<<this->neuron_state_buffer[0]<<std::endl;

    std::cout << "\033[1;32mTest passed correctly \033[0m" << std::endl << std::endl;

    return local_cycles;
}

// 执行一行卷积，input_buffer中的数据都是真实的数据，需要im2col单元实现加padding逻辑，output寄存器文件的大小是一行的输出大小
int Controller::process_conv(int i, int j, int cols, layer_topology layer_parameters){
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;

    // 片外数据读取到片上buffer之后，对输入数据进行重排
    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;

    std::vector<int> im2col_bank(this->bankSize,0);
    int add_0_above; // 考虑到池化，需要补零
    int add_0_below;
    if(j>=0 && j<P){
        add_0_above = P - j;
        add_0_below = 0; 
    } else if(j<=(X_-1) && j>(X_-1-P)){
        add_0_above = 0;
        add_0_below = j - (X_-1-P);
    } else {
        add_0_above = 0; 
        add_0_below = 0;
    }

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();

    // 对input_buffer的数据进行排序，然后进行计算
    int num_tile = 0;
    int rows = 0;
    for(int k=0; k<Y_; k++){
        if(k==0){  // 第一个卷积窗口基地址计算不同，单独处理
            for(int c=0; c<C; c++){
                // 1. 上方补零，补add_0_above行0
                for(int num=0; num<add_0_above*S; num++){
                    im2col_bank[c*R*S + num] = 0;
                }

                // 2. 取数据
                for(int r=add_0_above; r<(R-add_0_below); r++){
                    for(int q=0; q<P; q++){  // 考虑池化，左侧P列0
                        im2col_bank[c*R*S + r*S + q] = 0;
                    }
                    for(int q=P; q<S; q++){ // 取真实的数据
                        //im2col_bank[c*R*S + r*S + q] = this->input_buffer[c*R*Y + (r+add_0_below-add_0_above)*Y +(q-P)];
                        im2col_bank[c*R*S + r*S + q] = this->ppbuf_input->current_buffer[c*R*Y + (r+add_0_below-add_0_above)*Y +(q-P)];
                        local_cycles++;
                    }
                }

                // 3. 下方补零，补add_0_below行0
                for(int num=0; num<add_0_below*S; num++){
                    im2col_bank[c*R*S + (R-add_0_below)*S + num] = 0;
                }
            }

            this->input_arranged.push_back(im2col_bank);
            rows++;
        } else {
            if(k >= Y_-P){  // 移位和右侧补0
                for(int c=0; c<C; c++){
                    for(int r=0; r<R; r++){
                        for(int q=0; q<(S-1); q++){  // 移位
                            im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                        }
                        im2col_bank[c*R*S + r*S + (S-1)] = 0;
                    }
                }
            } else {  // 移位和取数据
                for(int c=0; c<C; c++){
                    for(int num=0; num<add_0_above*S; num++){
                        im2col_bank[c*R*S + num] = 0;
                    }

                    // 2. 取数据
                    for(int r=add_0_above; r<(R-add_0_below); r++){
                        for(int q=0; q<(S-1); q++){
                            im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                        }
                        // 取数据
                        //im2col_bank[c*R*S + r*S + (S-1)] = this->input_buffer[c*R*Y + (r+add_0_below-add_0_above)*Y + (S-P+k-1)];
                        im2col_bank[c*R*S + r*S + (S-1)] = this->ppbuf_input->current_buffer[c*R*Y + (r+add_0_below-add_0_above)*Y + (S-P+k-1)];
                        local_cycles++;
                    }

                    // 3. 下方补零，补add_0_below行0
                    for(int num=0; num<add_0_below*S; num++){
                        im2col_bank[c*R*S + (R-add_0_below)*S + num]=0;
                    }
                }
            }

            this->input_arranged.push_back(im2col_bank);
            rows++;

            if(rows==this->numBanks || k == Y_-1){
                // std::cout<<"rows : "<<rows<<std::endl;
                // std::cout<<"cols : "<<cols<<std::endl;
                // std::cout<<"num_tile : "<<num_tile<<std::endl;

                // std::cout<<"first row data is : ******************"<<std::endl;
                // for(int num=0; num<this->bankSize; num++){
                //     std::cout<<this->input_arranged[0][num]<<"  ";
                // }
                // std::cout<<std::endl;

                // std::cout<<"first col data is : ******************"<<std::endl;
                // for(int num=0; num<this->bankSize; num++){
                //     std::cout<<this->weight_buffer[num]<<"  ";
                // }
                // std::cout<<std::endl;


                // 将this->input_arranged转化为数组
                int total = rows * this->bankSize;
                this->spikes = new int[total];
                int index = 0;
                for(const auto& row : input_arranged){
                    for(int val : row){
                        this->spikes[index++] = val;
                    }
                }

                // // 根据需要例化当前tile的output buffer 和 neuron_state buffer
                // unsigned int current_num_output_buffer_need = rows * cols;
                // unsigned int current_num_neuron_state_buffer_need = rows * cols;
                // int* current_tile_output_buffer = new int[current_num_output_buffer_need]();
                // int* current_tile_output_buffer_cpu = new int[current_num_output_buffer_need]();
                // int* current_tile_neuron_state_buffer = new int[current_num_neuron_state_buffer_need]();
                // int* current_tile_neuron_state_buffer_cpu = new int[current_num_neuron_state_buffer_need]();

                //std::cout<<"begin cpu compute"<<std::endl;
                matrixMultiply_new(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
                //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                //std::cout<<"begin load layer"<<std::endl;
                stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, rows, this->spikes, this->weight_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
                //std::cout<<"begin load tile"<<std::endl;
                stonne_instance->loadGEMMTile(cols,1,rows);
                //std::cout<<"begin run"<<std::endl;
                stonne_instance->mem->reset(); // 重置信号
                stonne_instance->asnet->resetSignals();
                stonne_instance->n_cycles = 0;
                stonne_instance->run();

                local_cycles += stonne_instance->n_cycles;

                // std::cout<<"addr : 0"<<std::endl;
                // std::cout<<this->output_regfile_conv[0]<<std::endl;
                // std::cout<<this->neuron_state_regfile_conv[0]<<std::endl;

                // std::cout<<"addr : 96"<<std::endl;
                // std::cout<<this->output_regfile_conv[96]<<std::endl;
                // std::cout<<this->neuron_state_regfile_conv[96]<<std::endl;

                // std::cout<<this->output_regfile_conv[256]<<std::endl;
                // std::cout<<this->neuron_state_regfile_conv[256]<<std::endl;

                // 对当前tile计算结果进行验证
                //std::cout<<"Test : Y_*cols = "<<Y_*cols<<std::endl;
                for(int m = 0; m<Y_ * cols; m++){
                    float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
                    if(difference>0){
                        std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                        std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                        std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                        assert(false);
                    }
                }
                // std::cout<<"------------------"<<std::endl;
                // std::cout<<"output : "<<this->output_regfile_conv[72]<<std::endl;
                // std::cout<<"neuron_state : "<<this->neuron_state_regfile_conv[72]<<std::endl;

                //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                // 重置数据，开始下一tile计算
                rows = 0;
                num_tile++;
                this->input_arranged.clear();
                //delete stonne_instance;
                delete[] this->spikes;
                // delete[] current_tile_output_buffer;
                // delete[] current_tile_output_buffer_cpu;
                // delete[] current_tile_neuron_state_buffer;
                // delete[] current_tile_neuron_state_buffer_cpu;
            }
        }
    }
    //std::cout<<"A convolutional layer has been processed~"<<std::endl;
    // 将当前tile的输出结果累积到this->output_buffer和this->neuron_state_buffer
    // 遍历每一列
    for(int p=0; p<cols; p++){
        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            // if(q==12){
            //     std::cout<<"Get the data address : "<<q*cols+p<<std::endl;
            // }
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期

        // std::cout<<"***************************"<<std::endl;
        // std::cout<<"output : "<<packed_col_output[12]<<std::endl;
        // std::cout<<"neuron_state : "<<packed_col_neuron_state[12]<<std::endl;
        // std::cout<<"***************************"<<std::endl;
        
        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            this->output_buffer[p*Y_ + q] = packed_col_output[q];
            this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
        }
    }

    delete stonne_instance;

    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;

    return local_cycles;
}

// 执行一层卷积，input_buffer中已经加了padding，
int Controller::process_conv_1(int i, int j, int cols, layer_topology layer_parameters){
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;

    int Y_padded = Y + 2*P;

    // 片外数据读取到片上buffer之后，对输入数据进行重排
    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;

    std::vector<int> im2col_bank(this->bankSize,0);
    // int add_0_above; // 考虑到池化，需要补零
    // int add_0_below;
    // if(j>=0 && j<P){
    //     add_0_above = P - j;
    //     add_0_below = 0; 
    // } else if(j<=(X_-1) && j>(X_-1-P)){
    //     add_0_above = 0;
    //     add_0_below = j - (X_-1-P);
    // } else {
    //     add_0_above = 0; 
    //     add_0_below = 0;
    // }

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();

    // 对input_buffer的数据进行排序，然后进行计算
    int num_tile = 0;
    int rows = 0;
    for(int k=0; k<Y_; k++){
        if(k==0){  // 第一个卷积窗口基地址计算不同，单独处理
            for(int c=0; c<C; c++){
                for(int r=0; r<R; r++){
                    for(int q=0; q<S; q++){ // 取真实的数据
                        //im2col_bank[c*R*S + r*S + q] = this->input_buffer[c*R*Y + (r+add_0_below-add_0_above)*Y +(q-P)];
                        im2col_bank[c*R*S + r*S + q] = this->ppbuf_input->current_buffer[c*R*Y_padded + r*Y_padded + q];
                        local_cycles++;
                    }
                }
            }
            this->input_arranged.push_back(im2col_bank);
            rows++;
        } else {
            for(int c=0; c<C; c++){
                for(int r=0; r<R; r++){
                    // 移位
                    for(int q=0; q<S-1; q++){
                        im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                    }
                    // 取数据
                    //im2col_bank[c*R*S + r*S + (S-1)] = this->input_buffer[c*R*Y + (r+add_0_below-add_0_above)*Y + (S-P+k-1)];
                    im2col_bank[c*R*S + r*S + (S-1)] = this->ppbuf_input->current_buffer[c*R*Y_padded + r*Y_padded + (S+(k-1))];
                    local_cycles++;
                }
            }
            this->input_arranged.push_back(im2col_bank);
            rows++;

            if(rows==this->numBanks || k == Y_-1){
                // std::cout<<"rows : "<<rows<<std::endl;
                // std::cout<<"num_tile : "<<num_tile<<std::endl;

                // std::cout<<"first row data is : ******************"<<std::endl;
                // for(int num=0; num<this->bankSize; num++){
                //     std::cout<<this->input_arranged[0][num]<<"  ";
                // }
                // std::cout<<std::endl;

                //std::cout<<"first col data is : ******************"<<std::endl;
                // for(int num=0; num<this->bankSize; num++){
                //     std::cout<<this->weight_buffer[num]<<"  ";
                // }
                //std::cout<<std::endl;


                // 将this->input_arranged转化为数组
                int total = rows * this->bankSize;
                this->spikes = new int[total];
                int index = 0;
                for(const auto& row : input_arranged){
                    for(int val : row){
                        this->spikes[index++] = val;
                    }
                }

                //std::cout<<"begin cpu compute"<<std::endl;
                matrixMultiply_new(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
                //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                //std::cout<<"begin load layer"<<std::endl;
                stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, rows, this->spikes, this->weight_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
                //std::cout<<"begin load tile"<<std::endl;
                stonne_instance->loadGEMMTile(cols,1,rows);
                //std::cout<<"begin run"<<std::endl;
                stonne_instance->mem->reset(); // 重置信号
                stonne_instance->asnet->resetSignals();
                stonne_instance->n_cycles = 0;
                stonne_instance->run();

                local_cycles += stonne_instance->n_cycles;

                // 对当前tile计算结果进行验证
                for(int m = 0; m<Y_ * cols; m++){
                    float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
                    if(difference>0){
                        std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                        std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                        std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                        assert(false);
                    }
                }
                std::cout<<std::endl;
                std::cout<<"cycles : "<<local_cycles<<std::endl;
                std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                // 重置数据，开始下一tile计算
                rows = 0;
                num_tile++;
                this->input_arranged.clear();
                delete[] this->spikes;
            }
        }
    }
    //std::cout<<"A convolutional layer has been processed~"<<std::endl;
    // 将当前tile的输出结果累积到this->output_buffer和this->neuron_state_buffer
    // 遍历每一列
    for(int p=0; p<cols; p++){
        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期
        //std::cout<<packed_col_neuron_state[0]<<std::endl;
        
        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            this->output_buffer[p*Y_ + q] = packed_col_output[q];
            this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
        }
    }

    delete stonne_instance;

    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;

    return local_cycles;
}

// 将image to colums功能模块化，纯单buffer
int Controller::im2col_0(int start, int num, layer_topology layer_parameters){
    // start : 起始的窗口
    // num ： 一共有多少个窗口
    // std::cout<<"start : "<<start<<std::endl;
    // std::cout<<"num : "<<num<<std::endl;
    // std::cout<<"The begin sorted data is : "<<std::endl;
    // for(int p=0; p<this->bankSize; p++){
    //     std::cout<<this->im2col_bank[p]<<"  ";
    // }
    // std::cout<<"--------------------------"<<std::endl;

    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;

    // int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    // this->input_arranged_buffer = new int[input_arranged_buffer_size];
    
    int flag = 0;
    if(start==0){
        //std::cout<<"debug"<<std::endl;
        for(int c=0; c<C; c++){
            for(int r=0; r<R; r++){
                for(int q=0; q<S; q++){
                    this->im2col_bank[c*R*S + r*S + q] = this->input_buffer[c*R*Y_padded + r*Y_padded + q];
                    local_cycles++;
                }
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        //this->input_arranged.push_back(im2col_bank);

        // 将排序好的数据写入待计算buffer
        for(int i=0; i<this->bankSize; i++){
            this->input_arranged_buffer[i] = this->im2col_bank[i];
        }
        flag = 1;
        num--;  // 从窗口个数 -1
        start++;  // 起始窗口 +1
    }

    for(int i=0; i<num; i++){ 
        for(int c=0; c<C; c++){
            for(int r=0; r<R; r++){
                // 移位
                for(int q=0; q<S-1; q++){
                    this->im2col_bank[c*R*S + r*S + q] = this->im2col_bank[c*R*S + r*S + q + 1];
                }
                // 取数据
                this->im2col_bank[c*R*S + r*S + (S-1)] = this->input_buffer[c*R*Y_padded + r*Y_padded + (S+(start+i-1))];
                local_cycles++;
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        //this->input_arranged.push_back(im2col_bank);

        for(int j=0; j<this->bankSize; j++){
            this->input_arranged_buffer[(flag+i)*this->bankSize + j] = this->im2col_bank[j];
        }
    }

    return local_cycles;
}

// 将image to columns功能模块化，输入为双buffer
int Controller::im2col(int start, int num, layer_topology layer_parameters){
    // start : 起始的窗口
    // num ： 一共有多少个窗口
    // std::cout<<"start : "<<start<<std::endl;
    // std::cout<<"num : "<<num<<std::endl;
    // std::cout<<"The begin sorted data is : "<<std::endl;
    // for(int p=0; p<this->bankSize; p++){
    //     std::cout<<this->im2col_bank[p]<<"  ";
    // }
    // std::cout<<"--------------------------"<<std::endl;

    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;

    // int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    // this->input_arranged_buffer = new int[input_arranged_buffer_size];
    
    int flag = 0;
    if(start==0){
        //std::cout<<"debug"<<std::endl;
        for(int c=0; c<C; c++){
            for(int r=0; r<R; r++){
                for(int q=0; q<S; q++){
                    this->im2col_bank[c*R*S + r*S + q] = this->ppbuf_input->current_buffer[c*R*Y_padded + r*Y_padded + q];
                    local_cycles++;
                }
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        //this->input_arranged.push_back(im2col_bank);

        // 将排序好的数据写入待计算buffer
        for(int i=0; i<this->bankSize; i++){
            this->input_arranged_buffer[i] = this->im2col_bank[i];
        }
        flag = 1;
        num--;  // 从窗口个数 -1
        start++;  // 起始窗口 +1
    }

    for(int i=0; i<num; i++){ 
        for(int c=0; c<C; c++){
            for(int r=0; r<R; r++){
                // 移位
                for(int q=0; q<S-1; q++){
                    this->im2col_bank[c*R*S + r*S + q] = this->im2col_bank[c*R*S + r*S + q + 1];
                }
                // 取数据
                this->im2col_bank[c*R*S + r*S + (S-1)] = this->ppbuf_input->current_buffer[c*R*Y_padded + r*Y_padded + (S+(start+i-1))];
                local_cycles++;
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        //this->input_arranged.push_back(im2col_bank);

        for(int j=0; j<this->bankSize; j++){
            this->input_arranged_buffer[(flag+i)*this->bankSize + j] = this->im2col_bank[j];
        }
    }

    return local_cycles;
}

// 输入为双buffer，input_arranged_buffer为双buffer
int Controller::im2col_ppbuffer(int start, int num, layer_topology layer_parameters){
    // start : 起始的窗口
    // num ： 一共有多少个窗口
    // std::cout<<"start : "<<start<<std::endl;
    // std::cout<<"num : "<<num<<std::endl;
    // std::cout<<"The begin sorted data is : "<<std::endl;
    // for(int p=0; p<this->bankSize; p++){
    //     std::cout<<this->im2col_bank[p]<<"  ";
    // }
    // std::cout<<"--------------------------"<<std::endl;

    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;
    int Y_padded = Y + 2*P;

    // int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    // this->input_arranged_buffer = new int[input_arranged_buffer_size];
    // std::cout<<"input buffer : "<<this->ppbuf_input->current_buffer[Y_padded+1]<<std::endl;
    int flag = 0;
    if(start==0){
        //std::cout<<"debug"<<std::endl;
        for(int c=0; c<C; c++){
            for(int r=0; r<R; r++){
                for(int q=0; q<S; q++){
                    this->im2col_bank[c*R*S + r*S + q] = this->ppbuf_input->current_buffer[c*R*Y_padded + r*Y_padded + q];
                    local_cycles++;
                }
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<im2col_bank[p]<<" ";
        // }
        // std::cout<<std::endl;

        //this->input_arranged.push_back(im2col_bank);

        // 将排序好的数据写入待计算buffer
        for(int i=0; i<this->bankSize; i++){
            this->ppbuf_input_arranged->next_buffer[i] = this->im2col_bank[i];
        }
        flag = 1;
        num--;  // 从窗口个数 -1
        start++;  // 起始窗口 +1
    }

    for(int i=0; i<num; i++){ 
        for(int c=0; c<C; c++){
            for(int r=0; r<R; r++){
                // 移位
                for(int q=0; q<S-1; q++){
                    this->im2col_bank[c*R*S + r*S + q] = this->im2col_bank[c*R*S + r*S + q + 1];
                }
                // 取数据
                this->im2col_bank[c*R*S + r*S + (S-1)] = this->ppbuf_input->current_buffer[c*R*Y_padded + r*Y_padded + (S+(start+i-1))];
                local_cycles++;
            }
        }

        // std::cout<<"The sorted data is : "<<std::endl;
        // for(int p=0; p<this->bankSize; p++){
        //     std::cout<<im2col_bank[p]<<"  ";
        // }
        // std::cout<<std::endl;

        //this->input_arranged.push_back(im2col_bank);

        for(int j=0; j<this->bankSize; j++){
            this->ppbuf_input_arranged->next_buffer[(flag+i)*this->bankSize + j] = this->im2col_bank[j];
        }
    }

    return local_cycles;
}

int Controller::process_conv_2(int i, int j, int cols, layer_topology layer_parameters){
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;

    int Y_padded = Y + 2*P;


    // std::cout<<"Below is input_buffer data : "<<std::endl;
    // for(int c=0; c<C; c++){
    //     std::cout<<"input channel is : "<<c<<std::endl;
    //     for(int x=0; x<R; x++){
    //         for(int y=0; y<Y_padded; y++){
    //             std::cout<<this->ppbuf_input->current_buffer[c*R*Y_padded + x*Y_padded + y]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    // }

    // 片外数据读取到片上buffer之后，对输入数据进行重排
    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->input_arranged_buffer = new int [input_arranged_buffer_size];

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();


    // 第一次排序，
    //std::cout<<"Y_ : "<<Y_<<std::endl;
    //std::cout<<"total num_tile : "<<std::ceil(Y_/(float)this->numBanks)<<std::endl;
    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){

       // this->input_arranged_buffer = new int [input_arranged_buffer_size];

        std::cout<<"num_tile : "<<num_tile<<"-------------"<<std::endl;
        int start = num_tile*this->numBanks;
        int end = std::min<int>(start+this->numBanks, Y_);
        int num = end-start;
        std::cout<<"rows : "<<num<<std::endl;
        std::cout<<"cols : "<<cols<<std::endl;
        // 调用转换函数
        int n_cycles_im2col = im2col(start, num, layer_parameters);
        local_cycles += n_cycles_im2col;
        // std::cout<<"debug"<<std::endl;

        // 调用脉动阵列
        matrixMultiply_new(num, this->bankSize, cols, this->input_arranged_buffer, this->weight_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
        //std::cout<<"begin load layer"<<std::endl;
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num, this->input_arranged_buffer, this->weight_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        //std::cout<<"begin load tile"<<std::endl;
        stonne_instance->loadGEMMTile(cols,1,num);
        //std::cout<<"begin run"<<std::endl;
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        stonne_instance->run();

        local_cycles += stonne_instance->n_cycles;

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }
        //std::cout<<this->neuron_state_regfile_conv[256]<<std::endl;
        // std::cout<<"cycles : "<<local_cycles<<std::endl;
        // std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

        // 重置数据，开始下一tile计算

        // delete[] this->input_arranged_buffer;
        // delete[] this->spikes;
    }

    // 遍历每一列，将输出结果累积到输出buffer
    for(int p=0; p<cols; p++){
        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期
        //std::cout<<packed_col_neuron_state[0]<<std::endl;
        
        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            this->output_buffer[p*Y_ + q] = packed_col_output[q];
            this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
        }
    }

    delete stonne_instance;
    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;

    delete[] this->input_arranged_buffer;

    return local_cycles;
}

// 在process_conv_2的基础上，进一步将input_arranged_buffer设置为双buffer
int Controller::process_conv_3(int layer_id, int i, int j, int cols, layer_topology layer_parameters){
    std::cout<<"call process_conv_3"<<std::endl;
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;

    int Y_padded = Y + 2*P;

    // std::cout<<"below is input_buffer data : "<<std::endl;
    // for(int c=0; c<C; c++){
    //     std::cout<<"channel : "<<c<<std::endl;
    //     for(int p=0; p<R; p++){
    //         for(int q=0; q<Y_padded; q++){
    //             std::cout<<this->ppbuf_input->current_buffer[c*R*Y_padded + p*Y_padded + q]<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // 片外数据读取到片上buffer之后，对输入数据进行重排
    this->bankSize = R*S*C;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小

    // std::cout<<"debug1"<<std::endl;

    // 排序buffer设置为双buffer
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->ppbuf_input_arranged = new PingPong_Buffer;
    this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // std::cout<<"debug2"<<std::endl;

    // if(layer_id == 5){
    //     std::cout<<"begin new stonne"<<std::endl;
    // }

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);

    // if(layer_id == 5){
    //     std::cout<<"end new stonne"<<std::endl;
    // }

    // std::cout<<"debug3"<<std::endl;
    this->output_regfile_conv = new int[Y_*cols]();
    // std::cout<<"debug4"<<std::endl;
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    // std::cout<<"debug5"<<std::endl;
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    // std::cout<<"debug6"<<std::endl;
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();
    // std::cout<<"debug7"<<std::endl;

    // std::cout<<"total num_tile : "<<std::ceil(Y_/(float)this->numBanks)<<std::endl;

    // if(layer_id == 5){
    //     std::cout<<"debug in process_conv_3   0"<<std::endl;
    // }

    // 第一次排序，将排序好的数据存在next_buffer中
    int num;
    if(Y_ >= this->numBanks){
        num = this->numBanks;
    } else {
        num = Y_;
    }

    // std::cout<<"debug3"<<std::endl;

    int n_cycles_im2col = im2col_ppbuffer(0,num,layer_parameters);

    // std::cout<<"debug4"<<std::endl;


    // if(layer_id == 5){
    //     std::cout<<"debug in process_conv_3   1"<<std::endl;
    // }

    local_cycles += n_cycles_im2col;

    // std::cout<<std::endl;
    // std::cout<<"************************************************************************"<<std::endl;
    // std::cout<<"Sort the input for the first time : "<<n_cycles_im2col<<"         local_cycles : "<<local_cycles<<std::endl;
    // 切换之后，上面加载数据的buffer成了current_buffer，用于接下来的计算
    PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // if(i==0 && j==15){
    //     std::cout<<"13 rows data is: "<<std::endl;
    //     for(int p=0; p<this->bankSize; p++){
    //         std::cout<<this->ppbuf_input_arranged->current_buffer[13*this->bankSize + p]<<" ";
    //     }
    //     std::cout<<std::endl;
    //     for(int p=0; p<this->bankSize; p++){
    //         std::cout<<this->ppbuf_weight->current_buffer[p]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }

    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
        // std::cout<<"current num_tile : "<<num_tile<<"-------------"<<std::endl;
        int n_cycles_im2col;
        if(num_tile+1 < std::ceil(Y_/(float)this->numBanks)){
            int start = (num_tile+1)*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            // 调用转换函数
            n_cycles_im2col = im2col_ppbuffer(start, num, layer_parameters);
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col<<std::endl;
        } else {
            n_cycles_im2col = 0;
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col<<std::endl;
        }

        int start_current = num_tile*this->numBanks;
        int end_current = std::min<int>(start_current+this->numBanks, Y_);
        int num_current = end_current-start_current;
        

        // 调用脉动阵列，下面的计算过程和对下一个tile的数据进行排序是并行的
        matrixMultiply_new(num_current, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        // matrixMultiply_new(num, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->weight_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
        //std::cout<<"begin load layer"<<std::endl;
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num_current, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        // stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num, this->ppbuf_input_arranged->current_buffer, this->weight_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        //std::cout<<"begin load tile"<<std::endl;
        stonne_instance->loadGEMMTile(cols,1,num_current);
        //std::cout<<"begin run"<<std::endl;
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        //std::cout<<"begin run()"<<std::endl;


        // if(layer_id == 5){
        //     std::cout<<"debug in process_conv_3   2"<<std::endl;
        // }

        stonne_instance->run();

        // if(layer_id == 5){
        //     std::cout<<"debug in process_conv_3   3"<<std::endl;
        // }



        int n_cycles_compute = stonne_instance->n_cycles;  // 计算所需时间
        // std::cout<<"compute the tile need cycles : "<<n_cycles_compute<<std::endl;
        local_cycles += std::max(n_cycles_im2col, n_cycles_compute);
        // std::cout<<"current local_cycles is : "<<local_cycles<<std::endl;

        PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }

        // if(layer_id == 5){
        //     std::cout<<"debug in process_conv_3   4"<<std::endl;
        // }

        // std::cout<<"neuron_state[208] : "<<this->neuron_state_regfile_conv[208]<<std::endl;
        // std::cout<<"output[208] : "<<this->output_regfile_conv[208]<<std::endl;

        // std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

        // 重置数据，开始下一tile计算

        //delete[] this->input_arranged_buffer;
        // delete[] this->spikes;
    }

    // std::cout<<"compute over, local_cycles is : "<<local_cycles<<std::endl;

    // if(layer_id == 5){
    //         std::cout<<"debug in process_conv_3   5"<<std::endl;
    // }

    // 遍历每一列，将输出结果累积到输出buffer
    for(int p=0; p<cols; p++){
        // if(layer_id == 5){
        //     std::cout<<"p : "<<p<<std::endl;
        // }
        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期

        // if(i==0 && j==15){
        //     std::cout<<"packed_neuron[13] : "<<packed_col_neuron_state[13]<<std::endl;
        //     std::cout<<"packed_output[13] : "<<packed_col_output[13]<<std::endl;
        // }
        
        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            this->output_buffer[p*Y_ + q] = packed_col_output[q];
            this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
        }

        // if(i==0 && j==15){
        //     std::cout<<"output_buffer[13] : "<<this->output_buffer[13]<<std::endl;
        //     std::cout<<"neuron_state_buffer[13] : "<<this->neuron_state_buffer[13]<<std::endl;
        // }

    }

    // if(layer_id == 5){
    //     std::cout<<"debug in process_conv_3   6"<<std::endl;
    // }

    // std::cout<<"************************************************************************"<<std::endl;
    // std::cout<<std::endl;
    
    delete stonne_instance;
    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;

    delete this->ppbuf_input_arranged;
    delete[] this->input_arranged_buffer_0;
    delete[] this->input_arranged_buffer_1;

    // if(layer_id == 5){
    //     std::cout<<"debug in process_conv_3   7"<<std::endl;
    // }

    return local_cycles;
}

// 执行一个input_buffer数据的卷积和池化
int Controller::process_conv_and_pooling(int layer_id, int i, int j, int cols, int count_rows, layer_topology layer_parameters){
    std::cout<<"call process_conv_and_pooling"<<std::endl;
    // count_rows : 用于累积输出到片上SRAM的计数器
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int P = layer_parameters.P;

    int Y_padded = Y + 2*P;

    // if(layer_id == 8){
    //     std::cout<<"below is input_buffer data : "<<std::endl;
    //     for(int c=0; c<C; c++){
    //         std::cout<<"channel : "<<c<<std::endl;
    //         for(int p=0; p<R; p++){
    //             for(int q=0; q<Y_padded; q++){
    //                 std::cout<<this->ppbuf_input->current_buffer[c*R*Y_padded + p*Y_padded + q]<<" ";
    //             }
    //             std::cout<<std::endl;
    //         }
    //         std::cout<<std::endl;
    //     }
    // }
    
    // std::cout<<"below is input_buffer data : "<<std::endl;
    // for(int c=0; c<C; c++){
    //     std::cout<<"channel : "<<c<<std::endl;
    //     for(int p=0; p<R; p++){
    //         for(int q=0; q<Y_padded; q++){
    //             std::cout<<this->ppbuf_input->current_buffer[c*R*Y_padded + p*Y_padded + q]<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // 片外数据读取到片上buffer之后，对输入数据进行重排
    this->bankSize = R*S*C;
    // std::cout<<"R*S*C : "<<this->bankSize<<std::endl;
    this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->im2col_bank.assign(this->bankSize,0); // 初始化，确定大小
    int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    // this->input_arranged_buffer = new int [input_arranged_buffer_size];

    this->ppbuf_input_arranged = new PingPong_Buffer;
    this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // 例化脉动阵列组件，执行矩阵乘法
    Stonne* stonne_instance = new Stonne(this->stonne_cfg);
    this->output_regfile_conv = new int[Y_*cols]();
    this->output_regfile_conv_cpu = new int[Y_*cols]();
    this->neuron_state_regfile_conv = new int[Y_*cols]();
    this->neuron_state_regfile_conv_cpu = new int[Y_*cols]();
    // std::cout<<"cpu neuron_state[0] : "<<this->neuron_state_regfile_conv_cpu[0]<<std::endl;
    // std::cout<<"cpu output[0] : "<<this->output_regfile_conv_cpu[0]<<std::endl;
    // std::cout<<"sim neuron_state[0] : "<<this->neuron_state_regfile_conv[0]<<std::endl;
    // std::cout<<"sim output[0] : "<<this->output_regfile_conv[0]<<std::endl;
    // std::cout<<"new regfiles --------------"<<std::endl;

    // 第一次排序
    int num;
    if(Y_ >= this->numBanks){
        num = this->numBanks;
    } else {
        num = Y_;
    }

    int n_cycles_im2col_first = im2col_ppbuffer(0,num,layer_parameters);
    local_cycles += n_cycles_im2col_first;
    PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // if(i==0 && j==15){
    //     for(int p=0; p<num; p++){
    //         for(int q=0; q<this->bankSize; q++){
    //             std::cout<<this->ppbuf_input_arranged->current_buffer[p*this->bankSize + q]<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    // }

    // std::cout<<std::endl;
    // std::cout<<"************************************************************************"<<std::endl;
    // std::cout<<"Sort the input for the first time : "<<n_cycles_im2col_first<<"         rows is : "<<num<<"         local_cycles : "<<local_cycles<<std::endl;

    for(int num_tile=0; num_tile<std::ceil(Y_/(float)this->numBanks); num_tile++){
        // std::cout<<"num_tile : "<<num_tile<<"-------------"<<std::endl;

        int n_cycles_im2col_next;
        if(num_tile+1 < std::ceil(Y_/(float)this->numBanks)){
            int start = (num_tile+1)*this->numBanks;
            int end = std::min<int>(start+this->numBanks, Y_);
            int num = end-start;
            // 调用转换函数
            n_cycles_im2col_next = im2col_ppbuffer(start, num, layer_parameters);
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col_next<<std::endl;
        } else {
            n_cycles_im2col_next = 0;
            // std::cout<<"Sort next input_arranged : "<<n_cycles_im2col_next<<std::endl;
        }

        int start_current = num_tile*this->numBanks;
        int end_current = std::min<int>(start_current+this->numBanks, Y_);
        int num_current = end_current - start_current;
        // std::cout<<"num_current : "<<num_current<<std::endl;

        // 调用脉动阵列
        // matrixMultiply_new(num, this->bankSize, cols, this->input_arranged_buffer, this->weight_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        matrixMultiply_new(num_current, this->bankSize, cols, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv_cpu, this->neuron_state_regfile_conv_cpu, this->stonne_cfg.V_th, num_tile, this->stonne_cfg.m_MSNetworkCfg.ms_rows);
        // std::cout<<"cpu neuron_state[1] : "<<this->neuron_state_regfile_conv_cpu[1]<<std::endl;
        // std::cout<<"cpu output[1] : "<<this->output_regfile_conv_cpu[1]<<std::endl;
        
        //std::cout<<"begin load layer"<<std::endl;
        stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num_current, this->ppbuf_input_arranged->current_buffer, this->ppbuf_weight->current_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        // stonne_instance->loadDenseGEMM(this->layer_name, cols, this->bankSize, num, this->input_arranged_buffer, this->weight_buffer, this->output_regfile_conv, this->neuron_state_regfile_conv, CNN_DATAFLOW);
        //std::cout<<"begin load tile"<<std::endl;
        stonne_instance->loadGEMMTile(cols,1,num_current);
        //std::cout<<"begin run"<<std::endl;
        stonne_instance->mem->reset(); // 重置信号
        stonne_instance->asnet->resetSignals();
        stonne_instance->n_cycles = 0;
        stonne_instance->run();
        // std::cout<<"sim neuron_state[1] : "<<this->neuron_state_regfile_conv[1]<<std::endl;
        // std::cout<<"sim output[1] : "<<this->output_regfile_conv[1]<<std::endl;

        int n_cycles_compute = stonne_instance->n_cycles;
        // std::cout<<"compute the tile need cycles : "<<n_cycles_compute<<std::endl;
        local_cycles += std::max(n_cycles_im2col_next, n_cycles_compute);
        // std::cout<<"current local_cycles is : "<<local_cycles<<std::endl;

        PingPongBuffer_Switch(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

        // 对当前tile计算结果进行验证
        for(int m = 0; m<Y_ * cols; m++){
            float difference = fabs(this->output_regfile_conv[m]-this->output_regfile_conv_cpu[m]) + fabs(this->neuron_state_regfile_conv[m]-this->neuron_state_regfile_conv_cpu[m]);
            if(difference>0){
                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_regfile_conv[m] << ". Value ofmap CPU: " << this->output_regfile_conv_cpu[m] << std::endl;
                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_regfile_conv[m] << ". Value neuron_state CPU: " << this->neuron_state_regfile_conv_cpu[m] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }
        //std::cout<<this->neuron_state_regfile_conv[256]<<std::endl;
        // std::cout<<"cycles : "<<local_cycles<<std::endl;
        // std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

        // 重置数据，开始下一tile计算

        // delete[] this->input_arranged_buffer;
        // delete[] this->spikes;
    }

    // 对输出和神经元状态分别处理
    // 1. 神经元状态写入output_buffer
    // 2. 输出累积到片上SRAM
    for(int p=0; p<cols; p++){
        std::vector<int> packed_col_output(Y_,0);
        std::vector<int> packed_col_neuron_state(Y_,0);
        for(int q=0; q<Y_; q++){
            packed_col_output[q] = this->output_regfile_conv[q*cols+p];
            packed_col_neuron_state[q] = this->neuron_state_regfile_conv[q*cols+p];
        }
        local_cycles++; // 假设打包需要一个周期
        //std::cout<<packed_col_neuron_state[0]<<std::endl;
        
        // 将打包后的数据写入片上buffer
        for(int q=0; q<Y_; q++){
            //std::cout<<"debug1"<<std::endl;
            this->on_chip_sram[p*2*Y_ + count_rows*Y_ + q] = packed_col_output[q];
            //std::cout<<"debug2"<<std::endl;
            this->neuron_state_buffer[p*Y_ + q] = packed_col_neuron_state[q];
            //std::cout<<"debug3"<<std::endl;
        }
    }

    delete stonne_instance;
    delete[] this->output_regfile_conv;
    delete[] this->output_regfile_conv_cpu;
    delete[] this->neuron_state_regfile_conv;
    delete[] this->neuron_state_regfile_conv_cpu;
    //delete[] this->input_arranged_buffer;
    delete this->ppbuf_input_arranged;
    delete[] this->input_arranged_buffer_0;
    delete[] this->input_arranged_buffer_1;

    return local_cycles;
}

// 执行池化
int Controller::process_pooling(int i, int j, int cols, layer_topology layer_parameters){
    int local_cycles = 0;

    int X_ = (layer_parameters.X - layer_parameters.R + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int Y_ = (layer_parameters.Y - layer_parameters.S + 2*layer_parameters.P)/layer_parameters.stride + 1;
    int num_output_buffer_need = Y_ * cols / 2;

    MYPOOL* pooling_instance = new MYPOOL(stonne_cfg);
    //std::cout<<"debug2"<<std::endl;
    pooling_instance->loadPOOLLayer(Y_, cols, this->on_chip_sram, this->output_buffer);
    //std::cout<<"debug3"<<std::endl;
    pooling_instance->run();
    //std::cout<<"debug4"<<std::endl;
    local_cycles += pooling_instance->n_cycle;
    
    // // 验证模拟器计算的结果
    pool2x2(this->on_chip_sram, this->output_buffer_cpu, Y_, cols);

    for(int k=0; k<num_output_buffer_need; k++) {
        float difference = fabs(this->output_buffer[k] - this->output_buffer_cpu[k]);
        if(difference>0){
            std::cout<<"error location : "<<k<<std::endl;
            std::cout<<"output_buffer : "<<this->output_buffer[k]<<std::endl;
            std::cout<<"output_buffer_cpu : "<<this->output_buffer_cpu[k]<<std::endl;
            assert(false);
        }
    }
    // std::cout << "\033[1;32m POOLING layer Test passed correctly\033[0m" << std::endl << std::endl;

    return local_cycles;
}

// 初始化乒乓buffer
void Controller::PingPongBuffer_Init(PingPong_Buffer* ppbuf, int* buffer_0, int* buffer_1){
    ppbuf->current_buffer = buffer_1;
    ppbuf->next_buffer = buffer_0;
    ppbuf->buffer_toggle = true;
}

// 乒乓buffer切换
void Controller::PingPongBuffer_Switch(PingPong_Buffer* ppbuf, int* buffer_0, int* buffer_1){
    ppbuf->buffer_toggle = !ppbuf->buffer_toggle;
    if(ppbuf->buffer_toggle) {
        ppbuf->current_buffer = buffer_1;
        ppbuf->next_buffer = buffer_0;
    } else {
        ppbuf->current_buffer = buffer_0;
        ppbuf->next_buffer = buffer_1;
    }
}

void Controller::read_callback(uint64_t addr){
    //std::cout << "Read completed at address: 0x" << std::hex << addr << std::endl;
    //std::cout << "Read completed at address: 0x" << std::hex << addr << " at cycle " << std::dec << globalCycle << std::endl;
    completed_reads++;
}

void Controller::write_callback(uint64_t addr){
    //std::cout << "Write completed at address: 0x" << std::hex << addr << std::endl;
    //std::cout << "Write completed at address: 0x" << std::hex << addr << " at cycle " << std::dec << globalCycle << std::endl;
    completed_writes++;
} 

// 数据存储方式是通道后置
std::tuple<int*, int*, int*, int*> Controller::runConv(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积
    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 从DRAM中取数据需要的一些（计数器）
    int num_input_buffer_fold = X_; 
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    int num_input_obtained = 0;
    int num_weight_obtained = 0;
    int weight_base_addr = this->weight_offset;
    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        //std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;
        int input_base_addr = this->input_offset;
        num_input_obtained = 0; // 重置

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;

        // 根据需要实例化片上buffer (input buffer 和 weight buffer)
        int num_input_buffer_need = R * Y * C;
        int num_weight_buffer_need = R * S * C * cols;
        if(num_input_buffer_need > ifmap_size){
            num_input_buffer_need = ifmap_size;  // 这种情况在padding不为0的时候可能出现
        }
        assert(num_input_buffer_need <= this->num_input);
        assert(num_weight_buffer_need <= this->num_weight);
        this->input_buffer = new int[num_input_buffer_need];
        this->weight_buffer = new int[num_weight_buffer_need];

        // 从DRAM中读取权重数据
        int num_weight_data = R * S * C * cols;
        int num_weight_read_request = std::ceil(num_weight_data*this->weight_width / (float)dram_instance->dram->GetBusBits());
        int num_weight_read_request_copy = num_weight_read_request;
        while(true){
            if(num_weight_read_request!=0){
                std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(weight_base_addr,false);  // 读请求
                this->read_request_fifo->push(read_request);
                weight_base_addr += this->addr_offset;
                num_weight_read_request--;
            } 

            if(completed_reads == num_weight_read_request_copy){  // 读请求处理完毕
                completed_reads = 0;
                // 模拟将片外数据写入片上buffer
                for(int j=0; j<num_weight_data; j++){
                    this->weight_buffer[j] = filter[num_weight_obtained+j];  // 权重数据的读取顺序在DRAM中是连续读取的
                }
                num_weight_obtained += num_weight_data;
                weight_base_addr = this->weight_offset + ((num_weight_obtained*this->weight_width)/8);
                break;
            }

            this->dram_instance->run();
            this->n_cycles++;
        }
        
        // 从DRAM中取出权重数据之后，开始取输入数据
        // 用于输入数据排序的数据结构
        this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
        this->bankSize = R * S * C;
        std::vector<int> skip_list;
        for(int j=0; j<num_input_buffer_fold; j++){
            if(j>0 && j<=P){
                skip_list.push_back(j);
            }
            if(j>num_input_buffer_fold-1-P && j<=num_input_buffer_fold-1){
                skip_list.push_back(j);
            }
        }

        // if(skip_list.size() > 0){
        //     std::cout<<"skip_list : "<<std::endl;
        //     for(int j=0; j<skip_list.size(); j++){
        //         std::cout<<skip_list[j]<<" ";
        //     }
        //     std::cout<<std::endl;
        // }
        // std::cout<<"num_input_buffer_fold : "<<num_input_buffer_fold<<std::endl;
        for(int j=0; j<num_input_buffer_fold; j++){
            //std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<num_input_buffer_fold-1<<std::endl;
            auto it = std::find(skip_list.begin(), skip_list.end(), j);
            if(it == skip_list.end()) {  // 如果不跳过，则取数据
                //std::cout<<"j : "<<j<<std::endl;
                int num_input_data;
                int num_input_read_request;
                int num_input_read_request_copy;
                if(j==0){
                    num_input_data = num_input_buffer_need;
                } else {
                    num_input_data = Y * C * 1;
                }
                num_input_read_request = std::ceil(num_input_data / (float)dram_instance->dram->GetBusBits());
                num_input_read_request_copy = num_input_read_request;

                while(true){
                    if(num_input_read_request!=0){ 
                        std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(input_base_addr,false);  // （地址，事件类型）false表示读请求，true表示写请求
                        this->read_request_fifo->push(read_request); // 将请求推入fifo
                        input_base_addr += this->addr_offset;  // 下一个读取地址
                        num_input_read_request--;
                    }

                    // completed_reads == num_total_read_request_copy 说明从外存取数据完毕
                    if(completed_reads == num_input_read_request_copy){ // 外存中输入数据存放的顺序：输入通道优先；权重数据：输入通道、行列、输出通道
                        // 请求全部响应之后，将片外的数据存入片上buffer
                        // 对于输入数据，第一次取时取R行，后续只取一行数据，要将重复的（R-1）行数据进行移动
                        if(j==0){
                            for(int q=0; q<num_input_data; q++){
                                this->input_buffer[q] = ifmap[q];
                            }
                        }else{
                            // 移位
                            for(int q=0; q<Y*C*(R-1); q++){ // （R-1）行在片上buffer内移动
                                this->input_buffer[q] = this->input_buffer[q + Y*C*1];
                            }
                            // 取数据
                            for(int q=0; q<Y*C*1; q++){
                                this->input_buffer[q + Y*C*(R-1)] = ifmap[num_input_obtained+q];
                            }
                        }
                        num_input_obtained += num_input_data;
                        input_base_addr = num_input_obtained / 8;
                        completed_reads = 0;
                        //std::cout<<"All input read requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                        break;
                    }
                    this->dram_instance->run();  // 一个周期驱动一次dram
                    this->n_cycles++;  // 一个周期发送一个内存请求事务
                }
            }
            
            // 片外数据读取到片上buffer之后，对输入数据进行重排
            std::vector<int> im2col_bank(this->bankSize,0);
            int add_0_above; // 考虑到池化，需要补零
            int add_0_below;
            if(j>=0 && j<P){
                add_0_above = P - j;
                add_0_below = 0; 
            } else if(j<=(num_input_buffer_fold-1) && j>(num_input_buffer_fold-1-P)){
                add_0_above = 0;
                add_0_below = j - (num_input_buffer_fold-1-P);
            } else {
                add_0_above = 0; 
                add_0_below = 0;
            }
            
            // std::cout<<"add_0_above : "<<add_0_above<<std::endl;
            // std::cout<<"add_0_below : "<<add_0_below<<std::endl;

            // 开始对片上 input buffer中的数据进行排序
            int num_tile = 0;
            int rows = 0;
            for(int k=0; k<Y_; k++){
                if(k==0){  // 第一个卷积窗口基地址计算方式不同，单独处理
                    if(add_0_above != 0){ // 上方补零
                        for(int r=0; r<add_0_above; r++){  // 这么些行是0
                            for(int q=0; q<C*S; q++){
                                im2col_bank[r*C*S + q] = 0;
                            }
                        }
    
                        for(int r=add_0_above; r<R; r++){  // 这些行从input_buffer中取数据，同时考虑最左侧有P列的0
                            int base_addr = (r - add_0_above) * C * Y;
                            for(int q=0; q<C*P; q++){  // 这些列是0
                                im2col_bank[r*C*S + q] = 0;
                            }
                            for(int q=C*P; q<C*S; q++){  // 需要从input buffer中取的值
                                im2col_bank[r*C*S + q] = this->input_buffer[base_addr + (q-C*P)];
                                this->n_cycles++;
                            }
                        }
                    } else if(add_0_below != 0){ // 下方补零
                        for(int r=0; r<(R-add_0_below); r++){  // 这些行是真实的数据
                            int base_addr = (r+add_0_below)*C*Y;
                            for(int q=0; q<P*C; q++){  // 这些列是0
                                im2col_bank[r*C*S + q] = 0; 
                            }
                            for(int q=P*C; q<C*S; q++){
                                im2col_bank[r*C*S + q] = this->input_buffer[base_addr+(q-C*P)];
                                this->n_cycles++;
                            }
                        }
                        for(int r=(R-add_0_above); r<R; r++){  // 这些行是0
                            for(int q=0; q<S*C; q++){
                                im2col_bank[r*C*S + q] = 0;
                            }
                        }

                    } else { // 不需要补零
                        for(int r=0; r<R; r++){
                            int base_addr = r*C*Y;
                            for(int q=0; q<C*P; q++){ // 这些列是0
                                im2col_bank[r*C*S + q] = 0;
                            }
                            for(int q=C*P; q<C*S; q++){
                                im2col_bank[r*C*S + q] = this->input_buffer[base_addr+(q-C*P)];
                                this->n_cycles++;
                            }
                        }
                    }

                    this->input_arranged.push_back(im2col_bank);
                    rows++;

                } else {  
                    if(k >= Y_-P){  // 加1列0
                        for(int r=0; r<R; r++){
                            for(int q=0; q<(S-1)*C; q++){
                                im2col_bank[r*S*C + q] = im2col_bank[r*S*C + q + C];
                            }
                            for(int q=(S-1)*C; q<S*C; q++){
                                im2col_bank[r*S*C + q] = 0;
                            }
                        }
                    } else {  // 移位和取数据
                        if(add_0_above != 0){
                            for(int r=0; r<add_0_above; r++){
                                for(int q=0; q<C*S; q++){
                                    im2col_bank[r*C*S + q] = 0;
                                }
                            }
                            for(int r=add_0_above; r<R; r++){
                                int base_addr = ((r-add_0_above)*Y + (S-P) + (k-1)) * C;
                                for(int q=0; q<C; q++){
                                    if(q==0){
                                        // 移位
                                        for(int num=0; num<(S-1)*C; num++){
                                            im2col_bank[r*C*S + num] = im2col_bank[r*C*S + num + C];
                                        }
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    } else {
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    }
                                }
                            }
                        } else if(add_0_below != 0){
                            for(int r=0; r<(R-add_0_below); r++){  // 这些行是真实的数据
                                int base_addr = ((r+add_0_below)*Y + (S-P) + (k-1)) * C;
                                for(int q=0; q<C; q++){
                                    if(q==0){
                                        for(int num=0; num<(S-1)*C; num++){  // 移位
                                            im2col_bank[r*C*S + num] = im2col_bank[r*C*S + num + C];
                                        }
                                        // 取数据
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    } else {
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    }
                                }
                            }
                            for(int r=R-add_0_below; r<R; r++){ // 这些行是0
                                for(int q=0; q<C*S; q++){
                                    im2col_bank[r*S*C + q] = 0;
                                }
                            }
                        } else {
                            for(int r=0; r<R; r++){
                                int base_addr = (r*Y + (S-P) + (k-1)) * C;
                                for(int q=0; q<C; q++){
                                    if(q==0){
                                        for(int num=0; num<(S-1)*C; num++){
                                            im2col_bank[r*C*S + num] = im2col_bank[r*C*S + num + C];
                                        }
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    } else {
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    }
                                }
                            }
                        }
                    }

                    this->input_arranged.push_back(im2col_bank);
                    rows++;

                    if(rows==this->numBanks || k == Y_-1){
                        //std::cout<<"rows : "<<rows<<std::endl;
                        num_tile++;

                        // 将this->input_arranged转化为数组
                        int total = rows * this->bankSize;
                        this->spikes = new int[total];
                        int index = 0;
                        for(const auto& row : input_arranged){
                            for(int val : row){
                                this->spikes[index++] = val;
                            }
                        }

                        // 根据需要例化 output buffer 和 neuron_state buffer
                        unsigned int num_output_buffer_need = rows * cols;
                        unsigned int num_neuron_state_buffer_need = rows * cols;
                        assert(num_output_buffer_need <= num_output);
                        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                        this->output_buffer = new int[num_output_buffer_need]();
                        this->output_buffer_cpu = new int[num_output_buffer_need]();
                        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                        this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                        // 例化脉动阵列组件，执行矩阵乘法
                        Stonne* stonne_instance = new Stonne(this->stonne_cfg);
                        matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                        stonne_instance->loadDenseGEMM(this->layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                        stonne_instance->loadGEMMTile(cols,1,rows);
                        stonne_instance->run();

                        this->n_cycles += stonne_instance->n_cycles;

                        // 对当前tile计算结果进行验证
                        for(int m = 0; m<num_output_buffer_need; m++){
                            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                            if(difference>0){
                                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                assert(false);
                            }
                        }

                        //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                        // 将计算结果写入DRAM
                        int num_total_write_request = 0;
                        bool only = true;  // 写命令发送标志
                        while(true){
                            while(only){
                                for(int m=0; m<rows; m++){
                                    int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                    int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                    int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                    num_total_write_request += current_total_write_request;

                                    // 该行的基地址
                                    int write_addr_output = this->output_offset + ((j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*K + i*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                    int write_addr_neuron_state = this->neuron_state_offset + ((j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*K + i*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                    while(current_total_write_request!=0){
                                        if(current_num_neuron_state_write_request!=0){
                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                            this->write_request_fifo->push(write_request);
                                            write_addr_neuron_state += this->addr_offset;
                                            current_num_neuron_state_write_request--;

                                        } else if(current_num_output_write_request!=0){
                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                            this->write_request_fifo->push(write_request);
                                            write_addr_output += this->addr_offset;
                                            current_num_output_write_request--;
                                        }

                                        this->dram_instance->run();
                                        this->n_cycles++; // 一个周期发送一个内存请求事务
                                        current_total_write_request--;
                                    }
                                    // 模拟将片上buffer中的数据写入DRAM
                                    for(int p=0; p<cols; p++){
                                        ofmap[(j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*K + i*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p] = this->output_buffer[m*cols + p];
                                        nfmap[(j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*K + i*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p] = this->neuron_state_buffer[m*cols + p];
                                    }
                                }
                                only = false;
                            }

                            if(completed_writes == num_total_write_request) {
                                completed_writes = 0; 
                                break;
                            }

                            this->dram_instance->run();
                            this->n_cycles++;

                        }

                        // 重置数据，开始下一tile计算
                        rows = 0;
                        this->input_arranged.clear();
                        delete stonne_instance;
                        delete[] this->spikes;
                        delete[] this->output_buffer;
                        delete[] this->output_buffer_cpu;
                        delete[] this->neuron_state_buffer;
                        delete[] this->neuron_state_buffer_cpu;
                    }
                }
            }
        }
        delete[] this->input_buffer;
        delete[] this->weight_buffer;
    }

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    conv_compute(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    for(int i=0; i<X_*Y_*this->layers[i].K; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

std::tuple<int*, int*, int*, int*> Controller::runConvandPooling(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    // 卷积层加池化层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积
    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K / 4;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        //std::cout<<"Initialize input data"<<std::endl;
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 从DRAM中取数据需要的一些（计数器）
    int num_input_buffer_fold = X_; 
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    int num_input_obtained = 0;
    int num_weight_obtained = 0;
    int weight_base_addr = this->weight_offset;
    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        //std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;
        int input_base_addr = this->input_offset;
        num_input_obtained = 0; // 重置

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;

        // 建模片上SRAM，用于累积卷积的输出进行池化
        int sram_size = 2 * Y_ * cols; 
        this->on_chip_sram = new int[sram_size]();

        // 根据需要实例化片上buffer (input buffer 和 weight buffer)
        int num_input_buffer_need = R * Y * C;
        int num_weight_buffer_need = R * S * C * cols;
        if(num_input_buffer_need > ifmap_size){
            num_input_buffer_need = ifmap_size;  // 这种情况在padding不为0的时候可能出现
        }
        assert(num_input_buffer_need <= this->num_input);
        assert(num_weight_buffer_need <= this->num_weight);
        this->input_buffer = new int[num_input_buffer_need];
        this->weight_buffer = new int[num_weight_buffer_need];

        // 从DRAM中读取权重数据
        int num_weight_data = R * S * C * cols;
        int num_weight_read_request = std::ceil(num_weight_data*this->weight_width / (float)dram_instance->dram->GetBusBits());
        int num_weight_read_request_copy = num_weight_read_request;
        while(true){
            if(num_weight_read_request!=0){
                std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(weight_base_addr,false);  // 读请求
                this->read_request_fifo->push(read_request);
                weight_base_addr += this->addr_offset;
                num_weight_read_request--;
            } 

            if(completed_reads == num_weight_read_request_copy){  // 读请求处理完毕
                completed_reads = 0;
                // 模拟将片外数据写入片上buffer
                for(int j=0; j<num_weight_data; j++){
                    this->weight_buffer[j] = filter[num_weight_obtained+j];  // 权重数据的读取顺序在DRAM中是连续读取的
                }
                num_weight_obtained += num_weight_data;
                weight_base_addr = this->weight_offset + ((num_weight_obtained*this->weight_width)/8);
                break;
            }

            this->dram_instance->run();
            this->n_cycles++;
        }
        
        // 从DRAM中取出权重数据之后，开始取输入数据
        // 用于输入数据排序的数据结构
        this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
        this->bankSize = R * S * C;
        std::vector<int> skip_list;
        for(int j=0; j<num_input_buffer_fold; j++){
            if(j>0 && j<=P){
                skip_list.push_back(j);
            }
            if(j>num_input_buffer_fold-1-P && j<=num_input_buffer_fold-1){
                skip_list.push_back(j);
            }
        }

        // std::cout<<"num_input_buffer_fold : "<<num_input_buffer_fold<<std::endl;
        int count_rows = 0; // 用于累加卷积的输出到片上SRAM计数
        for(int j=0; j<num_input_buffer_fold; j++){
            //std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<num_input_buffer_fold-1<<std::endl;
            auto it = std::find(skip_list.begin(), skip_list.end(), j);
            if(it == skip_list.end()) {  // 如果不跳过，则取数据
                //std::cout<<"j : "<<j<<std::endl;
                int num_input_data;
                int num_input_read_request;
                int num_input_read_request_copy;
                if(j==0){
                    num_input_data = num_input_buffer_need;
                } else {
                    num_input_data = Y * C * 1;
                }
                num_input_read_request = std::ceil(num_input_data / (float)dram_instance->dram->GetBusBits());
                num_input_read_request_copy = num_input_read_request;

                while(true){
                    if(num_input_read_request!=0){ 
                        std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(input_base_addr,false);  // （地址，事件类型）false表示读请求，true表示写请求
                        this->read_request_fifo->push(read_request); // 将请求推入fifo
                        input_base_addr += this->addr_offset;  // 下一个读取地址
                        num_input_read_request--;
                    }

                    // completed_reads == num_total_read_request_copy 说明从外存取数据完毕
                    if(completed_reads == num_input_read_request_copy){ // 外存中输入数据存放的顺序：输入通道优先；权重数据：输入通道、行列、输出通道
                        // 请求全部响应之后，将片外的数据存入片上buffer
                        // 对于输入数据，第一次取时取R行，后续只取一行数据，要将重复的（R-1）行数据进行移动
                        if(j==0){
                            for(int q=0; q<num_input_data; q++){
                                this->input_buffer[q] = ifmap[q];
                            }
                        }else{
                            // 移位
                            for(int q=0; q<Y*C*(R-1); q++){ // （R-1）行在片上buffer内移动
                                this->input_buffer[q] = this->input_buffer[q + Y*C*1];
                            }
                            // 取数据
                            for(int q=0; q<Y*C*1; q++){
                                this->input_buffer[q + Y*C*(R-1)] = ifmap[num_input_obtained+q];
                            }
                        }
                        num_input_obtained += num_input_data;
                        input_base_addr = num_input_obtained / 8;
                        completed_reads = 0;
                        //std::cout<<"All input read requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                        break;
                    }
                    this->dram_instance->run();  // 一个周期驱动一次dram
                    this->n_cycles++;  // 一个周期发送一个内存请求事务
                }
            }
            
            // 片外数据读取到片上buffer之后，对输入数据进行重排
            std::vector<int> im2col_bank(this->bankSize,0);
            int add_0_above; // 考虑到池化，需要补零
            int add_0_below;
            if(j>=0 && j<P){
                add_0_above = P - j;
                add_0_below = 0; 
            } else if(j<=(num_input_buffer_fold-1) && j>(num_input_buffer_fold-1-P)){
                add_0_above = 0;
                add_0_below = j - (num_input_buffer_fold-1-P);
            } else {
                add_0_above = 0; 
                add_0_below = 0;
            }
            
            // std::cout<<"add_0_above : "<<add_0_above<<std::endl;
            // std::cout<<"add_0_below : "<<add_0_below<<std::endl;

            // 开始对片上 input buffer中的数据进行排序
            int num_tile = 0;
            int rows = 0;
            
            for(int k=0; k<Y_; k++){
                if(k==0){  // 第一个卷积窗口基地址计算方式不同，单独处理
                    if(add_0_above != 0){ // 上方补零
                        for(int r=0; r<add_0_above; r++){  // 这么些行是0
                            for(int q=0; q<C*S; q++){
                                im2col_bank[r*C*S + q] = 0;
                            }
                        }
    
                        for(int r=add_0_above; r<R; r++){  // 这些行从input_buffer中取数据，同时考虑最左侧有P列的0
                            int base_addr = (r - add_0_above) * C * Y;
                            for(int q=0; q<C*P; q++){  // 这些列是0
                                im2col_bank[r*C*S + q] = 0;
                            }
                            for(int q=C*P; q<C*S; q++){  // 需要从input buffer中取的值
                                im2col_bank[r*C*S + q] = this->input_buffer[base_addr + (q-C*P)];
                                this->n_cycles++;
                            }
                        }
                    } else if(add_0_below != 0){ // 下方补零
                        for(int r=0; r<(R-add_0_below); r++){  // 这些行是真实的数据
                            int base_addr = (r+add_0_below)*C*Y;
                            for(int q=0; q<P*C; q++){  // 这些列是0
                                im2col_bank[r*C*S + q] = 0; 
                            }
                            for(int q=P*C; q<C*S; q++){
                                im2col_bank[r*C*S + q] = this->input_buffer[base_addr+(q-C*P)];
                                this->n_cycles++;
                            }
                        }
                        for(int r=(R-add_0_above); r<R; r++){  // 这些行是0
                            for(int q=0; q<S*C; q++){
                                im2col_bank[r*C*S + q] = 0;
                            }
                        }

                    } else { // 不需要补零
                        for(int r=0; r<R; r++){
                            int base_addr = r*C*Y;
                            for(int q=0; q<C*P; q++){ // 这些列是0
                                im2col_bank[r*C*S + q] = 0;
                            }
                            for(int q=C*P; q<C*S; q++){
                                im2col_bank[r*C*S + q] = this->input_buffer[base_addr+(q-C*P)];
                                this->n_cycles++;
                            }
                        }
                    }

                    this->input_arranged.push_back(im2col_bank);
                    rows++;

                } else {  
                    if(k >= Y_-P){  // 加1列0
                        for(int r=0; r<R; r++){
                            for(int q=0; q<(S-1)*C; q++){
                                im2col_bank[r*S*C + q] = im2col_bank[r*S*C + q + C];
                            }
                            for(int q=(S-1)*C; q<S*C; q++){
                                im2col_bank[r*S*C + q] = 0;
                            }
                        }
                    } else {  // 移位和取数据
                        if(add_0_above != 0){
                            for(int r=0; r<add_0_above; r++){
                                for(int q=0; q<C*S; q++){
                                    im2col_bank[r*C*S + q] = 0;
                                }
                            }
                            for(int r=add_0_above; r<R; r++){
                                int base_addr = ((r-add_0_above)*Y + (S-P) + (k-1)) * C;
                                for(int q=0; q<C; q++){
                                    if(q==0){
                                        // 移位
                                        for(int num=0; num<(S-1)*C; num++){
                                            im2col_bank[r*C*S + num] = im2col_bank[r*C*S + num + C];
                                        }
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    } else {
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    }
                                }
                            }
                        } else if(add_0_below != 0){
                            for(int r=0; r<(R-add_0_below); r++){  // 这些行是真实的数据
                                int base_addr = ((r+add_0_below)*Y + (S-P) + (k-1)) * C;
                                for(int q=0; q<C; q++){
                                    if(q==0){
                                        for(int num=0; num<(S-1)*C; num++){  // 移位
                                            im2col_bank[r*C*S + num] = im2col_bank[r*C*S + num + C];
                                        }
                                        // 取数据
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    } else {
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    }
                                }
                            }
                            for(int r=R-add_0_below; r<R; r++){ // 这些行是0
                                for(int q=0; q<C*S; q++){
                                    im2col_bank[r*S*C + q] = 0;
                                }
                            }
                        } else {
                            for(int r=0; r<R; r++){
                                int base_addr = (r*Y + (S-P) + (k-1)) * C;
                                for(int q=0; q<C; q++){
                                    if(q==0){
                                        for(int num=0; num<(S-1)*C; num++){
                                            im2col_bank[r*C*S + num] = im2col_bank[r*C*S + num + C];
                                        }
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    } else {
                                        im2col_bank[r*C*S + (S-1)*C + q] = this->input_buffer[base_addr+q];
                                        this->n_cycles++;
                                    }
                                }
                            }
                        }
                    }

                    this->input_arranged.push_back(im2col_bank);
                    rows++;

                    if(rows==this->numBanks || k == Y_-1){
                        //std::cout<<"rows : "<<rows<<std::endl;
                        num_tile++;

                        // 将this->input_arranged转化为数组
                        int total = rows * this->bankSize;
                        this->spikes = new int[total];
                        int index = 0;
                        for(const auto& row : input_arranged){
                            for(int val : row){
                                this->spikes[index++] = val;
                            }
                        }

                        // 根据需要例化 output buffer 和 neuron_state buffer
                        unsigned int num_output_buffer_need = rows * cols;
                        unsigned int num_neuron_state_buffer_need = rows * cols;
                        assert(num_output_buffer_need <= num_output);
                        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                        this->output_buffer = new int[num_output_buffer_need]();
                        this->output_buffer_cpu = new int[num_output_buffer_need]();
                        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                        this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                        // 例化脉动阵列组件，执行矩阵乘法
                        Stonne* stonne_instance = new Stonne(this->stonne_cfg);
                        matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                        stonne_instance->loadDenseGEMM(this->layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                        stonne_instance->loadGEMMTile(cols,1,rows);
                        stonne_instance->run();

                        this->n_cycles += stonne_instance->n_cycles;

                        // 对当前tile计算结果进行验证
                        for(int m = 0; m<num_output_buffer_need; m++){
                            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                            if(difference>0){
                                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                assert(false);
                            }
                        }

                        //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                        // 将当前tile的输出结果和神经元中间状态分别进行处理
                        // 1. 将当前tile的神经元中间状态写入DRAM
                        int num_total_write_request = 0;
                        bool only = true;  // 写命令发送标志
                        while(true){
                            while(only){
                                for(int m=0; m<rows; m++){
                                    int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                    num_total_write_request += current_num_neuron_state_write_request;

                                    // 该行的基地址
                                    int write_addr_neuron_state = this->neuron_state_offset + ((j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*K + i*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                    while(current_num_neuron_state_write_request!=0){
                                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                        this->write_request_fifo->push(write_request);
                                        write_addr_neuron_state += this->addr_offset;
                                        current_num_neuron_state_write_request--;

                                        this->dram_instance->run();
                                        this->n_cycles++; // 一个周期发送一个内存请求事务
                                    }
                                    // 模拟将片上buffer中的数据写入DRAM
                                    for(int p=0; p<cols; p++){
                                        nfmap[(j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*K + i*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p] = this->neuron_state_buffer[m*cols + p];
                                    }
                                }
                                only = false;
                            }

                            if(completed_writes == num_total_write_request) {
                                completed_writes = 0; 
                                break;
                            }

                            this->dram_instance->run();
                            this->n_cycles++;
                        }

                        // 2. 将当前tile的输出结果进行累积
                        for(int p=0; p<cols; p++){  // 遍历output_buffer中的每一列
                            std::vector<int> packed_col(rows,0);
                            for(int q=0; q<rows; q++){
                                packed_col[q] = this->output_buffer[q*cols+p];
                                //std::cout<<"location in output_buffer is : "<<q*cols+p<<"   location in packed_col is : "<<q<<std::endl;
                            }
                            this->n_cycles++;  // 打包需要一个周期

                            // 将打包后的数据写入SRAM
                            for(int q=0; q<rows; q++){
                                this->on_chip_sram[p*2*Y_ + count_rows*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + q] = packed_col[q];
                                //std::cout<<"location in sram is : "<<p*2*Y_ + count_rows*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + q<<"   location in packed_col is : "<<q<<std::endl;
                            }
                            this->n_cycles++; // 将打包后的数据写入SRAM需要一个时钟周期，SRAM的总线宽度设置为脉动阵列的高度
                        }
                        // 重置数据，开始下一tile计算
                        rows = 0;
                        this->input_arranged.clear();
                        delete stonne_instance;
                        delete[] this->spikes;
                        delete[] this->output_buffer;
                        delete[] this->output_buffer_cpu;
                        delete[] this->neuron_state_buffer;
                        delete[] this->neuron_state_buffer_cpu;
                    }
                }
            }

            count_rows++;

            if(j>0 && j%2!=0){
                int X_pool = (j-1)/2; // 当前池化得到的行在输出特征图的位置
                //std::cout<<"X_pool : "<<X_pool<<std::endl;
                count_rows = 0;

                // 根据需要例化存储池化结果的寄存器组
                int num_output_regs_need = 2*cols*Y_/4;
                this->output_regfile_pooling = new int[num_output_regs_need]();
                this->output_regfile_pooling_cpu = new int[num_output_regs_need]();

                // 将累积的卷积输出结果进行池化
                MYPOOL* pooling_instance = new MYPOOL(stonne_cfg);
                //std::cout<<"debug2"<<std::endl;
                pooling_instance->loadPOOLLayer(Y_, cols, this->on_chip_sram, this->output_regfile_pooling);
                //std::cout<<"debug3"<<std::endl;
                pooling_instance->run();
                //std::cout<<"debug4"<<std::endl;
                this->n_cycles += pooling_instance->n_cycle;
                
                // // 验证模拟器计算的结果
                pool2x2(this->on_chip_sram, this->output_regfile_pooling_cpu, Y_, cols);

                for(int k=0; k<num_output_regs_need; k++) {
                    float difference = fabs(output_regfile_pooling[i] - output_regfile_pooling_cpu[i]);
                    if(difference>0){
                        std::cout<<"error location : "<<i<<std::endl;
                        assert(false);
                    }
                }
                
                //std::cout << "\033[1;32m POOLING layer Test passed correctly\033[0m" << std::endl << std::endl;

                // std::cout<<"Below is on_chip_sram : "<<std::endl;
                // for(int i=0;i<sram_size; i++){
                //     if(i>0 && i%Y_==0){
                //         std::cout<<std::endl;
                //     }
                //     std::cout<<this->on_chip_sram[i]<<"  ";
                // }
                // std::cout<<std::endl;

                // std::cout<<"Below is output_regfile_pooling : "<<std::endl;
                // for(int i=0;i<num_output_regs_need; i++){
                //     if(i>0 && i%(Y_/2)==0){
                //         std::cout<<std::endl;
                //     }
                //     std::cout<<this->output_regfile_pooling[i]<<"  ";
                // }
                // std::cout<<std::endl;

                // 将池化结果写入DRAM
                // 可以将池化结果看成一个二维阵列，其中一列是各个输出通道相同位置的数据，所以将一列打包进行写
                int num_total_write_request = 0;
                bool only=true;
                while(true){
                    while(only){
                        for(int m=0; m<Y_/2; m++){
                            std::vector<int> packed_col(cols,0);
                            // 逐列进行打包
                            for(int q=0; q<cols; q++){
                                packed_col[q] = output_regfile_pooling[q*Y_/2 + m];
                            }
                            this->n_cycles++; // 打包需要一个周期
                            int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                            num_total_write_request += current_num_output_write_request;

                            //该行基地址
                            int write_addr_output = ((X_pool * K * Y_/2) + (m * K) + (i * this->stonne_cfg.m_MSNetworkCfg.ms_cols)) / 8;
                            while(current_num_output_write_request!=0){
                                std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                this->write_request_fifo->push(write_request);
                                write_addr_output += this->addr_offset;
                                current_num_output_write_request--;

                                this->dram_instance->run();
                                this->n_cycles++; // 一个周期发送一个内存请求事务
                            }
                            // 模拟将output_regs中的数据写入DRAM
                            for(int p=0; p<cols; p++){
                               // std::cout<<"m : "<<m<<std::endl;
                                ofmap[(X_pool * K * Y_/2) + (m * K) + (i * this->stonne_cfg.m_MSNetworkCfg.ms_cols + p)] = packed_col[p];
                                //std::cout<<"location in ofmap is : "<<(X_pool * K * Y_/2) + (m * K) + (i * this->stonne_cfg.m_MSNetworkCfg.ms_cols + p)<<"   location in packed_col is : "<<p<<std::endl;
                            }

                        }
                        only = false;
                    }
                    if(completed_writes == num_total_write_request) {
                        completed_writes = 0; 
                        break;
                    }
                    this->dram_instance->run();
                    this->n_cycles++;
                }

                delete[] this->output_regfile_pooling;
                delete[] this->output_regfile_pooling_cpu;
            }
            //std::cout<<"a input loop over"<<std::endl;
        }
        delete[] this->on_chip_sram;
        delete[] this->input_buffer;
        delete[] this->weight_buffer;
    }

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确

    conv_and_pooling_compute(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    // 检查神经元状态
    for(int i=0; i<X_*Y_*this->layers[i].K; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    // 检查输出
    // for(int i=0; i<X_*Y_*K/4; i++){
    //     std::cout<<ofmap[i]<<" ";
    // }
    // std::cout<<std::endl;
    // for(int i=0; i<X_*Y_*K/4; i++){
    //     std::cout<<ofmap_cpu[i]<<" ";
    // }
    // std::cout<<std::endl;

    for(int i=0; i<X_*Y_*K/4; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(ofmap[i]-ofmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

// 单buffer，顺序执行，未模块化，假设input_buffer足够大，可以一次性加载所有的输入
std::tuple<int*, int*, int*, int*> Controller::runFC_0(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters) {
    // 全连接层控制逻辑
    std::cout<<"Call runFC_0 function !!!"<<std::endl;
    std::cout<<std::endl;

    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;

    std::cout<<"currrent cycles : "<<this->n_cycles<<std::endl;
    this->n_fc++;
    this->layer_name = "fc"+std::to_string(this->n_fc);

    // 提取层参数，输入层神经元个数和输出层神经元个数
    int output_layer = layer_parameters.output_neuron;
    int input_layer = layer_parameters.input_neuron;

    // 建模存储真实的数据
    int ifmap_size = input_layer;
    int filter_size = output_layer * input_layer;
    int ofmap_size = output_layer;
    int nfmap_size = output_layer;

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim
    this->dram_instance = new Dram(read_callback, write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 下面开始从DRAM取数据到片上buffer 
    // 这里假设 input buffer 的容量足够大，可以一次性容纳所有的输入数据，所以输入只需要读一次，权重数据需要分时处理
    // 先读输入， 根据需要例化片上 input buffer
    int num_input_buffer_need = input_layer;
    assert(num_input_buffer_need <= this->num_input);
    this->input_buffer = new int[num_input_buffer_need];

    // 一些用于读取数据的计数器
    int num_input_data = input_layer; 
    int num_input_read_request = std::ceil(num_input_data / (float)dram_instance->dram->GetBusBits());
    int num_input_read_request_copy = num_input_read_request;

    int input_base_addr = this->input_offset;
    int load_input_cycles = 0;
    while(true){
        if(num_input_read_request!=0){
            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(input_base_addr,false);
            this->read_request_fifo->push(read_request);
            input_base_addr += this->addr_offset;
            num_input_read_request--;
        }
        if(completed_reads == num_input_read_request_copy) {
            for(int i=0; i<input_layer; i++){
                this->input_buffer[i] = ifmap[i];
            }
            completed_reads = 0;
            break;
        }
        this->dram_instance->run();
        this->n_cycles++;
        std::cout<<"load_input_cycles : "<<load_input_cycles++<<"           global cycles : "<<this->n_cycles<<std::endl;
    }

    // 下面开始读取权重数据
    int num_weight_buffer_fold = std::ceil(output_layer / (float)this->stonne_cfg.m_MSNetworkCfg.ms_rows);
    int num_weight_obtained = 0; // 记录已经从DRAM中取出的数据个数
    int weight_base_addr = this->weight_offset;

    int num_output_written = 0;
    int num_neuron_state_written = 0;
    int base_output_addr = this->output_offset;
    int base_neuron_state_addr = this->neuron_state_offset;
    int load_weight_cycles = 0;
    for(int i=0; i<num_weight_buffer_fold; i++){
        int start_row = i*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
        int end_row = std::min<int>(start_row+this->stonne_cfg.m_MSNetworkCfg.ms_rows, output_layer);
        int rows = end_row - start_row;

        // 根据需要例化片上输出buffer
        int num_weight_buffer_need = rows*input_layer;
        int num_output_buffer_need = rows;
        int num_neuron_state_buffer_need = rows;
        assert(num_weight_buffer_need <= this->num_weight);
        assert(num_output_buffer_need <= this->num_output);
        assert(num_neuron_state_buffer_need <= this->num_neuron_state);
        this->weight_buffer = new int[num_weight_buffer_need];            
        this->output_buffer = new int[num_output_buffer_need]();
        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
        this->output_buffer_cpu = new int[num_output_buffer_need]();
        this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();
        
        int num_weight_data = rows * input_layer;
        int num_weight_read_request = std::ceil(num_weight_data*this->weight_width /(float)dram_instance->dram->GetBusBits());
        int num_weight_read_request_copy = num_weight_read_request;
        while(true){  // 发送DRAM内存事务请求
            if(num_weight_read_request!=0){
                std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(weight_base_addr,false);
                this->read_request_fifo->push(read_request);
                weight_base_addr += this->addr_offset; 
                num_weight_read_request--;
            }

            if(completed_reads == num_weight_read_request_copy){  // 内存读取事务响应完毕
                // 将片外的数据存储片上buffer
                // 权重数据
                for(int j=0; j<num_weight_data; j++){
                    this->weight_buffer[j] = filter[num_weight_obtained+j];
                }

                num_weight_obtained += num_weight_data;
                weight_base_addr = this->weight_offset + num_weight_obtained*this->weight_width/8;  // 下一轮取数据的基地址
                completed_reads = 0;
                //std::cout<<"All read requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                break;
            }
            this->dram_instance->run();
            this->n_cycles++;
            std::cout<<"load_weight_cycles : "<<load_weight_cycles++<<"           global cycles : "<<this->n_cycles<<std::endl;
        }

        // 调用模拟器卷积模块部分完成一个tile的计算
        // 得到数据之后，送入计算单元计算
        Stonne* stonne_instance = new Stonne(stonne_cfg);
        matrixMultiply(rows,input_layer,1, this->weight_buffer, this->input_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
        //sequential_layer(1, input_layer, 1, this->stonne_cfg.m_MSNetworkCfg.ms_rows, 1, 1, this->stonne_cfg.m_MSNetworkCfg.ms_cols, input_layer, 1, this->weight_buffer, this->input_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->Timestamp, this->pooling_enabled);
        //std::cout<<"debug"<<std::endl;
        stonne_instance->loadDenseGEMM(layer_name, 1, input_layer, rows, this->weight_buffer, this->input_buffer, this->output_buffer, this->neuron_state_buffer, CNN_DATAFLOW);
        //std::cout<<"debug"<<std::endl;
        stonne_instance->loadGEMMTile(1, 1,(end_row-start_row));
        stonne_instance->run();

        this->n_cycles += stonne_instance->n_cycles;
        std::cout<<"stonne_instance cycles : "<<stonne_instance->n_cycles<<"           global cycles : "<<this->n_cycles<<std::endl;

        // 对当前tile的计算结果进行验证
        for(int j=0; j<num_output_buffer_need; j++){
            float difference = fabs(this->output_buffer[j] - this->output_buffer_cpu[j]) + fabs(this->neuron_state_buffer[j] - this->neuron_state_buffer_cpu[j]);
            if(difference>0){
                std::cout << "ERROR position " << j <<  ": Value ofmap simulator: " << this->output_buffer[j] << ". Value ofmap CPU: " << this->output_buffer_cpu[j] << std::endl;
                std::cout << "ERROR position " << j <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[j] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[j] << std::endl;
                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                assert(false);
            }
        }
        //std::cout << "\033[1;32mTest passed correctly \033[0m" << std::endl << std::endl;

        // 将计算结果写入DRAM
        int num_output_write_request = std::ceil(num_output_buffer_need / (float)this->dram_instance->dram->GetBusBits());
        int num_neuron_state_write_request = std::ceil(num_neuron_state_buffer_need*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)this->dram_instance->dram->GetBusBits());
        int num_total_write_requset = num_output_write_request + num_neuron_state_write_request;
        int num_total_write_request_copy = num_total_write_requset;
        int store_result_cycles = 0;
        while(true){  // 发送DRAM内存事务请求
            if(num_total_write_requset!=0){
                if(num_output_write_request!=0){
                    std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(base_output_addr,true);
                    this->write_request_fifo->push(write_request);
                    base_output_addr += this->addr_offset;
                    num_output_write_request--;
                }else if(num_neuron_state_write_request!=0){
                    std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(base_neuron_state_addr,true);
                    this->write_request_fifo->push(write_request);
                    base_neuron_state_addr += this->addr_offset;
                    num_neuron_state_write_request--;
                }
                num_total_write_requset--;
            }
            //std::cout<<"debug"<<std::endl;
            if(completed_writes == num_total_write_request_copy){ // 请求响应完之后，将片上buffer数据写入DRAM

                for(int k=0; k<num_output_buffer_need; k++){
                    ofmap[num_output_written+k] = this->output_buffer[k];
                }
                for(int k=0; k<num_neuron_state_buffer_need; k++){
                    nfmap[num_neuron_state_written+k] = this->neuron_state_buffer[k];
                }

                num_output_written += num_output_buffer_need;
                num_neuron_state_written += num_neuron_state_buffer_need;

                base_output_addr = this->output_offset + num_output_written / 8;
                base_neuron_state_addr = this->neuron_state_offset + num_neuron_state_written*(std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;

                completed_writes = 0;
                //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                break;
            }

            this->dram_instance->run();
            this->n_cycles++;
            std::cout<<"store_result_cycles : "<<store_result_cycles++<<"           global cycles : "<<this->n_cycles<<std::endl;
        }
        delete stonne_instance;
        delete[] this->weight_buffer;            
        delete[] this->output_buffer;
        delete[] this->neuron_state_buffer;
        delete[] this->output_buffer_cpu;
        delete[] this->neuron_state_buffer_cpu;

    }
    delete[] this->input_buffer;

    // 计算完毕，验证写到DRAM的结果是否正确
    matrixMultiply(output_layer, input_layer, 1, filter, ifmap, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th, this->pooling_enabled);

    for(int i=0; i<output_layer; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

// 单buffer，顺序执行，模块化，input_buffer容量有限
std::tuple<int*, int*, int*, int*> Controller::runFC_1(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters) {
    // 全连接层控制逻辑
    std::cout<<"Call runFC_1 function !!!"<<std::endl;
    std::cout<<std::endl;

    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_fc++;
    this->layer_name = "fc"+std::to_string(this->n_fc);

    // 提取层参数，输入层神经元个数和输出层神经元个数
    int output_layer = layer_parameters.output_neuron;
    int input_layer = layer_parameters.input_neuron;

    // 建模存储真实的数据
    int ifmap_size = input_layer;
    int filter_size = output_layer * input_layer;
    int ofmap_size = output_layer;
    int nfmap_size = output_layer;

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim
    this->dram_instance = new Dram(read_callback, write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 例化片上输入buffer和权重buffer
    int num_input_buffer_need = this->num_input;
    int num_weight_buffer_need = input_layer * this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    int num_output_buffer_need = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    int num_neuron_state_buffer_need = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    assert(num_weight_buffer_need <= this->num_weight);
    assert(num_output_buffer_need <= this->num_output);
    assert(num_neuron_state_buffer_need <= this->num_neuron_state);

    this->input_buffer = new int[num_input_buffer_need];
    this->weight_buffer = new int[num_weight_buffer_need];
    this->output_buffer = new int[num_output_buffer_need]();
    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
    this->output_buffer_cpu = new int[num_output_buffer_need]();
    this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

    // 累积中间结果
    int* acc_inter_result = new int[num_neuron_state_buffer_need]();
    int* acc_inter_output = new int[num_input_buffer_need]();

    // 从片外读取数据的计数器
    int num_input_buffer_fold = std::ceil(input_layer / (float)this->num_input);  // 片上input_buffer容量有限，计算需要分几次读取
    int num_weight_buffer_fold = std::ceil(output_layer / (float)this->stonne_cfg.m_MSNetworkCfg.ms_rows);  // 权重数据也需要分批读取，需要（num_input_buffer_fold*num_weight_buffer_fold）

    for(int i=0; i<num_weight_buffer_fold; i++){
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;
        int num_input_obtained = 0;
        memset(acc_inter_result, 0, num_neuron_state_buffer_need * sizeof(int)); // 将存储中间结果的数据结构置为0
        for(int j=0; j<num_input_buffer_fold; j++){
            // 取相对应的输入数据和权重数据到片上buffer
            // 取输入数据
            int start = j*this->num_input;
            int end = std::min<int>(start+this->num_input, input_layer);
            int num_input_data = end-start;
            int n_cycles_load_input = load_input_data_fc(ifmap, this->dram_instance, num_input_obtained, num_input_data);
            num_input_obtained += num_input_data;
            std::cout<<"load the input need cycles : "<<n_cycles_load_input<<std::endl;
            this->n_cycles += n_cycles_load_input;

            // 取权重数据
            int n_cycles_load_weight = load_weight_data_fc(filter, this->dram_instance, i, j, layer_parameters);
            std::cout<<"load the weight need cycles : "<<n_cycles_load_weight<<std::endl;
            this->n_cycles += n_cycles_load_weight;

            // 调用脉动阵列计算
            int n_cycles_process_fc = process_fc(i, j, layer_parameters);
            std::cout<<"process_fc need cycles : "<<n_cycles_process_fc<<std::endl;
            this->n_cycles += n_cycles_process_fc;

            // 对中间结果进行累积
            for(int k=0; k<num_neuron_state_buffer_need; k++){
                acc_inter_result[k] += this->neuron_state_buffer[k];
                acc_inter_output[k] += this->output_buffer[k];
            }
        }
        // 对累积的结果进行脉冲激活
        for(int k=0; k<num_neuron_state_buffer_need; k++){
            if(acc_inter_output[k] >= 1){
                this->output_buffer[k] = 1;
                this->neuron_state_buffer[k] = 0;
            } else {
                this->output_buffer[k] = 0;
                this->neuron_state_buffer[k] = acc_inter_result[k];
            }
        }

        // 将结果写入DRAM
        int n_cycles_write_result = store_output_and_neuronstate_data_fc(ofmap, nfmap, dram_instance, i, layer_parameters);
        std::cout<<"n_cycles_write_result : "<<n_cycles_write_result<<std::endl;
        this->n_cycles += n_cycles_write_result;
    }

    // 计算完毕，验证写到DRAM的结果是否正确
    matrixMultiply(output_layer, input_layer, 1, filter, ifmap, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th, this->pooling_enabled);

    for(int i=0; i<output_layer; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

// 输入和权重双buffer，模块化，input_buffer容量有限
std::tuple<int*, int*, int*, int*> Controller::runFC_2(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters) {
    // 全连接层控制逻辑
    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                                 Call runFC_2 function                                          "<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;

    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_fc++;
    this->layer_name = "fc"+std::to_string(this->n_fc);

    // 提取层参数，输入层神经元个数和输出层神经元个数
    int output_layer = layer_parameters.output_neuron;
    int input_layer = layer_parameters.input_neuron;

    // 建模存储真实的数据
    int ifmap_size = input_layer;
    int filter_size = output_layer * input_layer;
    int ofmap_size = output_layer;
    int nfmap_size = output_layer;

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim
    this->dram_instance = new Dram(read_callback, write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 例化片上输入buffer和权重buffer
    int num_input_buffer_need = this->num_input;
    int num_weight_buffer_need = input_layer * this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    int num_output_buffer_need = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    int num_neuron_state_buffer_need = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    assert(num_weight_buffer_need <= this->num_weight);
    assert(num_output_buffer_need <= this->num_output);
    assert(num_neuron_state_buffer_need <= this->num_neuron_state);

    // this->input_buffer = new int[num_input_buffer_need];
    // this->weight_buffer = new int[num_weight_buffer_need];
    this->output_buffer = new int[num_output_buffer_need]();
    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
    this->output_buffer_cpu = new int[num_output_buffer_need]();
    this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

    this->ppbuf_input = new PingPong_Buffer;
    this->input_buffer_0 = new int[num_input_buffer_need];
    this->input_buffer_1 = new int[num_input_buffer_need];
    PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    // 累积中间结果
    int* acc_inter_result = new int[num_neuron_state_buffer_need]();
    int* acc_inter_output = new int[num_input_buffer_need]();

    // 从片外读取数据的计数器
    int num_input_buffer_fold = std::ceil(input_layer / (float)this->num_input);  // 片上input_buffer容量有限，计算需要分几次读取
    int num_weight_buffer_fold = std::ceil(output_layer / (float)this->stonne_cfg.m_MSNetworkCfg.ms_rows);  // 权重数据也需要分批读取，需要（num_input_buffer_fold*num_weight_buffer_fold）

    // 第一次取输入数据和权重数据
    int num_input_obtained = 0;
    int num_input_data_first = std::min<int>(this->num_input, input_layer);
    int n_cycles_load_first_input = load_input_data_fc(ifmap, this->dram_instance, num_input_obtained, num_input_data_first);
    num_input_obtained += num_input_data_first;
    // std::cout<<"load the first input need cycles : "<<n_cycles_load_first_input<<std::endl;
    this->n_cycles += n_cycles_load_first_input;
    PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);  // 切换缓冲区

    // 第一次取权重数据
    int n_cycles_load_first_weight = load_weight_data_fc(filter, this->dram_instance, 0, 0, layer_parameters);
    // std::cout<<"load the first weight need cycles : "<<n_cycles_load_first_weight<<std::endl;
    this->n_cycles += n_cycles_load_first_weight;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);  // 切换缓冲区


    for(int i=0; i<num_weight_buffer_fold; i++){
        // std::cout<<"current weight loop begin cycles : "<<this->n_cycles<<std::endl;
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;
        memset(acc_inter_output, 0, num_neuron_state_buffer_need * sizeof(int)); // 将存储中间结果的数据结构置为0
        for(int j=0; j<num_input_buffer_fold; j++){ 
            // 取相对应的输入数据和权重数据到片上buffer
            // 取下一块输入数据
            int n_cycles_load_next_input;
            if(num_input_buffer_fold == 1){  // 输入buffer足够大，可以容纳所有的输入数据（对于较小的网络规模可以是这样的）
                n_cycles_load_next_input = 0;
                // std::cout<<"load the input need cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                assert(false);
            }

            // 取下一块权重数据
            int n_cycles_load_next_weight;
            if(i+1 < num_weight_buffer_fold){
                n_cycles_load_next_weight = load_weight_data_fc(filter, this->dram_instance, i+1, 0, layer_parameters);
                // std::cout<<"load the next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            } else {
                n_cycles_load_next_weight = 0;
                // std::cout<<"load the next weight need cycles : "<<n_cycles_load_next_weight<<std::endl;
            }

            // 调用脉动阵列计算
            int n_cycles_process_fc = process_fc(i, j, layer_parameters);
            // std::cout<<"process_fc need cycles : "<<n_cycles_process_fc<<std::endl;

            this->n_cycles += std::max(n_cycles_load_next_weight, n_cycles_process_fc);

            // 对中间结果进行累积
            for(int k=0; k<num_neuron_state_buffer_need; k++){
                acc_inter_result[k] += this->neuron_state_buffer[k];
                acc_inter_output[k] += this->output_buffer[k];
            }

            PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);
        }
        // 对累积的结果进行脉冲激活
        for(int k=0; k<num_neuron_state_buffer_need; k++){
            if(acc_inter_output[k] >= 1){
                this->output_buffer[k] = 1;
                this->neuron_state_buffer[k] = 0;
            } else {
                this->output_buffer[k] = 0;
                this->neuron_state_buffer[k] = acc_inter_result[k];
            }
        }

        // std::cout<<"output[0] : "<<this->output_buffer[0]<<std::endl;
        // std::cout<<"neuron[0] : "<<this->neuron_state_buffer[0]<<std::endl;

        // 将结果写入DRAM
        int n_cycles_write_result= store_output_and_neuronstate_data_fc(ofmap, nfmap, dram_instance, i, layer_parameters);
        if(i==num_weight_buffer_fold-1){
            this->n_cycles += n_cycles_write_result;
            // std::cout<<"n_cycles_write_result : "<<n_cycles_write_result<<std::endl;
        } else {
            n_cycles_write_result = 0;
            this->n_cycles += n_cycles_write_result;
            // std::cout<<"n_cycles_write_result : "<<n_cycles_write_result<<std::endl;
        }

        num_input_obtained = 0;
    }

    // 计算完毕，验证写到DRAM的结果是否正确
    matrixMultiply(output_layer, input_layer, 1, filter, ifmap, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th, this->pooling_enabled);

    for(int i=0; i<output_layer; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    delete dram_instance;
    delete[] this->output_buffer;
    delete[] this->output_buffer_cpu;
    delete[] this->neuron_state_buffer;
    delete[] this->neuron_state_buffer_cpu;
    delete this->ppbuf_input;
    delete[] this->input_buffer_0;
    delete[] this->input_buffer_1;
    delete this->ppbuf_weight;
    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;
    delete[] acc_inter_output;
    delete[] acc_inter_result;

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}


// 数据存储方式是通道前置
// 都是单buffer，顺序执行，且未模块化
std::tuple<int*, int*, int*, int*> Controller::runConv_DataFlow_0(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    std::cout<<"Call runConv_DataFlow_0 function !!!"<<std::endl;
    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积
    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // std::cout<<"X_ : "<<X_<<std::endl;
    // std::cout<<"Y_ : "<<Y_<<std::endl;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    // std::cout<<"Below is input data : "<<std::endl;
    // for(int c=0; c<C; c++){
    //     std::cout<<"input channel is : "<<c<<std::endl;
    //     for(int x=0; x<X; x++){
    //         for(int y=0; y<Y; y++){
    //             std::cout<<ifmap[c*X*Y + x*Y + y]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    // }

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 从DRAM中取数据需要的一些（计数器）
    int num_input_buffer_fold = X_;  // 遍历行的次数，即输出特征图的高度
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    int num_input_obtained = 0;
    int num_weight_obtained = 0;
    int weight_base_addr = this->weight_offset;
    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;
        int input_base_addr = this->input_offset;
        num_input_obtained = 0; // 重置

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;

        // 根据需要实例化片上buffer (input buffer 和 weight buffer)
        int num_input_buffer_need = R * Y * C;
        int num_weight_buffer_need = R * S * C * cols;
        if(num_input_buffer_need > ifmap_size){
            num_input_buffer_need = ifmap_size;  // 这种情况在padding不为0的时候可能出现
        }
        assert(num_input_buffer_need <= this->num_input);
        assert(num_weight_buffer_need <= this->num_weight);
        this->input_buffer = new int[num_input_buffer_need];
        this->weight_buffer = new int[num_weight_buffer_need];

        // 从DRAM中读取权重数据
        int num_weight_data = R * S * C * cols;
        int num_weight_read_request = std::ceil(num_weight_data*this->weight_width / (float)dram_instance->dram->GetBusBits());
        int num_weight_read_request_copy = num_weight_read_request;
        while(true){
            if(num_weight_read_request!=0){
                std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(weight_base_addr,false);  // 读请求
                this->read_request_fifo->push(read_request);
                weight_base_addr += this->addr_offset;
                num_weight_read_request--;
            } 

            if(completed_reads == num_weight_read_request_copy){  // 读请求处理完毕
                completed_reads = 0;
                // 模拟将片外数据写入片上buffer
                for(int j=0; j<num_weight_data; j++){
                    this->weight_buffer[j] = filter[num_weight_obtained+j];  // 权重数据的读取顺序在DRAM中是连续读取的
                }
                num_weight_obtained += num_weight_data;
                weight_base_addr = this->weight_offset + ((num_weight_obtained*this->weight_width)/8);
                break;
            }

            this->dram_instance->run();
            this->n_cycles++;
        }
        
        // 从DRAM中取出权重数据之后，开始取输入数据
        // 用于输入数据排序的数据结构
        this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
        this->bankSize = R * S * C;
        std::vector<int> skip_list;
        for(int j=0; j<num_input_buffer_fold; j++){
            if(j>0 && j<=P){
                skip_list.push_back(j);
            }
            if(j>num_input_buffer_fold-1-P && j<=num_input_buffer_fold-1){
                skip_list.push_back(j);
            }
        }

        // std::cout<<"num_input_buffer_fold : "<<num_input_buffer_fold<<std::endl;
        int num_retrieve_input_data = -1;
        for(int j=0; j<num_input_buffer_fold; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<num_input_buffer_fold-1<<std::endl;
            auto it = std::find(skip_list.begin(), skip_list.end(), j);
            if(it == skip_list.end()) {  // 如果不跳过，则取数据
                num_retrieve_input_data++; // 记录取数据的次数，用于计算取数据的地址
                if(j==0){
                    // 从DRAM中取出R行数据，R*Y*C
                    int num_total_read_request = 0;
                    bool only = true;
                    while(true){
                        while(only){
                            for(int c=0; c<C; c++){ // 遍历每个输入通道，因为每个输入通道内的数据是连续的
                                int current_num_read_request = std::ceil(R*Y / (float)dram_instance->dram->GetBusBits());
                                num_total_read_request += current_num_read_request;
                                
                                int read_addr_input = c * X * Y / 8; // 字节地址
                                while(current_num_read_request!=0){
                                    std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(read_addr_input,false);  
                                    this->read_request_fifo->push(read_request);
                                    current_num_read_request--;
                                    read_addr_input += this->addr_offset;
                                    
                                    this->dram_instance->run();
                                    this->n_cycles++;
                                }

                                // 模拟从DRAM中读数据到片上buffer
                                for(int num=0; num<R*Y; num++){
                                    this->input_buffer[c*R*Y + num] = ifmap[c*X*Y + num];
                                }
                            }
                            only = false;
                        }

                        if(completed_reads == num_total_read_request) {
                            completed_reads = 0; 
                            break;
                        }

                        this->dram_instance->run();
                        this->n_cycles++;
                    }

                    // std::cout<<"Below is input_buffer data : "<<std::endl;
                    // for(int c=0; c<C; c++){
                    //     std::cout<<"input channel is : "<<c<<std::endl;
                    //     for(int x=0; x<R; x++){
                    //         for(int y=0; y<Y; y++){
                    //             std::cout<<this->input_buffer[c*R*Y + x*Y + y]<<"  ";
                    //         }
                    //         std::cout<<std::endl;
                    //     }
                    // }

                } else {
                    // 从DRAM中取出1行数据，1*Y*C
                    // input_buffer内部移动(R-1)行数据，(R-1)*Y*C
                    int num_total_read_request = 0;
                    bool only = true;
                    while(true){
                        while(only){
                            for(int c=0; c<C; c++){ // 遍历每个输入通道，从每个输入通道中取出一行数据，数量为Y
                                int current_num_read_request = std::ceil(Y / (float)dram_instance->dram->GetBusBits());
                                num_total_read_request += current_num_read_request;
                                
                                int read_addr_input = ((num_retrieve_input_data + R -1)*Y + c*X*Y) / 8;  //字节地址
                                while(current_num_read_request!=0){
                                    std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(read_addr_input,false);  
                                    this->read_request_fifo->push(read_request);
                                    current_num_read_request--;
                                    read_addr_input += this->addr_offset;

                                    this->dram_instance->run();
                                    this->n_cycles++;
                                }

                                // （R-1）行数据移动
                                for(int r=0; r<(R-1); r++){
                                    for(int y=0; y<Y; y++){  // 一行Y个数据
                                        this->input_buffer[c*R*Y + r*Y + y] = this->input_buffer[c*R*Y + r*Y + y + Y];
                                    }
                                }

                                // 模拟从DRAM中读数据到片上buffer
                                for(int num=0; num<Y; num++){
                                    this->input_buffer[c*R*Y + (R -1)*Y + num] = ifmap[c*X*Y + (num_retrieve_input_data + R -1)*Y + num];
                                }
                            }
                            only = false;
                        }

                        if(completed_reads == num_total_read_request) {
                            completed_reads = 0; 
                            break;
                        }

                        this->dram_instance->run();
                        this->n_cycles++;
                    }

                    // std::cout<<"Below is input_buffer data : "<<std::endl;
                    // for(int c=0; c<C; c++){
                    //     std::cout<<"input channel is : "<<c<<std::endl;
                    //     for(int x=0; x<R; x++){
                    //         for(int y=0; y<Y; y++){
                    //             std::cout<<this->input_buffer[c*R*Y + x*Y + y]<<"  ";
                    //         }
                    //         std::cout<<std::endl;
                    //     }
                    // }
                }
            }
            
            // 片外数据读取到片上buffer之后，对输入数据进行重排
            std::vector<int> im2col_bank(this->bankSize,0);
            int add_0_above; // 考虑到池化，需要补零
            int add_0_below;
            if(j>=0 && j<P){
                add_0_above = P - j;
                add_0_below = 0; 
            } else if(j<=(num_input_buffer_fold-1) && j>(num_input_buffer_fold-1-P)){
                add_0_above = 0;
                add_0_below = j - (num_input_buffer_fold-1-P);
            } else {
                add_0_above = 0; 
                add_0_below = 0;
            }
            
            // std::cout<<"add_0_above : "<<add_0_above<<std::endl;
            // std::cout<<"add_0_below : "<<add_0_below<<std::endl;

            // 开始对片上 input buffer中的数据进行排序
            int num_tile = 0;
            int rows = 0;
            for(int k=0; k<Y_; k++){
                if(k==0){  // 第一个卷积窗口基地址计算不同，单独处理
                    if(add_0_above != 0){ // 考虑池化，上方补零
                        //std::cout<<k<<std::endl;
                        for(int c=0; c<C; c++){  // input_buffer中的数据是逐通道存储的，一个通道内的数据是连续存储的，所以要遍历每个通道
                            //std::cout<<c<<std::endl;
                            for(int num=0; num<add_0_above*S; num++){ // 补add_0_above行0
                                im2col_bank[c*R*S + num] = 0;
                            }
                        }
                        for(int c=0; c<C; c++){  // 从input_buffer中取(R-add_0_above)行数据
                            for(int r=add_0_above; r<R; r++){  // 遍历存储真实数据的行
                                for(int q=0; q<P; q++){  // 考虑池化，左侧P列0
                                    im2col_bank[c*R*S + r*S + q] = 0;
                                }
                                for(int q=P; q<S; q++){ // 取真实的数据
                                    im2col_bank[c*R*S + r*S + q] = this->input_buffer[c*R*Y + (r-add_0_above)*Y +(q-P)];
                                    this->n_cycles++;
                                }
                            }
                        }

                    } else if(add_0_below != 0){ // 下方补零
                        for(int c=0; c<C; c++){
                            for(int r=0; r<(R-add_0_below); r++){ // 从input_buffer中取数据，同时考虑左侧P列0
                                for(int q=0; q<P; q++){
                                    im2col_bank[c*R*S + r*S + q] = 0;
                                }
                                for(int q=P; q<S; q++){
                                    im2col_bank[c*R*S + r*S + q] = this->input_buffer[c*R*Y + (r+add_0_below)*Y + (q-P)];
                                    this->n_cycles++;
                                }
                            }
                        }
                        for(int c=0; c<C; c++){  // 补零
                            for(int num=0; num<add_0_below*S; num++){
                                im2col_bank[c*R*S + (R-add_0_below)*S + num] = 0;
                            }
                        }

                    } else { // 不用补零
                        //std::cout<<"k==0"<<std::endl;
                        for(int c=0; c<C; c++){
                            for(int r=0; r<R; r++){
                                for(int q=0; q<P; q++){  // 考虑左侧P列0
                                    im2col_bank[c*R*S + r*S + q] = 0;
                                }
                                for(int q=P; q<S; q++){
                                    im2col_bank[c*R*S + r*S + q] = this->input_buffer[c*R*Y + r*Y + (q-P)];
                                    this->n_cycles++;
                                }
                            }
                        }
                    }

                    // std::cout<<"The sorted data is : "<<std::endl;
                    // for(int p=0; p<this->bankSize; p++){
                    //     std::cout<<im2col_bank[p]<<"  ";
                    // }
                    // std::cout<<std::endl;

                    this->input_arranged.push_back(im2col_bank);
                    rows++;
                    //std::cout<<"debug"<<std::endl;
                } else {
                    //std::cout<<"k>0"<<std::endl;
                    if(k >= Y_-P){  // 移位和右侧补0
                        //std::cout<<"debug 1"<<std::endl;
                        for(int c=0; c<C; c++){
                            for(int r=0; r<R; r++){
                                for(int q=0; q<(S-1); q++){  // 移位
                                    im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                                }
                                im2col_bank[c*R*S + r*S + (S-1)] = 0;
                            }
                        }
                    } else {  // 移位和取数据
                        //std::cout<<"debug 2"<<std::endl;
                        if(add_0_above != 0){
                            for(int c=0; c<C; c++){  // 补0
                                for(int num=0; num<add_0_above*S; num++){
                                    im2col_bank[c*R*S + num] = 0;
                                }
                            }
                            for(int c=0; c<C; c++){
                                for(int r=add_0_above; r<R; r++){
                                    for(int q=0; q<(S-1); q++){  // 移位
                                        im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                                    }
                                    // 取数据
                                    im2col_bank[c*R*S + r*S + (S-1)] = this->input_buffer[c*R*Y + (r-add_0_above)*Y + (S-P+k-1)];
                                    this->n_cycles++;
                                }
                            }

                        } else if(add_0_below !=0){
                            for(int c=0; c<C; c++){
                                for(int r=0; r<(R-add_0_below); r++){
                                    for(int q=0; q<(S-1); q++){
                                        im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                                    }
                                    // 取数据
                                    im2col_bank[c*R*S + r*S + (S-1)] = this->input_buffer[c*R*Y + (r+add_0_below)*Y + (S-P+k-1)];
                                    this->n_cycles++;
                                }
                            }
                            for(int c=0; c<C; c++){
                                for(int num=0; num<add_0_below*S; num++){
                                    im2col_bank[c*R*S + (R-add_0_below)*S + num]=0;
                                }
                            }

                        } else {
                            //std::cout<<"debug in k>0"<<std::endl;
                            for(int c=0; c<C; c++){
                                for(int r=0; r<R; r++){
                                    for(int q=0; q<(S-1); q++){  // 移位
                                        im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                                    }
                                    // 取数据
                                    im2col_bank[c*R*S + r*S + (S-1)] = this->input_buffer[c*R*Y + r*Y + (S-P+k-1)];
                                    this->n_cycles++;
                                }
                            }
                        }
                    }

                    // std::cout<<"The sorted data is : "<<std::endl;
                    // for(int p=0; p<this->bankSize; p++){
                    //     std::cout<<im2col_bank[p]<<"  ";
                    // }
                    // std::cout<<std::endl;

                    this->input_arranged.push_back(im2col_bank);
                    rows++;

                    if(rows==this->numBanks || k == Y_-1){
                        std::cout<<"rows : "<<rows<<std::endl;
                        std::cout<<"num_tile : "<<num_tile<<std::endl;
                        num_tile++;

                        // 将this->input_arranged转化为数组
                        int total = rows * this->bankSize;
                        this->spikes = new int[total];
                        int index = 0;
                        for(const auto& row : input_arranged){
                            for(int val : row){
                                this->spikes[index++] = val;
                            }
                        }

                        // 根据需要例化 output buffer 和 neuron_state buffer
                        unsigned int num_output_buffer_need = rows * cols;
                        unsigned int num_neuron_state_buffer_need = rows * cols;
                        assert(num_output_buffer_need <= num_output);
                        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                        this->output_buffer = new int[num_output_buffer_need]();
                        this->output_buffer_cpu = new int[num_output_buffer_need]();
                        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                        this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                        // 例化脉动阵列组件，执行矩阵乘法
                        Stonne* stonne_instance = new Stonne(this->stonne_cfg);
                        matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                        stonne_instance->loadDenseGEMM(this->layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                        stonne_instance->loadGEMMTile(cols,1,rows);
                        stonne_instance->run();

                        this->n_cycles += stonne_instance->n_cycles;
                        // 对当前tile计算结果进行验证
                        for(int m = 0; m<num_output_buffer_need; m++){
                            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                            if(difference>0){
                                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                assert(false);
                            }
                        }
                        
                        std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;
                        std::cout<<"first output is : "<<this->output_buffer[0]<<std::endl;
                        std::cout<<"first neuron_state is : "<<this->neuron_state_buffer[0]<<std::endl;

                        // 将计算结果写入DRAM
                        // output_buffer中一列数据是一个输出通道的数据，将一列数据打包，写入DRAM
                        int num_total_write_request = 0;
                        bool only = true;
                        while(true){
                            while(only){
                                for(int p=0; p<cols; p++){  // 遍历每一列

                                    std::vector<int> packed_col_output(rows,0);
                                    std::vector<int> packed_col_neuron_state(rows,0);
                                    for(int q=0; q<rows; q++){
                                        packed_col_output[q] = this->output_buffer[q*cols+p];
                                        packed_col_neuron_state[q] = this->neuron_state_buffer[q*cols+p];
                                    }
                                    this->n_cycles++; // 假设打包需要一个周期

                                    int current_num_output_write_request = std::ceil(rows / (float)dram_instance->dram->GetBusBits());
                                    int current_num_neuron_state_write_request = std::ceil(rows*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                    int current_num_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                    num_total_write_request += current_num_total_write_request;

                                    int write_addr_output = this->output_offset + ((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows) / 8;
                                    int write_addr_neuron_state = this->neuron_state_offset + ((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                    while(current_num_total_write_request!=0){
                                        if(current_num_output_write_request!=0){
                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                            this->write_request_fifo->push(write_request);
                                            write_addr_output += this->addr_offset;
                                            current_num_output_write_request--;
                                        } else if(current_num_neuron_state_write_request!=0){
                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                            this->write_request_fifo->push(write_request);
                                            write_addr_neuron_state += this->addr_offset;
                                            current_num_neuron_state_write_request--;
                                        }
                                        this->dram_instance->run();
                                        this->n_cycles++;
                                        current_num_total_write_request--;
                                    }
                                    // 模拟将片上buffer中的数据写入DRAM
                                    for(int q=0; q<rows; q++){
                                        ofmap[((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows) + q] = packed_col_output[q];
                                        nfmap[((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows) + q] = packed_col_neuron_state[q];
                                    }
                                }
                                only = false;
                            }

                            if(completed_writes == num_total_write_request) {
                                completed_writes = 0; 
                                break;
                            }

                            this->dram_instance->run();
                            this->n_cycles++;
                        }

                        // 重置数据，开始下一tile计算
                        rows = 0;
                        this->input_arranged.clear();
                        delete stonne_instance;
                        delete[] this->spikes;
                        delete[] this->output_buffer;
                        delete[] this->output_buffer_cpu;
                        delete[] this->neuron_state_buffer;
                        delete[] this->neuron_state_buffer_cpu;
                    }
                }
            }
        }
        delete[] this->input_buffer;
        delete[] this->weight_buffer;
    }

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    conv_compute_dataflow(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    // 检查输出
    // std::cout<<"below is simlator output : "<<std::endl;
    // for(int i=0; i<X_*Y_*K; i++){
    //     std::cout<<ofmap[i]<<"   ";
    // }
    // std::cout<<std::endl;
    // std::cout<<"below is simlator neuron state : "<<std::endl;
    // for(int i=0; i<X_*Y_*K; i++){
    //     std::cout<<nfmap[i]<<"   ";
    // }
    // std::cout<<std::endl;

    // std::cout<<"below is cpu output : "<<std::endl;
    // for(int i=0; i<X_*Y_*K; i++){
    //     std::cout<<ofmap_cpu[i]<<"   ";
    // }
    // std::cout<<std::endl;

    // std::cout<<"below is cpu neuron state : "<<std::endl;
    // for(int i=0; i<X_*Y_*K; i++){
    //     std::cout<<nfmap_cpu[i]<<"   ";
    // }
    // std::cout<<std::endl;


    for(int i=0; i<X_*Y_*K; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

// input_buffer设置为双buffer，需要im2col单元实现加padding逻辑(process_conv)
std::tuple<int*, int*, int*, int*> Controller::runConv_DataFlow_1(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    std::cout<<"Call runConv_DataFlow_1 function !!! 1"<<std::endl;
    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积
    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 取输入数据，考虑到加padding，下面是因为加padding而存在的跳过逻辑
    for(int i=0; i<X_; i++){
        if(i>0 && i<=P){
            this->skip_list.push_back(i);
        }
        if(i>X_-1-P && i<=X_-1){
            this->skip_list.push_back(i);
        }
    }

    // 从DRAM中取数据需要的（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    int num_weight_obtained = 0;
    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;

        // 根据需要实例化片上buffer (input buffer 和 weight buffer)
        int num_input_buffer_need = R * Y * C;
        int num_weight_buffer_need = R * S * C * cols;
        if(num_input_buffer_need > ifmap_size){
            num_input_buffer_need = ifmap_size;  // 这种情况在padding不为0的时候可能出现
        }
        assert(num_input_buffer_need <= this->num_input);
        assert(num_weight_buffer_need <= this->num_weight);
        //this->input_buffer = new int[num_input_buffer_need];
        this->weight_buffer = new int[num_weight_buffer_need];
        // 累积每R行输入数据中每个tile的输出和神经元状态
        // 根据需要例化 output buffer 和 neuron_state buffer
        unsigned int num_output_buffer_need = Y_ * cols;
        unsigned int num_neuron_state_buffer_need = Y_ * cols;
        assert(num_output_buffer_need <= num_output);
        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
        this->output_buffer = new int[num_output_buffer_need]();
        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();

        // 初始化乒乓buffer
        this->ppbuf_input = new PingPong_Buffer;
        this->input_buffer_0 = new int[num_input_buffer_need];
        this->input_buffer_1 = new int[num_input_buffer_need];
        PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

        // 从DRAM中读取权重数据
        int num_weight_data = R * S * C * cols;
        int n_cycles_load_weight = load_weight_data(filter, this->dram_instance, num_weight_obtained, num_weight_data); // 调用函数**************************
        num_weight_obtained += num_weight_data;
        this->n_cycles += n_cycles_load_weight;
        std::cout<<"load weight cycles : "<<n_cycles_load_weight<<std::endl;
        
        this->num_retrieve_input_data = -1;  // 不能删

        // 加载数据到用于计算的buffer，加载到next_buffer
        int n_cycles_load_input = load_input_data_step2_ppbuffer(ifmap, this->dram_instance, 0, layer_parameters);
        // 切换，切换之后，上一步加载的数据到了current_buffer，用于下面的计算，next_buffer是空的，用于加载下一块数据（和计算同时进行）
        PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
        this->n_cycles += n_cycles_load_input;
        std::cout<<"load the first input cycles : "<<n_cycles_load_input<<std::endl;

        std::cout<<"current weight loop begin cycles : "<<this->n_cycles<<std::endl;
        for(int j=0; j<X_; j++){ 
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;
            if(j+1<X_){
                // 加载下一块数据到next_buffer中
                int n_cycles_load_input = load_input_data_step2_ppbuffer(ifmap, this->dram_instance, j+1, layer_parameters);  // 调用函数**************************
                //this->n_cycles += n_cycles_load_input;
                std::cout<<"load input cycles : "<<n_cycles_load_input<<std::endl;
            } else {
                int n_cycles_load_input = 0;
            }
            
            // 重排和计算，使用current_buffer中的数据进行计算
            int n_cycles_process = process_conv(i, j, cols, layer_parameters);
            //this->n_cycles += n_cycles_process;
            std::cout<<"process_conv cycles : "<<n_cycles_process<<std::endl;

            // 将计算结果写入DRAM
            int n_cycles_write = store_output_and_neuronstate_data(ofmap, nfmap, this->dram_instance, i, j, cols, layer_parameters);
            //this->n_cycles += n_cycles_write;
            std::cout<<"write result cycles : "<<n_cycles_write<<std::endl;
            std::cout<<"process_conv and write result cycles : "<<(n_cycles_process+n_cycles_write)<<std::endl;

            this->n_cycles += std::max(n_cycles_load_input, n_cycles_process+n_cycles_write);

            std::cout<<std::endl;
            std::cout<<"Global cycles : "<<this->n_cycles<<std::endl;

            // input双缓冲区切换
            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
        }

        delete[] this->output_buffer;
        delete[] this->neuron_state_buffer;
        //delete[] this->input_buffer;
        delete[] this->weight_buffer;

        delete this->ppbuf_input;
        delete[] this->input_buffer_0;
        delete[] this->input_buffer_1;
    }

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    //std::cout<<"begin final test"<<std::endl;
    conv_compute_dataflow(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    for(int i=0; i<X_*Y_*K; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;
    return {ifmap, filter, ofmap, nfmap};
}

// input_buffer设置为双buffer，在从DRAM中读取输入数据时完成加padding操作，进一步将input_arranged_buffer设置为双buffer(process_conv3)，单buffer(process_conv3)
std::tuple<int*, int*, int*, int*> Controller::runConv_DataFlow_2(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    std::cout<<"Call runConv_DataFlow_2 function !!! 1"<<std::endl;
    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积
    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 取输入数据，考虑到加padding，下面是因为加padding而存在的跳过逻辑
    // for(int i=0; i<X_; i++){
    //     if(i>0 && i<=P){
    //         this->skip_list.push_back(i);
    //     }
    //     if(i>X_-1-P && i<=X_-1){
    //         this->skip_list.push_back(i);
    //     }
    // }

    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    // 排序buffer设置为双buffer
    // this->bankSize = R*S*C;
    // int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    // this->input_arranged_buffer = new int [input_arranged_buffer_size];
    // this->ppbuf_input_arranged = new PingPong_Buffer;
    // this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    // this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    // PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);


    // 从DRAM中取数据需要的（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    int num_weight_obtained = 0;
    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;

        // 根据需要实例化片上buffer (input buffer 和 weight buffer)
        int num_input_buffer_need = R * Y_padded * C;  // 加padding
        int num_weight_buffer_need = R * S * C * cols;
        if(num_input_buffer_need > ifmap_size){
            num_input_buffer_need = ifmap_size;  // 这种情况在padding不为0的时候可能出现
        }
        assert(num_input_buffer_need <= this->num_input);
        assert(num_weight_buffer_need <= this->num_weight);
        //this->input_buffer = new int[num_input_buffer_need];
        this->weight_buffer = new int[num_weight_buffer_need];
        // 累积每R行输入数据中每个tile的输出和神经元状态
        // 根据需要例化 output buffer 和 neuron_state buffer
        unsigned int num_output_buffer_need = Y_ * cols;
        unsigned int num_neuron_state_buffer_need = Y_ * cols;
        assert(num_output_buffer_need <= num_output);
        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
        this->output_buffer = new int[num_output_buffer_need]();
        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();

        // 初始化乒乓buffer
        this->ppbuf_input = new PingPong_Buffer;
        this->input_buffer_0 = new int[num_input_buffer_need];
        this->input_buffer_1 = new int[num_input_buffer_need];
        PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

        // 从DRAM中读取权重数据
        int num_weight_data = R * S * C * cols;
        int n_cycles_load_weight = load_weight_data(filter, this->dram_instance, num_weight_obtained, num_weight_data); // 调用函数**************************
        num_weight_obtained += num_weight_data;
        this->n_cycles += n_cycles_load_weight;
        // std::cout<<"num_weight_data : "<<num_weight_data<<std::endl;
        std::cout<<"load weight cycles : "<<n_cycles_load_weight<<std::endl;
        
        //this->num_retrieve_input_data = -1;  // 不能删

        // 加载数据到用于计算的buffer，加载到next_buffer
        int n_cycles_load_input = load_input_data_step1_ppbuffer(layer_id, ifmap, this->dram_instance, 0, layer_parameters);
        // 切换，切换之后，上一步加载的数据到了current_buffer，用于下面的计算，next_buffer是空的，用于加载下一块数据（和计算同时进行）
        PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
        this->n_cycles += n_cycles_load_input;
        std::cout<<"load the first input cycles : "<<n_cycles_load_input<<std::endl;

        std::cout<<"current weight loop begin cycles : "<<this->n_cycles<<std::endl;
        for(int j=0; j<X_; j++){ 
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;
            int n_cycles_load_input;
            // 加载下一块数据到next_buffer中
            if(j+1<X_){
                n_cycles_load_input = load_input_data_step1_ppbuffer(layer_id, ifmap, this->dram_instance, j+1, layer_parameters);  // 调用函数**************************
                //this->n_cycles += n_cycles_load_input;
                std::cout<<"load input cycles : "<<n_cycles_load_input<<std::endl;
            } else {
                n_cycles_load_input = 0;
            }
            
            // 重排和计算，使用current_buffer中的数据进行计算
            int n_cycles_process = process_conv_2(i, j, cols, layer_parameters);
            //this->n_cycles += n_cycles_process;
            std::cout<<"process_conv cycles : "<<n_cycles_process<<std::endl;

            // 将计算结果写入DRAM
            int n_cycles_write = store_output_and_neuronstate_data(ofmap, nfmap, this->dram_instance, i, j, cols, layer_parameters);
            //this->n_cycles += n_cycles_write;
            std::cout<<"write result cycles : "<<n_cycles_write<<std::endl;
            std::cout<<"process_conv and write result cycles : "<<(n_cycles_process+n_cycles_write)<<std::endl;

            this->n_cycles += std::max(n_cycles_load_input, n_cycles_process+n_cycles_write);

            std::cout<<std::endl;
            std::cout<<"Global cycles : "<<this->n_cycles<<std::endl;

            // input双缓冲区切换
            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
        }

        delete[] this->output_buffer;
        delete[] this->neuron_state_buffer;
        //delete[] this->input_buffer;
        delete[] this->weight_buffer;

        delete this->ppbuf_input;
        delete[] this->input_buffer_0;
        delete[] this->input_buffer_1;
    }

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    //std::cout<<"begin final test"<<std::endl;
    conv_compute_dataflow(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    for(int i=0; i<X_*Y_*K; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    this->records.clear();
    delete[] ofmap_cpu;
    delete[] nfmap_cpu;
    return {ifmap, filter, ofmap, nfmap};
}

// input_arranged_buffer 和 weight_buffer 和 input_buffer都设置为双buffer，
// weight_buffer 取下一次计算所需数据的时间是：input_buffer第二次取数据之后（较简单的控制逻辑）
std::tuple<int*, int*, int*, int*> Controller::runConv_DataFlow_3(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                   Call runConv_DataFlow_3 function                                             "<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;

    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积
    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // if(layer_id == 5){
    //     std::cout<<std::endl;
    //     std::cout<<"ifmap data is :"<<std::endl;
    //     for(int c=0; c<C; c++){
    //         std::cout<<"input channel is : "<<c<<std::endl;
    //         for(int x=0; x<X; x++){
    //             for(int y=0; y<Y; y++){
    //                 std::cout<<ifmap[c*X*Y + x*Y +y]<<" ";
    //             }
    //             std::cout<<std::endl;
    //         }
    //         std::cout<<std::endl;
    //     }

    //     std::cout<<std::endl;
    // }

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 取输入数据，考虑到加padding，下面是因为加padding而存在的跳过逻辑
    // for(int i=0; i<X_; i++){
    //     if(i>0 && i<=P){
    //         this->skip_list.push_back(i);
    //     }
    //     if(i>X_-1-P && i<=X_-1){
    //         this->skip_list.push_back(i);
    //     }
    // }
    // std::cout<<"X_ : "<<X_<<std::endl;
    // std::cout<<"P : "<<P<<std::endl;

    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            // std::cout<<"i : "<<i<<std::endl;
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
            // std::cout<<r.num_rows<<std::endl;
            // std::cout<<r.start_rows<<std::endl;
            // std::cout<<r.add_0_above<<std::endl;
            // std::cout<<r.add_0_below<<std::endl;
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    // std::cout<<this->records[15].num_rows<<std::endl;
    // std::cout<<this->records[15].start_rows<<std::endl;
    // std::cout<<this->records[15].add_0_above<<std::endl;
    // std::cout<<this->records[15].add_0_below<<std::endl;

    // // 排序buffer设置为双buffer
    // this->bankSize = R*S*C;
    // int input_arranged_buffer_size = this->bankSize*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    // this->ppbuf_input_arranged = new PingPong_Buffer;
    // this->input_arranged_buffer_0 = new int[input_arranged_buffer_size];
    // this->input_arranged_buffer_1 = new int[input_arranged_buffer_size];
    // PingPongBuffer_Init(this->ppbuf_input_arranged, this->input_arranged_buffer_0, this->input_arranged_buffer_1);

    // 权重buffer设置为双buffer
    int num_weight_buffer_need = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    assert(num_weight_buffer_need <= this->num_weight);
    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    // 从DRAM中取数据需要的（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    int num_weight_obtained = 0;

    // 第一次从DRAM中加载权重数据，加载到next_buffer中
    int num_weight_data; // 要从DRAM中取的权重个数
    if(num_weight_buffer_fold>1){
        num_weight_data = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    } else {
        num_weight_data = R * S * C * K;
    }
    int n_cycles_load_first_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
    //std::cout<<"load the first weights cycles : "<<n_cycles_load_first_weight<<std::endl;
    this->n_cycles += n_cycles_load_first_weight;
    num_weight_obtained += num_weight_data;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    // 根据需要实例化片上buffer (input buffer 和 weight buffer)
    int num_input_buffer_need = R * Y_padded * C;  // 加padding
    // int num_weight_buffer_need = R * S * C * cols;
    // if(num_input_buffer_need > ifmap_size){
    //     num_input_buffer_need = ifmap_size;  // 这种情况在padding不为0的时候可能出现
    // }
    assert(num_input_buffer_need <= this->num_input);

    // 初始化乒乓buffer
    this->ppbuf_input = new PingPong_Buffer;
    this->input_buffer_0 = new int[num_input_buffer_need];
    this->input_buffer_1 = new int[num_input_buffer_need];

    PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;

        // assert(num_weight_buffer_need <= this->num_weight);
        // this->input_buffer = new int[num_input_buffer_need];
        // this->weight_buffer = new int[num_weight_buffer_need];
        // 累积每R行输入数据中每个tile的输出和神经元状态
        // 根据需要例化 output buffer 和 neuron_state buffer
        unsigned int num_output_buffer_need = Y_ * cols;
        unsigned int num_neuron_state_buffer_need = Y_ * cols;
        assert(num_output_buffer_need <= num_output);
        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
        this->output_buffer = new int[num_output_buffer_need]();
        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
        
        //this->num_retrieve_input_data = -1;  // 不能删

        
        if(layer_id == 5){
            std::cout<<"debug0"<<std::endl;
        }

        // 加载数据到用于计算的buffer，加载到next_buffer
        int n_cycles_load_first_input = load_input_data_step1_ppbuffer(layer_id, ifmap, this->dram_instance, 0, layer_parameters);
        // 切换，切换之后，上一步加载的数据到了current_buffer，用于下面的计算，next_buffer是空的，用于加载下一块数据（和计算同时进行）
        PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
        this->n_cycles += n_cycles_load_first_input;
        //std::cout<<"load the first input cycles : "<<n_cycles_load_first_input<<std::endl;

        
        if(layer_id == 5){
            std::cout<<"debug1"<<std::endl;
        }
        
        //std::cout<<"current weight loop begin cycles : "<<this->n_cycles<<std::endl;
        for(int j=0; j<X_; j++){ 
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;
            
            // 加载下一块数据到next_buffer中
            int n_cycles_load_next_input;
            if(j+1<X_){
                //std::cout<<" below load next input cycles"<<std::endl;
                n_cycles_load_next_input = load_input_data_step1_ppbuffer(layer_id, ifmap, this->dram_instance, j+1, layer_parameters);  // 调用函数**************************
                //this->n_cycles += n_cycles_load_input;
                //std::cout<<"load next input cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                //std::cout<<"load next input cycles : "<<n_cycles_load_next_input<<std::endl;
            }

            // 在第二次从DRAM中取输入数据之后，与计算同时进行加载下一块权重数据
            int n_cycles_load_next_weight;
            if(i+1<num_weight_buffer_fold && j==0){  // 取下一块权重
                int start_col_next = (i+1)*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col_next = std::min<int>(start_col_next+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
                int cols_next = end_col_next - start_col_next;
                num_weight_data = R * S * C * cols_next;  // 要从DRAM中取出的数据个数
                n_cycles_load_next_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
                //std::cout<<"load next weight cycles : "<<n_cycles_load_next_weight<<std::endl;
                num_weight_obtained += num_weight_data;
            } else {
                n_cycles_load_next_weight = 0;
                //std::cout<<"load next weight cycles : "<<n_cycles_load_next_weight<<std::endl;
            }
            
            // 重排和计算，使用current_buffer中的数据进行计算

            //std::cout<<"begin compute : "<<std::endl;

            int n_cycles_process = process_conv_3(layer_id, i, j, cols, layer_parameters);
            //this->n_cycles += n_cycles_process;
            //std::cout<<"process_conv cycles : "<<n_cycles_process<<std::endl;

            if(layer_id == 5){
                std::cout<<"debug    0"<<std::endl;
            }

            // 将计算结果写入DRAM
            int n_cycles_write = store_output_and_neuronstate_data(ofmap, nfmap, this->dram_instance, i, j, cols, layer_parameters);
            //this->n_cycles += n_cycles_write;
            //std::cout<<"write result cycles : "<<n_cycles_write<<std::endl;
            //std::cout<<"process_conv and write result cycles : "<<(n_cycles_process+n_cycles_write)<<std::endl;

            if(layer_id == 5){
                std::cout<<"debug    1"<<std::endl;
            }

            if(i==num_weight_buffer_fold-1 && j==X_-1){
                this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process+n_cycles_write);
                // std::cout<<std::endl;
                //std::cout<<"Global cycles : "<<this->n_cycles<<std::endl;
            } else {
                this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process);
                std::cout<<std::endl;
                //std::cout<<"Global cycles : "<<this->n_cycles<<std::endl;
            }
            if(layer_id == 5){
                std::cout<<"debug    2"<<std::endl;
            }

            // input双缓冲区切换
            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
            
        }

        if(layer_id == 5){
            std::cout<<"debug    3"<<std::endl;
        }

        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1); // 切换缓冲区

        delete[] this->output_buffer;
        delete[] this->neuron_state_buffer;

    }

    if(layer_id == 5){
        std::cout<<"debug    4"<<std::endl;
    }
    delete this->ppbuf_weight;
    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;

    if(layer_id == 5){
        std::cout<<"debug    5"<<std::endl;
    }

    delete this->ppbuf_input;

    if(layer_id == 5){
        std::cout<<"debug    6"<<std::endl;

        std::cout<<this->input_buffer_0<<std::endl;
        std::cout<<*(int*)input_buffer_0<<std::endl;
    }

    delete[] this->input_buffer_0;

    if(layer_id == 5){
        std::cout<<"debug    7"<<std::endl;
    }

    delete[] this->input_buffer_1;

    if(layer_id == 5){
        std::cout<<"debug    8"<<std::endl;
    }



    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    //std::cout<<"begin final test"<<std::endl;
    conv_compute_dataflow(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    for(int i=0; i<X_*Y_*K; i++){
        float difference = fabs(ofmap[i]-ofmap_cpu[i]) + fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    // std::cout<<std::endl;
    // std::cout<<"The conv compute output result is : "<<std::endl;
    // for(int k=0; k<K; k++){
    //     std::cout<<"output channel : "<<std::endl;
    //     for(int p=0; p<X_; p++){
    //         for(int q=0; q<Y_; q++){
    //             std::cout<<ofmap[k*X_*Y_ + p*Y_ +q]<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // std::cout<<std::endl;
    // std::cout<<"The conv compute neuron_state result is : "<<std::endl;
    // for(int k=0; k<K; k++){
    //     std::cout<<"output channel : "<<std::endl;
    //     for(int p=0; p<X_; p++){
    //         for(int q=0; q<Y_; q++){
    //             std::cout<<nfmap[k*X_*Y_ + p*Y_ +q]<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    this->records.clear();

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

// 都是单buffer，顺序执行，未模块化，加padding操作由im2col单元实现
std::tuple<int*, int*, int*, int*> Controller::runConvandPooling_DataFlow_0(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积
    std::cout<<"Call runConvandPooling_DataFlow_0 function !!! 1"<<std::endl;
    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 从DRAM中取数据需要的一些（计数器）
    int num_input_buffer_fold = X_;  // 遍历行的次数，即输出特征图的高度
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    int num_input_obtained = 0;
    int num_weight_obtained = 0;
    int weight_base_addr = this->weight_offset;
    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;
        int input_base_addr = this->input_offset;
        num_input_obtained = 0; // 重置

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;

        // 建模片上SRAM，用于累积卷积的输出进行池化
        int sram_size = 2 * Y_ * cols; 
        this->on_chip_sram = new int[sram_size]();

        // 根据需要实例化片上buffer (input buffer 和 weight buffer)
        int num_input_buffer_need = R * Y * C;
        int num_weight_buffer_need = R * S * C * cols;
        if(num_input_buffer_need > ifmap_size){
            num_input_buffer_need = ifmap_size;  // 这种情况在padding不为0的时候可能出现
        }
        assert(num_input_buffer_need <= this->num_input);
        assert(num_weight_buffer_need <= this->num_weight);
        this->input_buffer = new int[num_input_buffer_need];
        this->weight_buffer = new int[num_weight_buffer_need];

        // 从DRAM中读取权重数据
        int num_weight_data = R * S * C * cols;
        int num_weight_read_request = std::ceil(num_weight_data*this->weight_width / (float)dram_instance->dram->GetBusBits());
        int num_weight_read_request_copy = num_weight_read_request;
        while(true){
            if(num_weight_read_request!=0){
                std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(weight_base_addr,false);  // 读请求
                this->read_request_fifo->push(read_request);
                weight_base_addr += this->addr_offset;
                num_weight_read_request--;
            } 

            if(completed_reads == num_weight_read_request_copy){  // 读请求处理完毕
                completed_reads = 0;
                // 模拟将片外数据写入片上buffer
                for(int j=0; j<num_weight_data; j++){
                    this->weight_buffer[j] = filter[num_weight_obtained+j];  // 权重数据的读取顺序在DRAM中是连续读取的
                }
                num_weight_obtained += num_weight_data;
                weight_base_addr = this->weight_offset + ((num_weight_obtained*this->weight_width)/8);
                break;
            }

            this->dram_instance->run();
            this->n_cycles++;
        }
        
        // 从DRAM中取出权重数据之后，开始取输入数据
        // 用于输入数据排序的数据结构
        this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
        this->bankSize = R * S * C;
        std::vector<int> skip_list;
        for(int j=0; j<num_input_buffer_fold; j++){
            if(j>0 && j<=P){
                skip_list.push_back(j);
            }
            if(j>num_input_buffer_fold-1-P && j<=num_input_buffer_fold-1){
                skip_list.push_back(j);
            }
        }

        // std::cout<<"num_input_buffer_fold : "<<num_input_buffer_fold<<std::endl;
        int count_rows = 0; // 用于累加卷积的输出到片上SRAM计数
        int num_retrieve_input_data = -1; // 记录取数据的次数，用于计算取数据的地址
        for(int j=0; j<num_input_buffer_fold; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<num_input_buffer_fold-1<<std::endl;
            auto it = std::find(skip_list.begin(), skip_list.end(), j);
            if(it == skip_list.end()) {  // 如果不跳过，则取数据
                num_retrieve_input_data++; // 记录取数据的次数，用于计算取数据的地址
                if(j==0){
                    // 从DRAM中取出R行数据，R*Y*C
                    int num_total_read_request = 0;
                    bool only = true;
                    while(true){
                        while(only){
                            for(int c=0; c<C; c++){ // 遍历每个输入通道，因为每个输入通道内的数据是连续的
                                int current_num_read_request = std::ceil(R*Y / (float)dram_instance->dram->GetBusBits());
                                num_total_read_request += current_num_read_request;
                                
                                int read_addr_input = c * X * Y / 8; // 字节地址
                                while(current_num_read_request!=0){
                                    std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(read_addr_input,false);  
                                    this->read_request_fifo->push(read_request);
                                    current_num_read_request--;
                                    read_addr_input += this->addr_offset;
                                    
                                    this->dram_instance->run();
                                    this->n_cycles++;
                                }

                                // 模拟从DRAM中读数据到片上buffer
                                for(int num=0; num<R*Y; num++){
                                    this->input_buffer[c*R*Y + num] = ifmap[c*X*Y + num];
                                }
                            }
                            only = false;
                        }

                        if(completed_reads == num_total_read_request) {
                            completed_reads = 0; 
                            break;
                        }

                        this->dram_instance->run();
                        this->n_cycles++;
                    }
                } else {
                    // 从DRAM中取出1行数据，1*Y*C
                    // input_buffer内部移动(R-1)行数据，(R-1)*Y*C
                    int num_total_read_request = 0;
                    bool only = true;
                    while(true){
                        while(only){
                            for(int c=0; c<C; c++){ // 遍历每个输入通道，从每个输入通道中取出一行数据，数量为Y
                                int current_num_read_request = std::ceil(Y / (float)dram_instance->dram->GetBusBits());
                                num_total_read_request += current_num_read_request;
                                
                                int read_addr_input = ((num_retrieve_input_data + R -1)*Y + c*X*Y) / 8;  //字节地址
                                while(current_num_read_request!=0){
                                    std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(read_addr_input,false);  
                                    this->read_request_fifo->push(read_request);
                                    current_num_read_request--;
                                    read_addr_input += this->addr_offset;

                                    this->dram_instance->run();
                                    this->n_cycles++;
                                }

                                // （R-1）行数据移动
                                for(int r=0; r<(R-1); r++){
                                    for(int y=0; y<Y; y++){  // 一行Y个数据
                                        this->input_buffer[c*R*Y + r*Y + y] = this->input_buffer[c*R*Y + r*Y + y + Y];
                                    }
                                }

                                // 模拟从DRAM中读数据到片上buffer
                                for(int num=0; num<Y; num++){
                                    this->input_buffer[c*R*Y + (R -1)*Y + num] = ifmap[c*X*Y + (num_retrieve_input_data + R -1)*Y + num];
                                }
                            }
                            only = false;
                        }

                        if(completed_reads == num_total_read_request) {
                            completed_reads = 0; 
                            break;
                        }

                        this->dram_instance->run();
                        this->n_cycles++;
                    }
                }
            }
            
            // 片外数据读取到片上buffer之后，对输入数据进行重排
            std::vector<int> im2col_bank(this->bankSize,0);
            int add_0_above; // 考虑到池化，需要补零
            int add_0_below;
            if(j>=0 && j<P){
                add_0_above = P - j;
                add_0_below = 0; 
            } else if(j<=(num_input_buffer_fold-1) && j>(num_input_buffer_fold-1-P)){
                add_0_above = 0;
                add_0_below = j - (num_input_buffer_fold-1-P);
            } else {
                add_0_above = 0; 
                add_0_below = 0;
            }
            
            // std::cout<<"add_0_above : "<<add_0_above<<std::endl;
            // std::cout<<"add_0_below : "<<add_0_below<<std::endl;

            // 开始对片上 input buffer中的数据进行排序
            int num_tile = 0;
            int rows = 0;
            for(int k=0; k<Y_; k++){
                if(k==0){  // 第一个卷积窗口基地址计算不同，单独处理
                    if(add_0_above != 0){ // 考虑池化，上方补零
                        //std::cout<<k<<std::endl;
                        for(int c=0; c<C; c++){  // input_buffer中的数据是逐通道存储的，一个通道内的数据是连续存储的，所以要遍历每个通道
                            //std::cout<<c<<std::endl;
                            for(int num=0; num<add_0_above*S; num++){ // 补add_0_above行0
                                im2col_bank[c*R*S + num] = 0;
                            }
                        }
                        for(int c=0; c<C; c++){  // 从input_buffer中取(R-add_0_above)行数据
                            for(int r=add_0_above; r<R; r++){  // 遍历存储真实数据的行
                                for(int q=0; q<P; q++){  // 考虑池化，左侧P列0
                                    im2col_bank[c*R*S + r*S + q] = 0;
                                }
                                for(int q=P; q<S; q++){ // 取真实的数据
                                    im2col_bank[c*R*S + r*S + q] = this->input_buffer[c*R*Y + (r-add_0_above)*Y +(q-P)];
                                    this->n_cycles++;
                                }
                            }
                        }

                    } else if(add_0_below != 0){ // 下方补零
                        for(int c=0; c<C; c++){
                            for(int r=0; r<(R-add_0_below); r++){ // 从input_buffer中取数据，同时考虑左侧P列0
                                for(int q=0; q<P; q++){
                                    im2col_bank[c*R*S + r*S + q] = 0;
                                }
                                for(int q=P; q<S; q++){
                                    im2col_bank[c*R*S + r*S + q] = this->input_buffer[c*R*Y + (r+add_0_below)*Y + (q-P)];
                                    this->n_cycles++;
                                }
                            }
                        }
                        for(int c=0; c<C; c++){  // 补零
                            for(int num=0; num<add_0_below*S; num++){
                                im2col_bank[c*R*S + (R-add_0_below)*S + num] = 0;
                            }
                        }

                    } else { // 不用补零
                        for(int c=0; c<C; c++){
                            for(int r=0; r<R; r++){
                                for(int q=0; q<P; q++){  // 考虑左侧P列0
                                    im2col_bank[c*R*S + r*S + q] = 0;
                                }
                                for(int q=P; q<S; q++){
                                    im2col_bank[c*R*S + r*S + q] = this->input_buffer[c*R*Y + r*Y + (q-P)];
                                    this->n_cycles++;
                                }
                            }
                        }
                    }

                    this->input_arranged.push_back(im2col_bank);
                    rows++;
                    //std::cout<<"debug"<<std::endl;
                } else {
                    if(k >= Y_-P){  // 移位和右侧补0
                        for(int c=0; c<C; c++){
                            for(int r=0; r<R; r++){
                                for(int q=0; q<(S-1); q++){  // 移位
                                    im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                                }
                                im2col_bank[c*R*S + r*S + (S-1)] = 0;
                            }
                        }
                    } else {  // 移位和取数据
                        if(add_0_above != 0){
                            for(int c=0; c<C; c++){  // 补0
                                for(int num=0; num<add_0_above*S; num++){
                                    im2col_bank[c*R*S + num] = 0;
                                }
                            }
                            for(int c=0; c<C; c++){
                                for(int r=add_0_above; r<R; r++){
                                    for(int q=0; q<(S-1); q++){  // 移位
                                        im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                                    }
                                    // 取数据
                                    im2col_bank[c*R*S + r*S + (S-1)] = this->input_buffer[c*R*Y + (r-add_0_above)*Y + (S-P+k-1)];
                                    this->n_cycles++;
                                }
                            }

                        } else if(add_0_below !=0){
                            for(int c=0; c<C; c++){
                                for(int r=0; r<(R-add_0_below); r++){
                                    for(int q=0; q<(S-1); q++){
                                        im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                                    }
                                    // 取数据
                                    im2col_bank[c*R*S + r*S + (S-1)] = this->input_buffer[c*R*Y + (r+add_0_below)*Y + (S-P+k-1)];
                                    this->n_cycles++;
                                }
                            }
                            for(int c=0; c<C; c++){
                                for(int num=0; num<add_0_below*S; num++){
                                    im2col_bank[c*R*S + (R-add_0_below)*S + num]=0;
                                }
                            }

                        } else {
                            for(int c=0; c<C; c++){
                                for(int r=0; r<R; r++){
                                    for(int q=0; q<(S-1); q++){  // 移位
                                        im2col_bank[c*R*S + r*S + q] = im2col_bank[c*R*S + r*S + q + 1];
                                    }
                                    // 取数据
                                    im2col_bank[c*R*S + r*S + (S-1)] = this->input_buffer[c*R*Y + r*Y + (S-P+k-1)];
                                    this->n_cycles++;
                                }
                            }
                        }
                    }

                    this->input_arranged.push_back(im2col_bank);
                    rows++;

                    if(rows==this->numBanks || k == Y_-1){
                        //std::cout<<"rows : "<<rows<<std::endl;
                        num_tile++;

                        // 将this->input_arranged转化为数组
                        int total = rows * this->bankSize;
                        this->spikes = new int[total];
                        int index = 0;
                        for(const auto& row : input_arranged){
                            for(int val : row){
                                this->spikes[index++] = val;
                            }
                        }

                        // 根据需要例化 output buffer 和 neuron_state buffer
                        unsigned int num_output_buffer_need = rows * cols;
                        unsigned int num_neuron_state_buffer_need = rows * cols;
                        assert(num_output_buffer_need <= num_output);
                        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                        this->output_buffer = new int[num_output_buffer_need]();
                        this->output_buffer_cpu = new int[num_output_buffer_need]();
                        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                        this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                        // 例化脉动阵列组件，执行矩阵乘法
                        Stonne* stonne_instance = new Stonne(this->stonne_cfg);
                        matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                        stonne_instance->loadDenseGEMM(this->layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                        stonne_instance->loadGEMMTile(cols,1,rows);
                        stonne_instance->run();

                        this->n_cycles += stonne_instance->n_cycles;
                        // 对当前tile计算结果进行验证
                        for(int m = 0; m<num_output_buffer_need; m++){
                            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                            if(difference>0){
                                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                assert(false);
                            }
                        }
                        
                        //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                        // 将当前tile的输出结果和神经元中间状态分别进行处理
                        // 1. 将当前tile的神经元状态写入DRAM
                        int num_total_write_request = 0;
                        bool only = true;
                        while(true){
                            while(only){
                                for(int p=0; p<cols; p++){  // 遍历每一列

                                    //std::vector<int> packed_col_output(rows,0);
                                    std::vector<int> packed_col_neuron_state(rows,0);
                                    for(int q=0; q<rows; q++){
                                        //packed_col_output[q] = this->output_buffer[q*cols+p];
                                        packed_col_neuron_state[q] = this->neuron_state_buffer[q*cols+p];
                                    }
                                    this->n_cycles++; // 假设打包需要一个周期

                                    //int current_num_output_write_request = std::ceil(rows / (float)dram_instance->dram->GetBusBits());
                                    int current_num_neuron_state_write_request = std::ceil(rows*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                    //int current_num_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                    num_total_write_request += current_num_neuron_state_write_request;

                                    //int write_addr_output = this->output_offset + ((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows) / 8;
                                    int write_addr_neuron_state = this->neuron_state_offset + ((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                    while(current_num_neuron_state_write_request!=0){
                                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                        this->write_request_fifo->push(write_request);
                                        write_addr_neuron_state += this->addr_offset;
                                        current_num_neuron_state_write_request--;
                                        this->dram_instance->run();
                                        this->n_cycles++;
                                    }
                                    // 模拟将片上buffer中的数据写入DRAM
                                    for(int q=0; q<rows; q++){
                                        //ofmap[((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows) + q] = packed_col_output[q];
                                        nfmap[((p+i*this->stonne_cfg.m_MSNetworkCfg.ms_cols)*X_*Y_ + j*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows) + q] = packed_col_neuron_state[q];
                                    }
                                }
                                only = false;
                            }

                            if(completed_writes == num_total_write_request) {
                                completed_writes = 0; 
                                break;
                            }

                            this->dram_instance->run();
                            this->n_cycles++;
                        }

                        // 2. 将当前tile的输出结果进行累积
                        for(int p=0; p<cols; p++){
                            std::vector<int> packed_col(rows,0);
                            for(int q=0; q<rows; q++){
                                packed_col[q] = this->output_buffer[q*cols+p];
                                //std::cout<<"location in output_buffer is : "<<q*cols+p<<"   location in packed_col is : "<<q<<std::endl;
                            }
                            this->n_cycles++;  // 打包需要一个周期

                            // 将打包后的数据写入SRAM
                            for(int q=0; q<rows; q++){
                                this->on_chip_sram[p*2*Y_ + count_rows*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + q] = packed_col[q];
                                //std::cout<<"location in sram is : "<<p*2*Y_ + count_rows*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + q<<"   location in packed_col is : "<<q<<std::endl;
                            }
                            this->n_cycles++; // 将打包后的数据写入SRAM需要一个时钟周期，SRAM的总线宽度设置为脉动阵列的高度
                        }

                        // 重置数据，开始下一tile计算
                        rows = 0;
                        this->input_arranged.clear();
                        delete stonne_instance;
                        delete[] this->spikes;
                        delete[] this->output_buffer;
                        delete[] this->output_buffer_cpu;
                        delete[] this->neuron_state_buffer;
                        delete[] this->neuron_state_buffer_cpu;
                    }
                }
            }
        
            count_rows++;

            if(j>0 && j%2!=0){
                int X_pool = (j-1)/2; // 当前池化得到的行在输出特征图的位置
                //std::cout<<"X_pool : "<<X_pool<<std::endl;
                count_rows = 0;

                // 根据需要例化存储池化结果的寄存器组
                int num_output_regs_need = 2*cols*Y_/4;
                this->output_regfile_pooling = new int[num_output_regs_need]();
                this->output_regfile_pooling_cpu = new int[num_output_regs_need]();

                // 将累积的卷积输出结果进行池化
                MYPOOL* pooling_instance = new MYPOOL(stonne_cfg);
                //std::cout<<"debug2"<<std::endl;
                pooling_instance->loadPOOLLayer(Y_, cols, this->on_chip_sram, this->output_regfile_pooling);
                //std::cout<<"debug3"<<std::endl;
                pooling_instance->run();
                //std::cout<<"debug4"<<std::endl;
                this->n_cycles += pooling_instance->n_cycle;
                
                // // 验证模拟器计算的结果
                pool2x2(this->on_chip_sram, this->output_regfile_pooling_cpu, Y_, cols);

                for(int k=0; k<num_output_regs_need; k++) {
                    float difference = fabs(output_regfile_pooling[i] - output_regfile_pooling_cpu[i]);
                    if(difference>0){
                        std::cout<<"error location : "<<i<<std::endl;
                        assert(false);
                    }
                }
                //std::cout << "\033[1;32m POOLING layer Test passed correctly\033[0m" << std::endl << std::endl;

                // 将池化结果写入DRAM
                // 将池化结果看成一个二维阵列，一行是一个输出通道内的一行数据，是连续存储的
                int num_total_write_request = 0;
                bool only=true;
                while(true){
                    while(only){
                        for(int p=0; p<cols; p++){  // 遍历每个输出通道，将每个输出通道内的一行输出数据存储到DRAM中
                            int current_num_output_write_request = std::ceil((Y_/2) / (float)dram_instance->dram->GetBusBits());
                            num_total_write_request += current_num_output_write_request;
                            
                            // 往DRAM中写的基地址
                            int write_addr_output = this->output_offset + (i*this->stonne_cfg.m_MSNetworkCfg.ms_cols*X_*Y_/4 + p*X_*Y_/4 + X_pool*Y_/2) / 8;
                            while(current_num_output_write_request!=0){
                                std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                this->write_request_fifo->push(write_request);
                                write_addr_output += this->addr_offset;
                                current_num_output_write_request--;

                                this->dram_instance->run();
                                this->n_cycles++; // 一个周期发送一个内存请求事务
                            }
                            // 模拟将output_regs中的数据写入DRAM，一行有Y_/2个数据
                            for(int q=0; q<Y_/2; q++){
                                ofmap[(i*this->stonne_cfg.m_MSNetworkCfg.ms_cols*X_*Y_/4 + p*X_*Y_/4 + X_pool*Y_/2) + q] = output_regfile_pooling[p*Y_/2 + q];
                            }
                        }
                        only = false;
                    }
                    if(completed_writes == num_total_write_request) {
                        completed_writes = 0; 
                        break;
                    }
                    this->dram_instance->run();
                    this->n_cycles++;
                }
                
                delete[] this->output_regfile_pooling;
                delete[] this->output_regfile_pooling_cpu;
            }
        }
        delete[] this->on_chip_sram;
        delete[] this->input_buffer;
        delete[] this->weight_buffer;
    }

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    conv_and_pooling_compute_dataflow(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    // 检查输出
    for(int i=0; i<X_*Y_*K/4; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(ofmap[i]-ofmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    // 检查神经元状态
    for(int i=0; i<X_*Y_*K; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    // // 检查输出
    // for(int i=0; i<X_*Y_*K/4; i++){
    //     std::cout<<ofmap[i]<<" ";
    // }
    // std::cout<<std::endl;
    // for(int i=0; i<X_*Y_*K/4; i++){
    //     std::cout<<ofmap_cpu[i]<<" ";
    // }
    // std::cout<<std::endl;


    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

// 单buffer，顺序执行，模块化，加padding操作放在step1阶段实现
std::tuple<int*, int*, int*, int*> Controller::runConvandPooling_DataFlow_1(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积
    std::cout<<"Call runConvandPooling_DataFlow_1 function !!! 1"<<std::endl;
    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长

    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 加padding操作在从DRAM中取数据到input_buffer这个阶段完成，在load_input中所需要的控制信号
    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    // 从DRAM中取数据需要的一些（计数器）
    // int num_input_buffer_fold = X_;  // 遍历行的次数，即输出特征图的高度
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址
    int num_weight_obtained = 0;
    // int weight_base_addr = this->weight_offset;
    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;

        // 建模片上SRAM，用于累积卷积的输出进行池化
        int sram_size = 2 * Y_ * cols; 
        this->on_chip_sram = new int[sram_size]();

        // 根据需要实例化片上buffer (input buffer 和 weight buffer)
        int num_input_buffer_need = R * Y_padded * C;
        int num_weight_buffer_need = R * S * C * cols;
        if(num_input_buffer_need > ifmap_size){
            num_input_buffer_need = ifmap_size;  // 这种情况在padding不为0的时候可能出现
        }
        assert(num_input_buffer_need <= this->num_input);
        assert(num_weight_buffer_need <= this->num_weight);
        this->input_buffer = new int[num_input_buffer_need];
        this->weight_buffer = new int[num_weight_buffer_need];

        // 输出buffer和神经元状态buffer
        unsigned int num_output_buffer_need = Y_ * cols / 2;  // 存储池化的结果
        unsigned int num_neuron_state_buffer_need = Y_ * cols;
        assert(num_output_buffer_need <= num_output);
        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
        this->output_buffer = new int[num_output_buffer_need]();
        this->output_buffer_cpu = new int[num_output_buffer_need]();
        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();

        // 调用取权重函数，读取权重
        int num_weight_data = R * S * C * cols;
        int n_cycles_load_weight = load_weight_data(filter, this->dram_instance, num_weight_obtained, num_weight_data);
        num_weight_obtained += num_weight_data;
        this->n_cycles += n_cycles_load_weight;

        // 从DRAM中取出权重数据之后，开始取输入数据
        // 用于输入数据排序的数据结构
        this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
        this->bankSize = R * S * C;
        // std::vector<int> skip_list;
        // for(int j=0; j<num_input_buffer_fold; j++){
        //     if(j>0 && j<=P){
        //         skip_list.push_back(j);
        //     }
        //     if(j>num_input_buffer_fold-1-P && j<=num_input_buffer_fold-1){
        //         skip_list.push_back(j);
        //     }
        // }

        // std::cout<<"num_input_buffer_fold : "<<num_input_buffer_fold<<std::endl;
        int count_rows = 0; // 用于累加卷积的输出到片上SRAM计数
        // int num_retrieve_input_data = -1; // 记录取数据的次数，用于计算取数据的地址
        for(int j=0; j<X_; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;

            // 取输入数据
            int n_cycles_load_input = load_input_data_step1_onebuffer(ifmap, this->dram_instance, j, layer_parameters);
            this->n_cycles += n_cycles_load_input;

            // 重排和计算
            int n_cycles_process = process_conv_and_pooling(layer_id, i, j, cols, count_rows, layer_parameters);
            count_rows++;
            this->n_cycles += n_cycles_process;

            // 将计算结果写入DRAM
            // 1. 将神经元状态写入DRAM
            int n_cycles_write_neuron_state = store_neuron_state(nfmap, this->dram_instance, i, j, cols, layer_parameters);
            this->n_cycles += n_cycles_write_neuron_state;

            // 2. 对累积的结果进行池化，将池化结果写入DRAM
            if(j>0 && j%2!=0){
                count_rows = 0;
                // 调用池化模块
                int n_cycle_process_pooling = process_pooling(i, j, cols, layer_parameters);
                this->n_cycles += n_cycle_process_pooling;

                // 将池化结果写入DRAM
                int n_cycles_write_output = store_output(ofmap, this->dram_instance, i, j, cols, layer_parameters);
                // std::cout<<"debug"<<std::endl;
                this->n_cycles += n_cycles_write_output;
            }
        }
        delete[] this->on_chip_sram;
        delete[] this->output_buffer;
        delete[] this->output_buffer_cpu;
        delete[] this->neuron_state_buffer;
        delete[] this->input_buffer;
        delete[] this->weight_buffer;
    }

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    conv_and_pooling_compute_dataflow(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    // 检查输出
    for(int i=0; i<X_*Y_*K/4; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(ofmap[i]-ofmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    // 检查神经元状态
    for(int i=0; i<X_*Y_*K; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    // // 检查输出
    // for(int i=0; i<X_*Y_*K/4; i++){
    //     std::cout<<ofmap[i]<<" ";
    // }
    // std::cout<<std::endl;
    // for(int i=0; i<X_*Y_*K/4; i++){
    //     std::cout<<ofmap_cpu[i]<<" ";
    // }
    // std::cout<<std::endl;


    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    this->records.clear();

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

// 三个双buffer，加padding操作在step1阶段实现
std::tuple<int*, int*, int*, int*> Controller::runConvandPooling_DataFlow_2(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积

    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"                                  Call runConvandPooling_DataFlow_2 function                                    "<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;

    std::cout<<"\033[1;33m"<<"Start simulation layer : "<<"\033[0m"<<layer_id<<" "<<layer_parameters.type<<std::endl;
    this->n_conv++;
    this->layer_name = "conv"+std::to_string(this->n_conv);

    // 提取层参数
    int X = layer_parameters.X;
    int Y = layer_parameters.Y;
    int R = layer_parameters.R;
    int S = layer_parameters.S;
    int C = layer_parameters.C;
    int K = layer_parameters.K; // 输出通道数
    int P = layer_parameters.P; // 池化
    int stride = layer_parameters.stride; // 步长
    int X_padded = X + 2*P;
    int Y_padded = Y + 2*P;

    // 计算输出特征图维度
    int X_ = (X + 2*P - R)/stride + 1;
    int Y_ = (Y + 2*P - S)/stride + 1;

    // if(layer_id == 8){
    //     std::cout<<std::endl;
    //     std::cout<<"ifmap data is :"<<std::endl;
    //     for(int c=0; c<C; c++){
    //         std::cout<<"input channel is : "<<c<<std::endl;
    //         for(int x=0; x<X; x++){
    //             for(int y=0; y<Y; y++){
    //                 std::cout<<ifmap[c*X*Y + x*Y +y]<<" ";
    //             }
    //             std::cout<<std::endl;
    //         }
    //         std::cout<<std::endl;
    //     }

    //     std::cout<<std::endl;
    // }

    // 建模存储真实的数据
    int ifmap_size = X * Y * C;
    int filter_size = R * S * C * K;
    int ofmap_size = X_ * Y_ * K / 4;   
    int nfmap_size = X_ * Y_ * K;  

    if(layer_id == 1){ // 对于第一个网络层，对输出数据进行随机初始化
        ifmap = new int[ifmap_size];
        for(int i=0; i<ifmap_size; i++){
            ifmap[i] = rand()%2;
        }
    } 

    filter = new int[filter_size];
    ofmap = new int[ofmap_size]();
    nfmap = new int[nfmap_size]();
    int* ofmap_cpu = new int[ofmap_size]();
    int* nfmap_cpu = new int[nfmap_size]();

    for(int i=0; i<filter_size; i++){
        filter[i] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
    }

    // 例化DRAMsim，模拟从DRAM取数据到片上buffer的过程
    this->dram_instance = new Dram(read_callback,write_callback);
    this->dram_instance->set_read_request_fifo(this->read_request_fifo);
    this->dram_instance->set_write_request_fifo(this->write_request_fifo);

    // 加padding操作在从DRAM中取数据到input_buffer这个阶段完成，在load_input中所需要的控制信号
    for(int i=0; i<X_; i++){
        record r;
        if(i>=0 && i<P){
            r.start_rows = 0;
            r.num_rows = R-(P-i);
            r.add_0_above = P-i;
            r.add_0_below = 0;
        } else if(i>X_-1-P && i<=X_-1){
            r.start_rows = i-P;
            r.num_rows = R - (i-(X_-1-P));
            r.add_0_above = 0;
            r.add_0_below = (i-(X_-1-P));
        } else {
            r.start_rows = i-P;
            r.num_rows = R;
            r.add_0_above = 0;
            r.add_0_below = 0;
        }
        this->records.push_back(r);
    }

    // 从DRAM中取数据需要的一些（计数器）
    int num_weight_buffer_fold = std::ceil(K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 权重数据需要从DRAM中取出多少次，当输出通道数很大时，需要分时复用

    // 权重乒乓buffer
    int num_weight_buffer_need = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    assert(num_weight_buffer_need <= this->num_weight);
    this->ppbuf_weight = new PingPong_Buffer;
    this->weight_buffer_0 = new int[num_weight_buffer_need];
    this->weight_buffer_1 = new int[num_weight_buffer_need];
    PingPongBuffer_Init(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);

    int num_weight_obtained = 0;  // 下面开始从DRAM中取数据，记录已经从片外取出的数据个数，用于下次取数据的基地址

    // 第一次从DRAM中取权重数据
    int num_weight_data; // 要从DRAM中取的权重个数
    if(num_weight_buffer_fold>1){
        num_weight_data = R * S * C * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    } else {
        num_weight_data = R * S * C * K;
    }

    int n_cycles_load_first_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
    num_weight_obtained += num_weight_data;
    this->n_cycles += n_cycles_load_first_weight;
    PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1);
    //std::cout<<"load the first weights cycles : "<<n_cycles_load_first_weight<<std::endl;
    
    for(int i=0; i<num_weight_buffer_fold; i++){  // 取权重循环，每次循环都要取全部的输入数据
        std::cout<<"=============================================== weight loop ===================================================== "<<i<<"/"<<num_weight_buffer_fold-1<<std::endl;

        // 权重数据是一个输出通道一个输出通道进行读取的
        int start_col = i*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
        int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
        int cols = end_col - start_col;

        // 建模片上SRAM，用于累积卷积的输出进行池化
        int sram_size = 2 * Y_ * cols; 
        this->on_chip_sram = new int[sram_size]();

        // 根据需要实例化片上buffer (input buffer 和 weight buffer)
        int num_input_buffer_need = R * Y_padded * C;
        // if(num_input_buffer_need > ifmap_size){
        //     num_input_buffer_need = ifmap_size;  // 这种情况在padding不为0的时候可能出现
        // }
        assert(num_input_buffer_need <= this->num_input);

        // 输入乒乓buffer
        this->ppbuf_input = new PingPong_Buffer;
        this->input_buffer_0 = new int[num_input_buffer_need];
        this->input_buffer_1 = new int[num_input_buffer_need];
        PingPongBuffer_Init(this->ppbuf_input, this->input_buffer_0, this->input_arranged_buffer_1);

        // 输出buffer和神经元状态buffer
        unsigned int num_output_buffer_need = Y_ * cols / 2;  // 存储池化的结果
        unsigned int num_neuron_state_buffer_need = Y_ * cols;
        assert(num_output_buffer_need <= num_output);
        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
        this->output_buffer = new int[num_output_buffer_need]();
        this->output_buffer_cpu = new int[num_output_buffer_need]();
        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();

        // 从DRAM中取出权重数据之后，开始取输入数据
        // 用于输入数据排序的数据结构
        this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows;
        this->bankSize = R * S * C;

        // 第一次加载输入数据到片上buffer
        int n_cycles_load_first_input = load_input_data_step1_ppbuffer(layer_id, ifmap, this->dram_instance, 0, layer_parameters);
        this->n_cycles += n_cycles_load_first_input;
        //std::cout<<"load the first input cycles : "<<n_cycles_load_first_input<<std::endl;
        PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);

        // std::cout<<"below is the first load input : "<<std::endl;
        // for(int c=0; c<C; c++){
        //     std::cout<<"channel is : "<<c<<std::endl;
        //     for(int p=0; p<R; p++){
        //         for(int q=0; q<Y_padded; q++){
        //             std::cout<<this->ppbuf_input->current_buffer[c*R*Y_padded + p*Y_padded + q]<<" ";
        //         }
        //         std::cout<<std::endl;
        //     }
        //     std::cout<<std::endl;
        // }

        //std::cout<<"current weight loop begin cycles : "<<this->n_cycles<<std::endl;
        int count_rows = 0; // 用于累加卷积的输出到片上SRAM计数
        for(int j=0; j<X_; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;

            // *** 加载下一块输入数据，与上一块数据的计算并行
            int n_cycles_load_next_input;
            if(j+1<X_){
                n_cycles_load_next_input = load_input_data_step1_ppbuffer(layer_id, ifmap, this->dram_instance, j+1, layer_parameters);
                //std::cout<<"load next input cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                //std::cout<<"load next input cycles : "<<n_cycles_load_next_input<<std::endl;
            }

            // *** 加载下一块权重数据，与当前权重循环的计算过程并行
            int n_cycles_load_next_weight;
            if(i+1<num_weight_buffer_fold && j==0){
                int start_col_next = (i+1)*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col_next = std::min<int>(start_col_next+this->stonne_cfg.m_MSNetworkCfg.ms_cols, K);
                int cols_next = end_col_next - start_col_next;
                num_weight_data = R * S * C * cols_next;  // 要从DRAM中取出的数据个数
                n_cycles_load_next_weight = load_weight_data_ppbuffer(filter, this->dram_instance, num_weight_obtained, num_weight_data);
                num_weight_obtained += num_weight_data;
                //std::cout<<"load next weight cycles : "<<n_cycles_load_next_weight<<std::endl;
            } else {
                n_cycles_load_next_weight = 0;
                //std::cout<<"load next weight cycles : "<<n_cycles_load_next_weight<<std::endl;
            }

            // *** 重排和计算
            int n_cycles_process = process_conv_and_pooling(layer_id, i, j, cols, count_rows, layer_parameters);
            count_rows++;
            //std::cout<<"process_conv_and_pooling cycles : "<<n_cycles_process<<std::endl;
            //this->n_cycles += n_cycles_process;

            // *** 将计算结果写入DRAM
            // 1. 将神经元状态写入DRAM
            int n_cycles_write_neuron_state = store_neuron_state(nfmap, this->dram_instance, i, j, cols, layer_parameters);
            //std::cout<<"write neuron_state cycles : "<<n_cycles_write_neuron_state<<std::endl;
            // this->n_cycles += n_cycles_write_neuron_state;

            // 2. 对累积的结果进行池化，将池化结果写入DRAM
            int n_cycles_pooling_and_write_output;
            int n_cycles_process_pooling;
            int n_cycles_write_output;
            if(j>0 && j%2!=0){
                count_rows = 0;
                // 调用池化模块
                n_cycles_process_pooling = process_pooling(i, j, cols, layer_parameters);
                //std::cout<<"process_pooling cycles : "<<n_cycles_process_pooling<<std::endl;
                //this->n_cycles += n_cycle_process_pooling;

                // 将池化结果写入DRAM
                n_cycles_write_output = store_output(ofmap, this->dram_instance, i, j, cols, layer_parameters);
                //std::cout<<"write output cycles : "<<n_cycles_write_output<<std::endl;
                // std::cout<<"debug"<<std::endl;
                //this->n_cycles += n_cycles_write_output;

                n_cycles_pooling_and_write_output = n_cycles_process_pooling + n_cycles_write_output;
                //std::cout<<"process_pooling and write output cycles : "<<n_cycles_pooling_and_write_output<<std::endl;
            } else {
                n_cycles_pooling_and_write_output = 0;
                //std::cout<<"process_pooling and write output cycles : "<<n_cycles_pooling_and_write_output<<std::endl;
            }

            if(i==num_weight_buffer_fold-1 && j==X_-1){
                // 最后一次计算结束，加上输出写入DRAM所需周期
                this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process+n_cycles_write_neuron_state+n_cycles_pooling_and_write_output);
                std::cout<<std::endl;
                //std::cout<<"Global cycles : "<<this->n_cycles<<std::endl;
            } else {
                this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process+n_cycles_process_pooling);
                std::cout<<std::endl;
                //std::cout<<"Global cycles : "<<this->n_cycles<<std::endl;
            }

            // 切换输入缓冲区
            PingPongBuffer_Switch(this->ppbuf_input, this->input_buffer_0, this->input_buffer_1);
            //std::cout<<"debug"<<std::endl;
        }

        // 切换权重缓冲区
        PingPongBuffer_Switch(this->ppbuf_weight, this->weight_buffer_0, this->weight_buffer_1); 

        delete[] this->on_chip_sram;
        delete[] this->output_buffer;
        delete[] this->output_buffer_cpu;
        delete[] this->neuron_state_buffer;
        delete[] this->input_buffer_0;
        delete[] this->input_buffer_1;
        delete this->ppbuf_input;
    }
    delete[] this->weight_buffer_0;
    delete[] this->weight_buffer_1;
    delete this->ppbuf_weight;

    // std::cout<<std::endl;
    // std::cout<<"The conv compute output result is : "<<std::endl;
    // for(int k=0; k<K; k++){
    //     std::cout<<"output channel : "<<k<<std::endl;
    //     for(int p=0; p<X_/2; p++){
    //         for(int q=0; q<Y_/2; q++){
    //             std::cout<<ofmap[k*X_*Y_/4 + p*Y_/2 +q]<<" ";
    //         }
    //         std::cout<<std::endl;
    //     }
    //     std::cout<<std::endl;
    // }

    // 所有tile计算完毕，验证写到DRAM中的结果是否正确
    conv_and_pooling_compute_dataflow(R, S, C, K, P, stride, X, Y, ifmap, filter, ofmap_cpu, nfmap_cpu, this->stonne_cfg.V_th);

    // 检查输出
    for(int i=0; i<X_*Y_*K/4; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(ofmap[i]-ofmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value ofmap simlator : "<<ofmap[i]<<"  value ofmap cpu : "<<ofmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    // 检查神经元状态
    for(int i=0; i<X_*Y_*K; i++){
        //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
        float difference = fabs(nfmap[i]-nfmap_cpu[i]);
        if(difference>0){
            std::cout<<"ERROR position : "<<i<<"  value nfmap simlator : "<<nfmap[i]<<"  value nfmap cpu : "<<nfmap_cpu[i]<<std::endl;
            assert(false);
        }
    }

    std::cout<<"current cycles : "<<this->n_cycles<<std::endl;

    std::cout<<"\033[1;32m"<<"OVER Test passed correctly"<<"\033[0m"<<std::endl<<std::endl;

    this->records.clear();

    delete[] ofmap_cpu;
    delete[] nfmap_cpu;

    return {ifmap, filter, ofmap, nfmap};
}

void Controller::run(){
    this->traverse(); // 遍历每一层
}

void Controller::traverse(){
    // 遍历每一层
    for(int i=0;i<this->layers.size();i++){
        if(layers[i].type == "conv"){
            // 卷积层
            this->stonne_cfg.layer_type = CONV; 
            this->n_conv++;
            this->layer_name = "conv"+std::to_string(this->n_conv);

            // 先计算出单卷积层的输出特征图size
            unsigned int X_ = (this->layers[i].X + 2*this->layers[i].P - this->layers[i].R)/this->layers[i].stride + 1;
            unsigned int Y_ = (this->layers[i].Y + 2*this->layers[i].P - this->layers[i].S)/this->layers[i].stride + 1;
            // unsigned int X_without_padding = (this->layers[i].X - this->layers[i].R)/this->layers[i].stride + 1;
            // unsigned int Y_without_padding = (this->layers[i].Y - this->layers[i].S)/this->layers[i].stride + 1;

            // 建模存储真实的数据
            unsigned int ifmap_size = this->layers[i].X * this->layers[i].Y * this->layers[i].C;
            unsigned int filter_size = this->layers[i].R * this->layers[i].S * this->layers[i].C * this->layers[i].K;
            unsigned int ofmap_size = X_ * Y_ * this->layers[i].K; 
            unsigned int nfmap_size = X_ * Y_ * this->layers[i].K;

            this->ifmap = new int[ifmap_size];
            this->filter = new int[filter_size];
            this->ofmap = new int[ofmap_size]();
            this->ofmap_cpu = new int[ofmap_size]();
            this->nfmap = new int[nfmap_size]();
            this->nfmap_cpu = new int[nfmap_size]();

            for(int n=0; n<ifmap_size; n++){
                this->ifmap[n] = rand()%2;
            }

            for(int n=0; n<filter_size; n++){
                this->filter[n] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
            }
            
            // std::cout<<"Below are the data of each channel of the input feature map : "<<std::endl;
            // for(int num=0; num<ifmap_size; num++){
            //     std::cout<<this->ifmap[num]<<"  ";
            // }
            // std::cout<<std::endl;           
            // for(int c=0; c<this->layers[i].C; c++){  // 遍历每个通道，输出每个通道的数据
            //     std::cout<<"channel : "<<c<<std::endl;
            //     for(int m=0; m<this->layers[i].X; m++){
            //         for(int n=0; n<this->layers[i].Y; n++){
            //             int index = c + this->layers[i].C * (m*this->layers[i].Y+n);
            //             std::cout<<this->ifmap[index]<<"    ";
            //         }
            //         std::cout<<std::endl;
            //     }
            // }

            // std::cout<<"Below is the weight data, each row is a convolution kernel : "<<std::endl;
            // for(int k=0; k<this->layers[i].K; k++){
            //     for(int num=0; num<this->layers[i].R*this->layers[i].S*this->layers[i].C; num++){
            //         std::cout<<this->filter[k*this->layers[i].R*this->layers[i].S*this->layers[i].C + num]<<" ";
            //     }
            //     std::cout<<std::endl;
            // }

            this->numBanks = this->stonne_cfg.m_MSNetworkCfg.ms_rows; // bank个数为脉动阵列行数
            this->bankSize = this->layers[i].R * this->layers[i].S * this->layers[i].C;  
            //int total = this->numBanks * this->bankSize;
            //this->spikes = new int[total];  // 用于存储经过 im2col单元排序好的脉冲数据

            // int input_arranegd_fold = std::ceil(Y_ / (float)this->stonne_cfg.m_MSNetworkCfg.ms_rows); // input buffer 中的数据需要排列成rows行的矩阵多少次
            // int remainder = Y_ % this->stonne_cfg.m_MSNetworkCfg.ms_rows;
            //std::cout<<"input_arranegd_fold : "<<input_arranegd_fold<<std::endl;

            // 例化stonne
            // Stonne* stonne_instance = new Stonne(stonne_cfg);

            // 例化dram
            this->dram_instance = new Dram(read_callback, write_callback);
            this->dram_instance->set_read_request_fifo(this->read_request_fifo);
            this->dram_instance->set_write_request_fifo(this->write_request_fifo);

            // 开始仿真前，先从DRAM中读取数据到片上buffer中
            // 权重数据和脉冲数据各自分成几块
            int num_input_buffer_fold = X_; // 从DRAM中取多少次数据
            if(this->layers[i].X <= this->layers[i].R){
                num_input_buffer_fold =1;
            }
            int num_weight_buffer_fold = std::ceil(this->layers[i].K / (float)this->stonne_cfg.m_MSNetworkCfg.ms_cols);
            // std::cout<<"this->layers[i].K : "<<this->layers[i].K<<std::endl;
            // std::cout<<"this->stonne_cfg.m_MSNetworkCfg.ms_cols : "<<this->stonne_cfg.m_MSNetworkCfg.ms_cols<<std::endl;
            // std::cout<<"num_input_buffer_fold : "<<num_input_buffer_fold<<std::endl;
            // std::cout<<"num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
            // 在DRAM中依次存放：输入、权重、输出、神经元状态
            uint64_t input_offset = 0x0000;  // 输入数据在DRAM中的起始地址
            uint64_t weight_offset = this->input_dram_size*1024*1024; // 权重数据在DRAM中的起始地址
            uint64_t weight_offset_copy = weight_offset;
            uint64_t output_offset = (this->input_dram_size + this->weight_dram_size)*1024*1024;  // 输出数据在DRAM中的起始地址
            uint64_t neuron_state_offset = (this->input_dram_size + this->weight_dram_size + this->output_dram_size)*1024*1024; // 神经元状态在DRAM中的起始地址
            uint64_t addr_offset = 8;  // 连续读取时，每次地址加8，因为一次可以读出64bit数据

            // 输出开始仿真信息
            std::cout<<"\033[1m\033[33m Start simulation layer :  \033[0m"<<this->layer_name<<std::endl;
            // std::cout<<"num_weight_buffer_fold = "<<num_weight_buffer_fold<<std::endl;
            // std::cout<<"num_input_buffer_fold = "<<num_input_buffer_fold<<std::endl;

            // 记录已经从片外取出的数据个数，用于下次取数据的基地址，因为上一次取出的数据不一定用完
            int num_input_obtained = 0;     
            int num_weight_obtained = 0; 
            //int flag=0;
            //int flag_in=0;
            // 层内主循环=====================================================================================================
            for(int j=0;j<num_weight_buffer_fold;j++){
                //std::cout<<"==================================================== weight loop ========================================================== : "<<j<<std::endl;
                input_offset = 0x0000; 
                num_input_obtained = 0; 

                // 对于权重数据的读取是一列一列读的
                int start_col = j*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                int end_col = std::min<int>(start_col+this->stonne_cfg.m_MSNetworkCfg.ms_cols, this->layers[i].K);
                int cols = end_col - start_col;
                //int delta_weight = this->stonne_cfg.m_MSNetworkCfg.ms_cols - cols;

                // 片上SRAM
                unsigned int sram_size = this->layers[i].pooling_size * Y_ * cols;
                this->on_chip_sram = new int[sram_size]();

                // 根据需要实例化片上buffer，输入buffer和权重buffer
                unsigned int num_input_buffer_need = this->layers[i].R * this->layers[i].Y * this->layers[i].C;  
                if(num_input_buffer_need>ifmap_size){
                    num_input_buffer_need = ifmap_size; 
                }
                unsigned int num_weight_buffer_need = this->layers[i].R * this->layers[i].S * this->layers[i].C * cols;
                //unsigned int num_neuron_state_buffer_need = this->stonne_cfg.m_MSNetworkCfg.ms_rows * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                assert(num_input_buffer_need <= num_input);
                assert(num_weight_buffer_need <= num_weight);
                //assert(num_output_buffer_need <= num_output);
                //assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                this->input_buffer = new int[num_input_buffer_need];
                this->weight_buffer = new int[num_weight_buffer_need];
                //this->output_buffer = new int[num_output_buffer_need]();
                //this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                //this->output_buffer_cpu = new int[num_output_buffer_need]();
                //this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                // 从DRAM中读取权重数据
                int num_weight_data = this->layers[i].R * this->layers[i].S * this->layers[i].C * cols;
                int num_weight_read_request = std::ceil(num_weight_data*this->weight_width / (float)dram_instance->dram->GetBusBits());
                int num_weight_read_request_copy = num_weight_read_request;

                // std::cout<<"num_weight_read_request : "<<num_weight_read_request<<std::endl;
                while(true){
                    
                    if(num_weight_read_request !=0 ){
                        std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(weight_offset,false);
                        this->read_request_fifo->push(read_request);
                        weight_offset += addr_offset;
                        num_weight_read_request--;
                    }

                    if(completed_reads == num_weight_read_request_copy){
                        completed_reads = 0;
                        //std::cout<<"All  weight read requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;

                        // 模拟将片外数据写入片上buffer
                        for(int k=0; k<num_weight_data; k++){
                            this->weight_buffer[k] = this->filter[num_weight_obtained+k];
                        }

                        num_weight_obtained += num_weight_data;
                        weight_offset = weight_offset_copy + ((num_weight_obtained*this->weight_width)/8);

                        break;
                    }
                    this->dram_instance->run();  // 一个周期驱动一次dram
                    this->n_cycles++;  // 一个周期发送一个内存请求事务
                }

                // 权重从DRAM中取出一次供ifmap所有数据复用
                int count_rows = 0; 
                for(int k=0;k<num_input_buffer_fold;k++){  // ifmap要从DRAM中取出多少次（每次取出一部分）

                    // 第一次从片外取数据时，需要取R行数据
                    // 后面从片外取数据时，需要取stride行，（R-stride）行在片上buffer内移动

                    //std::cout<<"================================ input loop =========================================================== : "<<k<<std::endl;
                    int num_input_data; // 从DRAM中读取的输入数据个数 
                    int num_input_read_request; // 需要往DRAM发送多少次读请求用于读取输入数据
                    int num_input_read_request_copy;
                    
                    // 第一次取input数据的时候取R行，后续只需要取stride行非重复的数据即可
                    if(k==0){   
                        num_input_data = num_input_buffer_need;
                    } else {
                        num_input_data = this->layers[i].Y * this->layers[i].C * this->layers[i].stride; // 后续只需往下读取stride行数据
                    }
                    num_input_read_request = std::ceil(num_input_data / (float)dram_instance->dram->GetBusBits());
                    num_input_read_request_copy = num_input_read_request;

                    // std::cout<<"num_input_read_request : "<<num_input_read_request<<std::endl;

                    while(true){  // 发送DRAM内存事务请求

                        if(num_input_read_request!=0){ // 读输入请求
                            std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(input_offset,false);  // （地址，事件类型）false表示读请求，true表示写请求
                            this->read_request_fifo->push(read_request); // 将请求推入fifo
                            input_offset += addr_offset;  // 下一个读取地址
                            num_input_read_request--;
                        }

                        // completed_reads == num_total_read_request_copy 说明从外存取数据完毕
                        if(completed_reads == num_input_read_request_copy){ // 外存中输入数据存放的顺序：输入通道优先；权重数据：输入通道、行列、输出通道
                            // 请求全部响应之后，将片外的数据存入片上buffer
                            // 对于输入数据，第一次取时取R行，后续只取一行数据，要将重复的（R-1）行数据进行移动
                            if(k==0){
                                for(int q=0; q<num_input_data; q++){
                                    this->input_buffer[q] = this->ifmap[q];
                                }
                            }else{
                                // 移位
                                for(int q=0; q<this->layers[i].Y * this->layers[i].C * (this->layers[i].R-this->layers[i].stride); q++){ // （R-stride）行在片上buffer内移动
                                    this->input_buffer[q] = this->input_buffer[q + this->layers[i].Y * this->layers[i].C * this->layers[i].stride];
                                }
                                // 取数据
                                for(int q=0; q<this->layers[i].Y * this->layers[i].C * this->layers[i].stride; q++){
                                    this->input_buffer[q+this->layers[i].Y * this->layers[i].C * (this->layers[i].R-this->layers[i].stride)] = this->ifmap[num_input_obtained+q];
                                }
                            }

                            num_input_obtained += num_input_data;

                            input_offset = num_input_obtained / 8;
                            
                            completed_reads = 0;
                            //std::cout<<"All input read requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                            break;
                        }

                        this->dram_instance->run();  // 一个周期驱动一次dram

                        this->n_cycles++;  // 一个周期发送一个内存请求事务
                    }

                    // std::cout<<"Below is the data in the current input buffer : "<<std::endl;
                    // for(int c=0; c<this->layers[i].C; c++){  // 遍历每个通道，输出每个通道的数据
                    //     std::cout<<"channel : "<<c<<std::endl;
                    //     for(int m=0; m<this->layers[i].R; m++){
                    //         for(int n=0; n<this->layers[i].Y; n++){
                    //             int index = c + this->layers[i].C * (m*this->layers[i].Y+n);
                    //             std::cout<<this->input_buffer[index]<<"    ";
                    //         }
                    //         std::cout<<std::endl;
                    //     }
                    // }

                    // 片外数据读取到片上buffer之后，对输入数据进行重排
                    // 片上buffer所容纳的数据脉动阵列一次不一定能计算完毕，所以需要分块

                    // 从片上buffer中取出数据存放在一个vector（转换单元）中，填满之后，将其添加到this->input_arranged中
                    // 一个周期取出一个数据，生成第一个卷积核数据需要filter_size个周期，后续卷积核每个需要R*C个周期

                    std::vector<int> im2col_bank(this->bankSize,0); // 用于重排序的单元，初始化其大小，为了后面能够对其进行索引
                    
                    if(this->layers[i].P == 0) { // 没有池化，只需考虑步长
                        int num_tile = 0;
                        int rows = 0; // 有多少个卷积核就排列成多少行数据，送入脉动阵列计算
                        for(int n=0; n<Y_; n++){  // buffer中的数据有多少个卷积窗口
                            // 当排序好一定数量卷积核后，调用计算核心进行计算
                            if(n==0){ // 排序第一个卷积窗口时，需要filter_size个周期
                                for(int p=0; p<this->layers[i].R; p++){
                                    int base_addr = p * this->layers[i].C * this->layers[i].Y;
                                    //std::cout<<"base_addr : "<<base_addr<<std::endl;
                                    for(int q=0; q<this->layers[i].C*this->layers[i].S; q++){
                                        im2col_bank[p * this->layers[i].C * this->layers[i].S + q] = this->input_buffer[base_addr + q];
                                        this->n_cycles++;
                                    }
                                }

                                // std::cout<<"The sorted data is : "<<std::endl;
                                // for(int p=0; p<this->bankSize; p++){
                                //     std::cout<<im2col_bank[p]<<"   ";
                                // }
                                // std::cout<<std::endl;

                                this->input_arranged.push_back(im2col_bank);
                                rows++;
                            } else {  // 后续排列卷积核需要 R*C*stride个周期
                                for(int st=0; st<this->layers[i].stride; st++){
                                    for(int p=0; p<this->layers[i].R; p++){
                                        int base_addr = (p*this->layers[i].Y + this->layers[i].S + (n-1)*this->layers[i].stride +st) * this->layers[i].C;
                                        //std::cout<<"base_addr : "<<base_addr<<std::endl;
                                        for(int q=0; q<this->layers[i].C; q++){
                                            if(q==0 && st==0){   // 移位（S-stride）*C 个数据，取stride*C个数据
                                                //std::cout<<"Shifting process : "<<std::endl;
                                                for(int num=0; num<(this->layers[i].S-this->layers[i].stride)*this->layers[i].C; num++){
                                                    //std::cout<<"location : "<<p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C*this->layers[i].stride<<"   -->  location : "<<p * this->layers[i].C * this->layers[i].S + num<<std::endl;
                                                    im2col_bank[p * this->layers[i].C * this->layers[i].S + num] = im2col_bank[p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C*this->layers[i].stride];
                                                }
                                                //取数据
                                                //std::cout<<"Data fetching process(in input_buffer location) : "<<base_addr + q<<"    to   im2col_bank location : "<<(this->layers[i].S-this->layers[i].stride+st) * this->layers[i].C + p * this->layers[i].C * this->layers[i].S + q<<std::endl;
                                                im2col_bank[(this->layers[i].S-this->layers[i].stride+st) * this->layers[i].C + p * this->layers[i].C * this->layers[i].S + q] = this->input_buffer[base_addr + q];
                                                this->n_cycles++;
                                                
                                            } else{  // 取stride个数据
                                                //std::cout<<"Data fetching process(in input_buffer location) : "<<base_addr + q<<"    to   im2col_bank location : "<<(this->layers[i].S-this->layers[i].stride+st) * this->layers[i].C + p * this->layers[i].C * this->layers[i].S + q<<std::endl;
                                                im2col_bank[(this->layers[i].S-this->layers[i].stride+st) * this->layers[i].C + p * this->layers[i].C * this->layers[i].S + q] = this->input_buffer[base_addr + q];
                                                this->n_cycles++;
                                            }
                                        }
                                    }
                                }

                                // std::cout<<"The sorted data is : "<<std::endl;
                                // for(int p=0; p<this->bankSize; p++){
                                //     std::cout<<im2col_bank[p]<<"   ";
                                // }
                                // std::cout<<std::endl;

                                this->input_arranged.push_back(im2col_bank);
                                rows++;

                                if((rows == this->numBanks) || n==Y_-1 ){  // 如果可以用于计算了 或者排序排完了
                                    // std::cout<<"current tile is : ============================================================ "<<num_tile<<std::endl;
                                    // std::cout<<"current tile rows is :     "<<rows<<std::endl;
                                    num_tile+=1;
                                    // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                    int total = rows * this->bankSize;
                                    this->spikes = new int[total];
                                    int index = 0;
                                    for(const auto& row : input_arranged){
                                        for(int val : row){
                                            this->spikes[index++] = val;
                                        }
                                    }

                                    // 根据需要例化存储输出结果的buffer
                                    unsigned int num_output_buffer_need = rows * cols;
                                    unsigned int num_neuron_state_buffer_need = rows * cols;
                                    assert(num_output_buffer_need <= num_output);
                                    assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                    this->output_buffer = new int[num_output_buffer_need]();
                                    this->output_buffer_cpu = new int[num_output_buffer_need]();
                                    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                    this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                    Stonne* stonne_instance = new Stonne(stonne_cfg);
                                    matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                    //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                    stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                    stonne_instance->loadGEMMTile(cols,1,rows);
                                    stonne_instance->run();

                                    this->n_cycles += stonne_instance->n_cycles;
                                    // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                    // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                    for(int m = 0; m<num_output_buffer_need; m++){
                                        float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                        if(difference>0){
                                            std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                            std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                            std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                            // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                            // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                            assert(false);
                                        }
                                    }
                                    //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                    // 将输出buffer中的数据存储在片上SRAM

                                    // 输出buffer可以看作是一个二维阵列，其中一列数据是一个通道内的数据，要将这列数据打包，存放在SRAM中
                                    std::vector<int>  packed(rows,0);
                                    for(int p=0; p<cols; p++){  // 遍历每个输出通道
                                        for(int q=0; q<rows; q++){
                                            packed[q] = this->output_buffer[q*cols+p];
                                        }
                                        this->n_cycles++;  // 打包这个过程需要一个周期

                                        // 将打包后的数据写入SRAM
                                        for(int q=0; q<rows; q++){
                                            // p*this->layers[i].pooling_size*Y_ 是当前通道的基地址
                                            // count_rows*Y_ 是当前输出行的基地址
                                            // (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows 是当前输出行已经存储的数据个数 
                                            this->on_chip_sram[p*this->layers[i].pooling_size*Y_ + count_rows*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + q] = packed[q];
                                        }
                                        this->n_cycles++;  // 将打包后的数据写入SRAM需要一个时钟周期，SRAM的总线宽度设置为脉动阵列的高度
                                        
                                    }

                                    // // 输出输入数据计算结果和输出数据
                                    // std::cout<<"input data ---------------------: "<<std::endl;
                                    // for(int m=0;m<rows;m++){
                                    //     for(int a=0;a<this->layers[i].R * this->layers[i].S * this->layers[i].C;a++){
                                    //         std::cout<<this->spikes[m*this->layers[i].R * this->layers[i].S * this->layers[i].C + a]<<"   ";
                                    //     }
                                    //     std::cout<<std::endl;
                                    // }
                                    // std::cout<<"weight data -------------------:"<<std::endl;
                                    // for(int m=0;m<cols;m++){
                                    //     for(int a=0;a<this->layers[i].R * this->layers[i].S * this->layers[i].C;a++){
                                    //         std::cout<<this->weight_buffer[m*this->layers[i].R * this->layers[i].S * this->layers[i].C + a]<<"   ";
                                    //     }
                                    //     std::cout<<std::endl;
                                    // }

                                    // std::cout<<"output_sim data -----------------:"<<std::endl;
                                    // for(int m=0;m<num_output_buffer_need;m++){
                                    //     std::cout<<this->output_buffer[m]<<"    ";
                                    // }
                                    // std::cout<<std::endl;

                                    // std::cout<<"output_cpu data ------------------:"<<std::endl;
                                    // for(int m=0;m<num_output_buffer_need;m++){
                                    //     std::cout<<this->output_buffer_cpu[m]<<"    ";
                                    // }
                                    // std::cout<<std::endl;

                                    // std::cout<<"neuron_state_sim data--------------:"<<std::endl;
                                    // for(int m=0;m<num_neuron_state_buffer_need;m++){
                                    //     std::cout<<this->neuron_state_buffer[m]<<"  ";
                                    // }
                                    // std::cout<<std::endl;
                                    // std::cout<<"neuron_state_cpu data-----------------:"<<std::endl;
                                    // for(int m=0;m<num_neuron_state_buffer_need;m++){
                                    //     std::cout<<this->neuron_state_buffer_cpu[m]<<"  ";
                                    // }
                                    // std::cout<<std::endl;


                                    // 将中间状态写入DRAM
                                    //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                    // int num_total_write_request = 0;
                                    // bool only = true;
                                    // while(true){
                                    //     while(only){
                                    //         for(int m=0; m<rows; m++){
                                    //             int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                    //             int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                    //             int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                    //             num_total_write_request += current_total_write_request;
                                    //             // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                    //             // assert(false);
                                    //             // 该行的基地址
                                    //             int write_addr_output = output_offset + ((k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                    //             int write_addr_neuron_state = neuron_state_offset + ((k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                    //             while(current_total_write_request!=0){
                                    //                 if(current_num_neuron_state_write_request!=0){
                                    //                     std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                    //                     this->write_request_fifo->push(write_request);
                                    //                     write_addr_neuron_state += addr_offset;
                                    //                     current_num_neuron_state_write_request--;
            
                                    //                 } else if(current_num_output_write_request!=0){
                                    //                     std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                    //                     this->write_request_fifo->push(write_request);
                                    //                     write_addr_output += addr_offset;
                                    //                     current_num_output_write_request--;
                                    //                 }
            
                                    //                 this->dram_instance->run();
                                    //                 this->n_cycles++; // 一个周期发送一个内存请求事务
                                    //                 current_total_write_request--;
                                    //             }
                                    //             // 模拟将片上buffer数据写入DRAM
                                    //             // 将当前行中的每一列数据写入DRAM
                                    //             for(int p=0; p<cols; p++){
                                    //                 this->ofmap[(k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                    //                 this->nfmap[(k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                    //             }
                                    //         }
                                    //         only = false;
                                    //     }
            
                                    //     // std::cout<<"num : "<<num<<std::endl;
                                    //     // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
            
                                    //     // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
            
                                    //     if(completed_writes == num_total_write_request){
                                    //         completed_writes = 0;
                                    //         //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                    //         break;
                                    //     }
            
                                    //     this->dram_instance->run();
                                    //     this->n_cycles++;
                                    // }

                                    // 重置数据，开始下一tile计算
                                    rows=0;
                                    input_arranged.clear();
                                    delete stonne_instance;
                                    delete[] this->spikes;
                                    delete[] this->output_buffer;
                                    delete[] this->output_buffer_cpu;
                                    delete[] this->neuron_state_buffer;
                                    delete[] this->neuron_state_buffer_cpu;

                                }
                            }
                        }

                        count_rows++;  // 当达到 pooling_size 行后，说明生成的卷积数据可以进行池化了

                        if(count_rows == this->layers[i].pooling_size){  // 将卷积结果进行池化
                            count_rows = 0;
                            
                        }


                    } else {  // 加padding，只支持padding为1或者2，且默认加padding时stride都为1
                        //std::cout<<"********************* The pooling step size is not 0 *******************************"<<std::endl;
                        // 第一次从片外DRAM取进来的数据可以用reused_times次
                        // 倒数第reused_times次从片外DRAM取进来的数据可以用reused_times次
                        int reused_times = this->layers[i].P + 2 - this->layers[i].stride;
                        assert(reused_times>1); // 目前只支持padding>=stride的卷积层

                        int num_tile = 0;
                        int rows = 0; // 当排列好的行数够用来计算或者排完了，就送到脉动阵列进行计算

                        if(k==0 && num_input_buffer_fold==1){
                            // 在VGG网络模型中，当输入尺寸为2*2时，特殊考虑
                            // 片外的数据只读取到片上一次，使用两次，第一次在顶层补一行零，第二次在底部补一行零

                            // 第一次，在顶层补一行零
                            int num_tile = 0;

                            for(int n=0; n<Y_; n++){
                                if(n==0){ // 第一块卷积窗口，左侧有一列0
                                    for(int q=0; q<this->layers[i].C*this->layers[i].S; q++){  // 第一行数据为0
                                        im2col_bank[q] = 0;
                                    }
                                    for(int p=1; p<this->layers[i].R; p++){  // 剩下的行取真实的值
                                        int base_addr = (p-1) * this->layers[i].C *this->layers[i].Y;
                                        for(int q=0; q<this->layers[i].C*this->layers[i].P; q++){   // 这些列是0
                                            im2col_bank[p*this->layers[i].C*this->layers[i].S + q] = 0;
                                        }
                                        for(int q=this->layers[i].C*this->layers[i].P; q<this->layers[i].C*this->layers[i].S; q++){
                                            im2col_bank[p * this->layers[i].C * this->layers[i].S + q] = this->input_buffer[base_addr+(q-this->layers[i].C*this->layers[i].P)];
                                            this->n_cycles++;
                                        }
                                    }

                                    this->input_arranged.push_back(im2col_bank);
                                    rows++;

                                } else if(n==Y_-1){  // 最后一个卷积窗口，右侧有一列0
                                    for(int p=0; p<this->layers[i].R; p++){
                                        for(int q=0; q<(this->layers[i].S-1)*this->layers[i].C; q++){
                                            im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = im2col_bank[p*this->layers[i].S*this->layers[i].C + q + this->layers[i].C]; 
                                        }
                                        for(int q=(this->layers[i].S-1)*this->layers[i].C; q<this->layers[i].S*this->layers[i].C; q++){
                                            im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = 0;
                                        }
                                    }
                                    
                                    this->input_arranged.push_back(im2col_bank);
                                    rows++;

                                    num_tile+=1;

                                    // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                    int total = rows * this->bankSize;
                                    this->spikes = new int[total];
                                    int index = 0;
                                    for(const auto& row : input_arranged){
                                        for(int val : row){
                                            this->spikes[index++] = val;
                                        }
                                    }

                                    // 根据需要例化存储输出结果的buffer
                                    unsigned int num_output_buffer_need = rows * cols;
                                    unsigned int num_neuron_state_buffer_need = rows * cols;
                                    assert(num_output_buffer_need <= num_output);
                                    assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                    this->output_buffer = new int[num_output_buffer_need]();
                                    this->output_buffer_cpu = new int[num_output_buffer_need]();
                                    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                    this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                    Stonne* stonne_instance = new Stonne(stonne_cfg);
                                    matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                    //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                    stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                    stonne_instance->loadGEMMTile(cols,1,rows);
                                    stonne_instance->run();

                                    this->n_cycles += stonne_instance->n_cycles;
                                    // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                    // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                    for(int m = 0; m<num_output_buffer_need; m++){
                                        float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                        if(difference>0){
                                            std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                            std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                            std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                            // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                            // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                            assert(false);
                                        }
                                    }
                                    //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                    // // 输出输入数据计算结果和输出数据
                                    // std::cout<<"input data ---------------------: "<<std::endl;
                                    // for(int m=0;m<rows;m++){
                                    //     for(int a=0;a<this->layers[i].R * this->layers[i].S * this->layers[i].C;a++){
                                    //         std::cout<<this->spikes[m*this->layers[i].R * this->layers[i].S * this->layers[i].C + a]<<"   ";
                                    //     }
                                    //     std::cout<<std::endl;
                                    // }
                                    // std::cout<<"weight data -------------------:"<<std::endl;
                                    // for(int m=0;m<cols;m++){
                                    //     for(int a=0;a<this->layers[i].R * this->layers[i].S * this->layers[i].C;a++){
                                    //         std::cout<<this->weight_buffer[m*this->layers[i].R * this->layers[i].S * this->layers[i].C + a]<<"   ";
                                    //     }
                                    //     std::cout<<std::endl;
                                    // }

                                    // std::cout<<"output_sim data -----------------:"<<std::endl;
                                    // for(int m=0;m<num_output_buffer_need;m++){
                                    //     std::cout<<this->output_buffer[m]<<"    ";
                                    // }
                                    // std::cout<<std::endl;

                                    // std::cout<<"output_cpu data ------------------:"<<std::endl;
                                    // for(int m=0;m<num_output_buffer_need;m++){
                                    //     std::cout<<this->output_buffer_cpu[m]<<"    ";
                                    // }
                                    // std::cout<<std::endl;

                                    // std::cout<<"neuron_state_sim data--------------:"<<std::endl;
                                    // for(int m=0;m<num_neuron_state_buffer_need;m++){
                                    //     std::cout<<this->neuron_state_buffer[m]<<"  ";
                                    // }
                                    // std::cout<<std::endl;
                                    // std::cout<<"neuron_state_cpu data-----------------:"<<std::endl;
                                    // for(int m=0;m<num_neuron_state_buffer_need;m++){
                                    //     std::cout<<this->neuron_state_buffer_cpu[m]<<"  ";
                                    // }
                                    // std::cout<<std::endl;

                                    // 将中间状态写入DRAM
                                    //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                    int num_total_write_request = 0;
                                    bool only = true;
                                    while(true){
                                        while(only){
                                            for(int m=0; m<rows; m++){
                                                int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                                int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                                int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                                num_total_write_request += current_total_write_request;
                                                // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                                // assert(false);
                                                // 该行的基地址
                                                int write_addr_output = output_offset + ((0*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                                int write_addr_neuron_state = neuron_state_offset + ((0*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                                while(current_total_write_request!=0){
                                                    if(current_num_neuron_state_write_request!=0){
                                                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                                        this->write_request_fifo->push(write_request);
                                                        write_addr_neuron_state += addr_offset;
                                                        current_num_neuron_state_write_request--;
            
                                                    } else if(current_num_output_write_request!=0){
                                                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                                        this->write_request_fifo->push(write_request);
                                                        write_addr_output += addr_offset;
                                                        current_num_output_write_request--;
                                                    }
            
                                                    this->dram_instance->run();
                                                    this->n_cycles++; // 一个周期发送一个内存请求事务
                                                    current_total_write_request--;
                                                }
                                                // 模拟将片上buffer数据写入DRAM
                                                // 将当前行中的每一列数据写入DRAM
                                                for(int p=0; p<cols; p++){
                                                    //std::cout<<" write data in output_buffer loaction : "<<(m*cols + p)<<"   to DRAM location : "<<((times*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p )<<std::endl;
                                                    this->ofmap[(0*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                                    this->nfmap[(0*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                                }
                                            }
                                            only = false;
                                        }
            
                                        // std::cout<<"num : "<<num<<std::endl;
                                        // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
            
                                        // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
            
                                        if(completed_writes == num_total_write_request){
                                            completed_writes = 0;
                                            //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                            break;
                                        }
            
                                        this->dram_instance->run();
                                        this->n_cycles++;
                                    }

                                    // 重置数据，开始下一tile计算
                                    rows=0;
                                    input_arranged.clear();
                                    delete stonne_instance;
                                    delete[] this->spikes;
                                    delete[] this->output_buffer;
                                    delete[] this->output_buffer_cpu;
                                    delete[] this->neuron_state_buffer;
                                    delete[] this->neuron_state_buffer_cpu;


                                } else {  // 中间卷积窗口，正常移位的补零，但要考虑顶层补零
                                    for(int q=0; q<this->layers[i].C*this->layers[i].S; q++){
                                        im2col_bank[q] = 0;
                                    }
                                    for(int p=1; p<this->layers[i].R; p++){
                                        int base_addr = ((p-1)*this->layers[i].Y + (this->layers[i].S-this->layers[i].P) + (n-1)) * this->layers[i].C;
                                        for(int q=0; q<this->layers[i].C; q++){
                                            if(q==0){
                                                for(int num=0; num<(this->layers[i].S-1)*this->layers[i].C; num++){
                                                    im2col_bank[p * this->layers[i].C * this->layers[i].S + num] = im2col_bank[p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C]; 
                                                    //assert((p * this->layers[i].C * this->layers[i].S + num)<this->bankSize && (p * this->layers[i].C * this->layers[i].S + num)>=0);
                                                    //assert((p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C)<this->bankSize && (p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C)>=0);
                                                }
                                                // 取数据
                                                im2col_bank[p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q] = this->input_buffer[base_addr+q];
                                                //assert((p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)<this->bankSize && (p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)>=0);

                                                //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr+q)<<"    to   im2col_bank location : "<<(p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)<<std::endl;
                                                this->n_cycles++;
                                            } else {
                                                im2col_bank[p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q] = this->input_buffer[base_addr+q];
                                                this->n_cycles;
                                            }
                                        }
                                    }

                                    this->input_arranged.push_back(im2col_bank);
                                    rows++;

                                    if(rows == this->numBanks){
                                        // std::cout<<"current tile is : ============================================================ "<<num_tile<<std::endl;
                                        // std::cout<<"current tile rows is :     "<<rows<<std::endl;
                                        num_tile+=1;

                                        // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                        int total = rows * this->bankSize;
                                        this->spikes = new int[total];
                                        int index = 0;
                                        for(const auto& row : input_arranged){
                                            for(int val : row){
                                                this->spikes[index++] = val;
                                            }
                                        }

                                        // 根据需要例化存储输出结果的buffer
                                        unsigned int num_output_buffer_need = rows * cols;
                                        unsigned int num_neuron_state_buffer_need = rows * cols;
                                        assert(num_output_buffer_need <= num_output);
                                        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                        this->output_buffer = new int[num_output_buffer_need]();
                                        this->output_buffer_cpu = new int[num_output_buffer_need]();
                                        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                        this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                        Stonne* stonne_instance = new Stonne(stonne_cfg);
                                        matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                        stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                        stonne_instance->loadGEMMTile(cols,1,rows);
                                        stonne_instance->run();

                                        this->n_cycles += stonne_instance->n_cycles;
                                        // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                        // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                        for(int m = 0; m<num_output_buffer_need; m++){
                                            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                            if(difference>0){
                                                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                                // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                                // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                                assert(false);
                                            }
                                        }
                                        //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                        // 输出输入数据计算结果和输出数据
                                        // std::cout<<"input data ---------------------: "<<std::endl;
                                        // for(int m=0;m<rows;m++){
                                        //     for(int a=0;a<this->layers[i].R * this->layers[i].S * this->layers[i].C;a++){
                                        //         std::cout<<this->spikes[m*this->layers[i].R * this->layers[i].S * this->layers[i].C + a]<<"   ";
                                        //     }
                                        //     std::cout<<std::endl;
                                        // }
                                        // std::cout<<"weight data -------------------:"<<std::endl;
                                        // for(int m=0;m<cols;m++){
                                        //     for(int a=0;a<this->layers[i].R * this->layers[i].S * this->layers[i].C;a++){
                                        //         std::cout<<this->weight_buffer[m*this->layers[i].R * this->layers[i].S * this->layers[i].C + a]<<"   ";
                                        //     }
                                        //     std::cout<<std::endl;
                                        // }

                                        // std::cout<<"output_sim data -----------------:"<<std::endl;
                                        // for(int m=0;m<num_output_buffer_need;m++){
                                        //     std::cout<<this->output_buffer[m]<<"    ";
                                        // }
                                        // std::cout<<std::endl;

                                        // std::cout<<"output_cpu data ------------------:"<<std::endl;
                                        // for(int m=0;m<num_output_buffer_need;m++){
                                        //     std::cout<<this->output_buffer_cpu[m]<<"    ";
                                        // }
                                        // std::cout<<std::endl;

                                        // std::cout<<"neuron_state_sim data--------------:"<<std::endl;
                                        // for(int m=0;m<num_neuron_state_buffer_need;m++){
                                        //     std::cout<<this->neuron_state_buffer[m]<<"  ";
                                        // }
                                        // std::cout<<std::endl;
                                        // std::cout<<"neuron_state_cpu data-----------------:"<<std::endl;
                                        // for(int m=0;m<num_neuron_state_buffer_need;m++){
                                        //     std::cout<<this->neuron_state_buffer_cpu[m]<<"  ";
                                        // }
                                        // std::cout<<std::endl;

                                        // 将中间状态写入DRAM
                                        //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                        int num_total_write_request = 0;
                                        bool only = true;
                                        while(true){
                                            while(only){
                                                for(int m=0; m<rows; m++){
                                                    int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                                    int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                                    int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                                    num_total_write_request += current_total_write_request;
                                                    // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                                    // assert(false);
                                                    // 该行的基地址
                                                    int write_addr_output = output_offset + ((0*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                                    int write_addr_neuron_state = neuron_state_offset + ((0*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                                    while(current_total_write_request!=0){
                                                        if(current_num_neuron_state_write_request!=0){
                                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                                            this->write_request_fifo->push(write_request);
                                                            write_addr_neuron_state += addr_offset;
                                                            current_num_neuron_state_write_request--;
                
                                                        } else if(current_num_output_write_request!=0){
                                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                                            this->write_request_fifo->push(write_request);
                                                            write_addr_output += addr_offset;
                                                            current_num_output_write_request--;
                                                        }
                
                                                        this->dram_instance->run();
                                                        this->n_cycles++; // 一个周期发送一个内存请求事务
                                                        current_total_write_request--;
                                                    }
                                                    // 模拟将片上buffer数据写入DRAM
                                                    // 将当前行中的每一列数据写入DRAM
                                                    for(int p=0; p<cols; p++){
                                                        this->ofmap[(0*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                                        this->nfmap[(0*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                                    }
                                                }
                                                only = false;
                                            }
                
                                            // std::cout<<"num : "<<num<<std::endl;
                                            // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
                
                                            // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
                
                                            if(completed_writes == num_total_write_request){
                                                completed_writes = 0;
                                                //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                                
                                                break;
                                            }
                
                                            this->dram_instance->run();
                                            this->n_cycles++;
                                        }

                                        // 重置数据，开始下一tile计算

                                        rows=0;
                                        input_arranged.clear();
                                        delete stonne_instance;
                                        delete[] this->spikes;
                                        delete[] this->output_buffer;
                                        delete[] this->output_buffer_cpu;
                                        delete[] this->neuron_state_buffer;
                                        delete[] this->neuron_state_buffer_cpu;
                                    }
                                }
                            }

                            // 第二次，在底部补一行零
                            num_tile = 0;
                            for(int n=0; n<Y_; n++){
                                if(n==0){
                                    for(int p=0; p<(this->layers[i].R-1); p++){  // 这些行是真实的值
                                        int base_addr = p * this->layers[i].C * this->layers[i].Y;   
                                        for(int q=0; q<this->layers[i].C*this->layers[i].P; q++){  // 第一个卷积窗口有P列的零
                                            im2col_bank[p*this->layers[i].C*this->layers[i].S + q] = 0;  
                                        }
                                        for(int q=this->layers[i].C*this->layers[i].P; q<this->layers[i].C*this->layers[i].S; q++){
                                            im2col_bank[p*this->layers[i].C*this->layers[i].S + q] = this->input_buffer[base_addr+(q-this->layers[i].C*this->layers[i].P)];
                                            this->n_cycles;
                                        }
                                    }
                                    for(int q=0; q<this->layers[i].S*this->layers[i].C; q++){
                                        im2col_bank[(this->layers[i].R-1)*this->layers[i].S*this->layers[i].C + q] = 0;
                                    }

                                    this->input_arranged.push_back(im2col_bank);
                                    rows++;

                                } else if(n==Y_-1){

                                    // 移位 和 补零
                                    for(int p=0; p<this->layers[i].R; p++){
                                        for(int q=0; q<(this->layers[i].S-1)*this->layers[i].C; q++){
                                            im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = im2col_bank[p*this->layers[i].S*this->layers[i].C + q + this->layers[i].C]; 
                                        }
                                        for(int q=(this->layers[i].S-1)*this->layers[i].C; q<this->layers[i].S*this->layers[i].C; q++){
                                            im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = 0;
                                        }
                                    }

                                    // std::cout<<"The sorted data is : "<<std::endl;
                                    // for(int p=0; p<this->bankSize; p++){
                                    //     std::cout<<im2col_bank[p]<<"   ";
                                    // }
                                    // std::cout<<std::endl;

                                    this->input_arranged.push_back(im2col_bank);

                                    rows++;

                                    //if(rows == this->numBanks){
                                    // std::cout<<"current tile is : ============================================================ "<<num_tile<<std::endl;
                                    // std::cout<<"current tile rows is :     "<<rows<<std::endl;
                                    num_tile+=1;

                                    // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                    int total = rows * this->bankSize;
                                    this->spikes = new int[total];
                                    int index = 0;
                                    for(const auto& row : input_arranged){
                                        for(int val : row){
                                            this->spikes[index++] = val;
                                        }
                                    }

                                    // 根据需要例化存储输出结果的buffer
                                    unsigned int num_output_buffer_need = rows * cols;
                                    unsigned int num_neuron_state_buffer_need = rows * cols;
                                    assert(num_output_buffer_need <= num_output);
                                    assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                    this->output_buffer = new int[num_output_buffer_need]();
                                    this->output_buffer_cpu = new int[num_output_buffer_need]();
                                    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                    this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                    Stonne* stonne_instance = new Stonne(stonne_cfg);
                                    matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                    //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                    stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                    stonne_instance->loadGEMMTile(cols,1,rows);
                                    stonne_instance->run();

                                    this->n_cycles += stonne_instance->n_cycles;
                                    // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                    // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                    for(int m = 0; m<num_output_buffer_need; m++){
                                        float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                        if(difference>0){
                                            std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                            std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                            std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                            // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                            // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                            assert(false);
                                        }
                                    }
                                    //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                    // 将中间状态写入DRAM
                                    //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                    int num_total_write_request = 0;
                                    bool only = true;
                                    while(true){
                                        while(only){
                                            for(int m=0; m<rows; m++){
                                                int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                                int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                                int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                                num_total_write_request += current_total_write_request;
                                                // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                                // assert(false);
                                                // 该行的基地址
                                                int write_addr_output = output_offset + ((Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                                int write_addr_neuron_state = neuron_state_offset + ((Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                                while(current_total_write_request!=0){
                                                    if(current_num_neuron_state_write_request!=0){
                                                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                                        this->write_request_fifo->push(write_request);
                                                        write_addr_neuron_state += addr_offset;
                                                        current_num_neuron_state_write_request--;
            
                                                    } else if(current_num_output_write_request!=0){
                                                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                                        this->write_request_fifo->push(write_request);
                                                        write_addr_output += addr_offset;
                                                        current_num_output_write_request--;
                                                    }
            
                                                    this->dram_instance->run();
                                                    this->n_cycles++; // 一个周期发送一个内存请求事务
                                                    current_total_write_request--;
                                                }
                                                // 模拟将片上buffer数据写入DRAM
                                                // 将当前行中的每一列数据写入DRAM
                                                for(int p=0; p<cols; p++){
                                                    //std::cout<<" write data in output_buffer loaction : "<<(m*cols + p)<<"   to DRAM location : "<<(((k + times)*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p )<<std::endl;
                                                    
                                                    this->ofmap[(Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                                    this->nfmap[(Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                                }
                                            }
                                            only = false;
                                        }
            
                                        // std::cout<<"num : "<<num<<std::endl;
                                        // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
            
                                        // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
            
                                        if(completed_writes == num_total_write_request){
                                            completed_writes = 0;
                                            //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                            break;
                                        }
            
                                        this->dram_instance->run();
                                        this->n_cycles++;
                                    }

                                    // 重置数据，开始下一tile计算
                                    rows=0;
                                    input_arranged.clear();
                                    delete stonne_instance;
                                    delete[] this->spikes;
                                    delete[] this->output_buffer;
                                    delete[] this->output_buffer_cpu;
                                    delete[] this->neuron_state_buffer;
                                    delete[] this->neuron_state_buffer_cpu;


                                } else {
                                    for(int p=0; p<(this->layers[i].R-1); p++){
                                        int base_addr = (p * this->layers[i].Y + (this->layers[i].S-this->layers[i].P) + (n-1)) * this->layers[i].C;
                                        for(int q=0; q<this->layers[i].C; q++){
                                            if(q==0){
                                                for(int num=0; num<(this->layers[i].S-1); num++){
                                                    im2col_bank[p * this->layers[i].C * this->layers[i].S + num] = im2col_bank[p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C]; 
                                                }
                                                im2col_bank[p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q] = this->input_buffer[base_addr+q];
                                            } else {
                                                im2col_bank[p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q] = this->input_buffer[base_addr+q];
                                                this->n_cycles;
                                            }
                                        }
                                    }
                                    for(int q=0; q<this->layers[i].S*this->layers[i].C; q++){
                                        im2col_bank[(this->layers[i].R-1)*this->layers[i].S*this->layers[i].C + q] = 0;
                                    }

                                    this->input_arranged.push_back(im2col_bank);
                                    rows++;

                                    if(rows == this->numBanks){
                                        // std::cout<<"current tile is : ============================================================ "<<num_tile<<std::endl;
                                        // std::cout<<"current tile rows is :     "<<rows<<std::endl;
                                        num_tile+=1;

                                        // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                        int total = rows * this->bankSize;
                                        this->spikes = new int[total];
                                        int index = 0;
                                        for(const auto& row : input_arranged){
                                            for(int val : row){
                                                this->spikes[index++] = val;
                                            }
                                        }

                                        // 根据需要例化存储输出结果的buffer
                                        unsigned int num_output_buffer_need = rows * cols;
                                        unsigned int num_neuron_state_buffer_need = rows * cols;
                                        assert(num_output_buffer_need <= num_output);
                                        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                        this->output_buffer = new int[num_output_buffer_need]();
                                        this->output_buffer_cpu = new int[num_output_buffer_need]();
                                        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                        this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                        Stonne* stonne_instance = new Stonne(stonne_cfg);
                                        matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                        stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                        stonne_instance->loadGEMMTile(cols,1,rows);
                                        stonne_instance->run();

                                        this->n_cycles += stonne_instance->n_cycles;
                                        // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                        // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                        for(int m = 0; m<num_output_buffer_need; m++){
                                            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                            if(difference>0){
                                                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                                // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                                // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                                assert(false);
                                            }
                                        }
                                        //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                        // 将中间状态写入DRAM
                                        //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                        int num_total_write_request = 0;
                                        bool only = true;
                                        while(true){
                                            while(only){
                                                for(int m=0; m<rows; m++){
                                                    int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                                    int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                                    int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                                    num_total_write_request += current_total_write_request;
                                                    // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                                    // assert(false);
                                                    // 该行的基地址
                                                    int write_addr_output = output_offset + ((Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                                    int write_addr_neuron_state = neuron_state_offset + ((Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                                    while(current_total_write_request!=0){
                                                        if(current_num_neuron_state_write_request!=0){
                                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                                            this->write_request_fifo->push(write_request);
                                                            write_addr_neuron_state += addr_offset;
                                                            current_num_neuron_state_write_request--;
                
                                                        } else if(current_num_output_write_request!=0){
                                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                                            this->write_request_fifo->push(write_request);
                                                            write_addr_output += addr_offset;
                                                            current_num_output_write_request--;
                                                        }
                
                                                        this->dram_instance->run();
                                                        this->n_cycles++; // 一个周期发送一个内存请求事务
                                                        current_total_write_request--;
                                                    }
                                                    // 模拟将片上buffer数据写入DRAM
                                                    // 将当前行中的每一列数据写入DRAM
                                                    for(int p=0; p<cols; p++){
                                                        this->ofmap[(Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                                        this->nfmap[(Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                                    }
                                                }
                                                only = false;
                                            }
                
                                            // std::cout<<"num : "<<num<<std::endl;
                                            // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
                
                                            // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
                
                                            if(completed_writes == num_total_write_request){
                                                completed_writes = 0;
                                                //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                                break;
                                            }
                
                                            this->dram_instance->run();
                                            this->n_cycles++;
                                        }

                                        // 重置数据，开始下一tile计算
                                        rows=0;
                                        input_arranged.clear();
                                        delete stonne_instance;
                                        delete[] this->spikes;
                                        delete[] this->output_buffer;
                                        delete[] this->output_buffer_cpu;
                                        delete[] this->neuron_state_buffer;
                                        delete[] this->neuron_state_buffer_cpu;
                                    }

                                }
                            }

                        } else if(k==0){
                            for(int times=0; times<reused_times; times++){  // 重复使用读到buffer中的数据，在这里要每个循环在顶层加不同数量的0
                                int num_tile = 0;
                                //std::cout<<" ******************************* k==0 ,  times ******************************** : "<<times<<std::endl;
                                // 每次循环在顶部加（P - times）行0，在两侧加P列的0
                                
                                for(int n=0; n<Y_; n++){
                                    if(n==0){ // 
                                        for(int p=0; p<(this->layers[i].P-times); p++){  // 这么些行是0
                                            for(int q=0; q<this->layers[i].C*this->layers[i].S; q++){
                                                im2col_bank[p * this->layers[i].C * this->layers[i].S + q] = 0;
                                            }
                                        } 

                                        for(int p=(this->layers[i].P-times); p<this->layers[i].R; p++){  // 这些行是真实的值，但要考虑最左侧有P列的0
                                            int base_addr = (p - (this->layers[i].P -times)) * this->layers[i].C * this->layers[i].Y;
                                            //std::cout<<"base_addr : "<<base_addr<<std::endl;
                                            for(int q=0; q<this->layers[i].C*this->layers[i].P; q++){  // 这些列是0
                                                im2col_bank[p * this->layers[i].C * this->layers[i].S + q] = 0;
                                            }
                                            for(int q=this->layers[i].C*this->layers[i].P; q<this->layers[i].C*this->layers[i].S; q++){ //这些是真实的值
                                                im2col_bank[p * this->layers[i].C * this->layers[i].S + q] = this->input_buffer[base_addr+(q-this->layers[i].C*this->layers[i].P)];
                                                //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr+(q-this->layers[i].C*this->layers[i].P))<<"    to   im2col_bank location : "<<(p * this->layers[i].C * this->layers[i].S + q)<<std::endl;
                                                this->n_cycles++;
                                            }

                                        }

                                        // std::cout<<"The sorted data is : "<<std::endl;
                                        // for(int p=0; p<this->bankSize; p++){
                                        //     std::cout<<im2col_bank[p]<<"   ";
                                        // }
                                        // std::cout<<std::endl;

                                        this->input_arranged.push_back(im2col_bank);
                                        rows++;

                                    }else if(this->layers[i].P==2 && n==Y_-2){ 
                                        // 移位 和 补零
                                        for(int p=0; p<this->layers[i].R; p++){
                                            for(int q=0; q<(this->layers[i].S-1)*this->layers[i].C; q++){
                                                im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = im2col_bank[p*this->layers[i].S*this->layers[i].C + q + this->layers[i].C]; 
                                            }
                                            for(int q=(this->layers[i].S-1)*this->layers[i].C; q<this->layers[i].S*this->layers[i].C; q++){
                                                im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = 0;
                                            }
                                        }

                                        // std::cout<<"The sorted data is : "<<std::endl;
                                        // for(int p=0; p<this->bankSize; p++){
                                        //     std::cout<<im2col_bank[p]<<"   ";
                                        // }
                                        // std::cout<<std::endl;

                                        this->input_arranged.push_back(im2col_bank);

                                        rows++;
                                        assert(rows < this->numBanks); //如果此时刚好可以用于计算了，需要进行计算

                                    }else if( n==Y_-1 ){ // 最后一个卷积窗口，其右侧有1列0
                                        // 移位 和 补零
                                        for(int p=0; p<this->layers[i].R; p++){
                                            for(int q=0; q<(this->layers[i].S-1)*this->layers[i].C; q++){
                                                im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = im2col_bank[p*this->layers[i].S*this->layers[i].C + q + this->layers[i].C]; 
                                            }
                                            for(int q=(this->layers[i].S-1)*this->layers[i].C; q<this->layers[i].S*this->layers[i].C; q++){
                                                im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = 0;
                                            }
                                        }

                                        // std::cout<<"The sorted data is : "<<std::endl;
                                        // for(int p=0; p<this->bankSize; p++){
                                        //     std::cout<<im2col_bank[p]<<"   ";
                                        // }
                                        // std::cout<<std::endl;

                                        this->input_arranged.push_back(im2col_bank);

                                        rows++;

                                        //if(rows == this->numBanks){
                                        // std::cout<<"current tile is : ============================================================ "<<num_tile<<std::endl;
                                        // std::cout<<"current tile rows is :     "<<rows<<std::endl;
                                        num_tile+=1;

                                        // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                        int total = rows * this->bankSize;
                                        this->spikes = new int[total];
                                        int index = 0;
                                        for(const auto& row : input_arranged){
                                            for(int val : row){
                                                this->spikes[index++] = val;
                                            }
                                        }

                                        // 根据需要例化存储输出结果的buffer
                                        unsigned int num_output_buffer_need = rows * cols;
                                        unsigned int num_neuron_state_buffer_need = rows * cols;
                                        assert(num_output_buffer_need <= num_output);
                                        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                        this->output_buffer = new int[num_output_buffer_need]();
                                        this->output_buffer_cpu = new int[num_output_buffer_need]();
                                        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                        this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                        Stonne* stonne_instance = new Stonne(stonne_cfg);
                                        matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                        stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                        stonne_instance->loadGEMMTile(cols,1,rows);
                                        stonne_instance->run();

                                        this->n_cycles += stonne_instance->n_cycles;
                                        // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                        // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                        for(int m = 0; m<num_output_buffer_need; m++){
                                            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                            if(difference>0){
                                                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                                // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                                // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                                assert(false);
                                            }
                                        }
                                        //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                        // // 输出输入数据计算结果和输出数据
                                        // std::cout<<"input data ---------------------: "<<std::endl;
                                        // for(int m=0;m<rows;m++){
                                        //     for(int a=0;a<this->layers[i].R * this->layers[i].S * this->layers[i].C;a++){
                                        //         std::cout<<this->spikes[m*this->layers[i].R * this->layers[i].S * this->layers[i].C + a]<<"   ";
                                        //     }
                                        //     std::cout<<std::endl;
                                        // }
                                        // std::cout<<"weight data -------------------:"<<std::endl;
                                        // for(int m=0;m<cols;m++){
                                        //     for(int a=0;a<this->layers[i].R * this->layers[i].S * this->layers[i].C;a++){
                                        //         std::cout<<this->weight_buffer[m*this->layers[i].R * this->layers[i].S * this->layers[i].C + a]<<"   ";
                                        //     }
                                        //     std::cout<<std::endl;
                                        // }

                                        // std::cout<<"output_sim data -----------------:"<<std::endl;
                                        // for(int m=0;m<num_output_buffer_need;m++){
                                        //     std::cout<<this->output_buffer[m]<<"    ";
                                        // }
                                        // std::cout<<std::endl;

                                        // std::cout<<"output_cpu data ------------------:"<<std::endl;
                                        // for(int m=0;m<num_output_buffer_need;m++){
                                        //     std::cout<<this->output_buffer_cpu[m]<<"    ";
                                        // }
                                        // std::cout<<std::endl;

                                        // std::cout<<"neuron_state_sim data--------------:"<<std::endl;
                                        // for(int m=0;m<num_neuron_state_buffer_need;m++){
                                        //     std::cout<<this->neuron_state_buffer[m]<<"  ";
                                        // }
                                        // std::cout<<std::endl;
                                        // std::cout<<"neuron_state_cpu data-----------------:"<<std::endl;
                                        // for(int m=0;m<num_neuron_state_buffer_need;m++){
                                        //     std::cout<<this->neuron_state_buffer_cpu[m]<<"  ";
                                        // }
                                        // std::cout<<std::endl;

                                        // 将中间状态写入DRAM
                                        //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                        int num_total_write_request = 0;
                                        bool only = true;
                                        while(true){
                                            while(only){
                                                for(int m=0; m<rows; m++){
                                                    int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                                    int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                                    int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                                    num_total_write_request += current_total_write_request;
                                                    // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                                    // assert(false);
                                                    // 该行的基地址
                                                    int write_addr_output = output_offset + ((times*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                                    int write_addr_neuron_state = neuron_state_offset + ((times*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                                    while(current_total_write_request!=0){
                                                        if(current_num_neuron_state_write_request!=0){
                                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                                            this->write_request_fifo->push(write_request);
                                                            write_addr_neuron_state += addr_offset;
                                                            current_num_neuron_state_write_request--;
                
                                                        } else if(current_num_output_write_request!=0){
                                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                                            this->write_request_fifo->push(write_request);
                                                            write_addr_output += addr_offset;
                                                            current_num_output_write_request--;
                                                        }
                
                                                        this->dram_instance->run();
                                                        this->n_cycles++; // 一个周期发送一个内存请求事务
                                                        current_total_write_request--;
                                                    }
                                                    // 模拟将片上buffer数据写入DRAM
                                                    // 将当前行中的每一列数据写入DRAM
                                                    for(int p=0; p<cols; p++){
                                                        //std::cout<<" write data in output_buffer loaction : "<<(m*cols + p)<<"   to DRAM location : "<<((times*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p )<<std::endl;
                                                        this->ofmap[(times*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                                        this->nfmap[(times*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                                    }
                                                }
                                                only = false;
                                            }
                
                                            // std::cout<<"num : "<<num<<std::endl;
                                            // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
                
                                            // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
                
                                            if(completed_writes == num_total_write_request){
                                                completed_writes = 0;
                                                //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                                break;
                                            }
                
                                            this->dram_instance->run();
                                            this->n_cycles++;
                                        }

                                        // 重置数据，开始下一tile计算
                                        rows=0;
                                        input_arranged.clear();
                                        delete stonne_instance;
                                        delete[] this->spikes;
                                        delete[] this->output_buffer;
                                        delete[] this->output_buffer_cpu;
                                        delete[] this->neuron_state_buffer;
                                        delete[] this->neuron_state_buffer_cpu;

                                    }else { // 中间的窗口，考虑顶部有多少行的0
                                        //std::cout<<"n = "<<n<<std::endl;
                                        for(int p=0; p<(this->layers[i].P-times); p++){ // 这么些行是0
                                            for(int q=0; q<this->layers[i].C*this->layers[i].S; q++){
                                                im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = 0;
                                                assert((p*this->layers[i].S*this->layers[i].C + q)<this->bankSize && (p*this->layers[i].C*this->layers[i].C + q)>=0);
                                            }
                                        }
                                        for(int p=this->layers[i].P-times; p<this->layers[i].R; p++){  // 这些行有数值
                                            // 只取一列，考虑移位的取数据
                                            int base_addr = ((p-(this->layers[i].P-times)) * this->layers[i].Y + (this->layers[i].S-this->layers[i].P) + (n-1)) * this->layers[i].C; 
                                            // std::cout<<"base_addr : "<<base_addr<<std::endl;
                                            for(int q=0; q<this->layers[i].C; q++){
                                                if(q==0){
                                                    // 移位
                                                    for(int num=0; num<(this->layers[i].S-1)*this->layers[i].C; num++){
                                                        im2col_bank[p * this->layers[i].C * this->layers[i].S + num] = im2col_bank[p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C]; 
                                                        assert((p * this->layers[i].C * this->layers[i].S + num)<this->bankSize && (p * this->layers[i].C * this->layers[i].S + num)>=0);
                                                        assert((p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C)<this->bankSize && (p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C)>=0);
                                                    }
                                                    // 取数据
                                                    im2col_bank[p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q] = this->input_buffer[base_addr+q];
                                                    assert((p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)<this->bankSize && (p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)>=0);

                                                    //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr+q)<<"    to   im2col_bank location : "<<(p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)<<std::endl;
                                                    this->n_cycles++;
                                                } else{
                                                    im2col_bank[p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q] = this->input_buffer[base_addr+q];
                                                    assert((p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)<this->bankSize && (p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)>=0);

                                                    //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr+q)<<"    to   im2col_bank location : "<<(p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)<<std::endl;
                                                    this->n_cycles++;
                                                }
                                            }
                                        }

                                        // std::cout<<"The sorted data is : "<<std::endl;
                                        // for(int p=0; p<this->bankSize; p++){
                                        //     std::cout<<im2col_bank[p]<<"   ";
                                        // }
                                        // std::cout<<std::endl;

                                        //std::cout<<"debug"<<std::endl;
                                        this->input_arranged.push_back(im2col_bank);
                                        rows++;

                                        //std::cout<<"rows = "<<rows<<std::endl;

                                        if(rows == this->numBanks){
                                            // std::cout<<"current tile is : ============================================================ "<<num_tile<<std::endl;
                                            // std::cout<<"current tile rows is :     "<<rows<<std::endl;
                                            num_tile+=1;

                                            // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                            int total = rows * this->bankSize;
                                            this->spikes = new int[total];
                                            int index = 0;
                                            for(const auto& row : input_arranged){
                                                for(int val : row){
                                                    this->spikes[index++] = val;
                                                }
                                            }

                                            // 根据需要例化存储输出结果的buffer
                                            unsigned int num_output_buffer_need = rows * cols;
                                            unsigned int num_neuron_state_buffer_need = rows * cols;
                                            assert(num_output_buffer_need <= num_output);
                                            assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                            this->output_buffer = new int[num_output_buffer_need]();
                                            this->output_buffer_cpu = new int[num_output_buffer_need]();
                                            this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                            this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                            Stonne* stonne_instance = new Stonne(stonne_cfg);
                                            matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                            //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                            stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                            stonne_instance->loadGEMMTile(cols,1,rows);
                                            stonne_instance->run();

                                            this->n_cycles += stonne_instance->n_cycles;
                                            // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                            // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                            for(int m = 0; m<num_output_buffer_need; m++){
                                                float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                                if(difference>0){
                                                    std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                                    std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                                    std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                                    // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                                    // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                                    assert(false);
                                                }
                                            }
                                            //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                            // 输出输入数据计算结果和输出数据
                                            // std::cout<<"input data ---------------------: "<<std::endl;
                                            // for(int m=0;m<rows;m++){
                                            //     for(int a=0;a<this->layers[i].R * this->layers[i].S * this->layers[i].C;a++){
                                            //         std::cout<<this->spikes[m*this->layers[i].R * this->layers[i].S * this->layers[i].C + a]<<"   ";
                                            //     }
                                            //     std::cout<<std::endl;
                                            // }
                                            // std::cout<<"weight data -------------------:"<<std::endl;
                                            // for(int m=0;m<cols;m++){
                                            //     for(int a=0;a<this->layers[i].R * this->layers[i].S * this->layers[i].C;a++){
                                            //         std::cout<<this->weight_buffer[m*this->layers[i].R * this->layers[i].S * this->layers[i].C + a]<<"   ";
                                            //     }
                                            //     std::cout<<std::endl;
                                            // }

                                            // std::cout<<"output_sim data -----------------:"<<std::endl;
                                            // for(int m=0;m<num_output_buffer_need;m++){
                                            //     std::cout<<this->output_buffer[m]<<"    ";
                                            // }
                                            // std::cout<<std::endl;

                                            // std::cout<<"output_cpu data ------------------:"<<std::endl;
                                            // for(int m=0;m<num_output_buffer_need;m++){
                                            //     std::cout<<this->output_buffer_cpu[m]<<"    ";
                                            // }
                                            // std::cout<<std::endl;

                                            // std::cout<<"neuron_state_sim data--------------:"<<std::endl;
                                            // for(int m=0;m<num_neuron_state_buffer_need;m++){
                                            //     std::cout<<this->neuron_state_buffer[m]<<"  ";
                                            // }
                                            // std::cout<<std::endl;
                                            // std::cout<<"neuron_state_cpu data-----------------:"<<std::endl;
                                            // for(int m=0;m<num_neuron_state_buffer_need;m++){
                                            //     std::cout<<this->neuron_state_buffer_cpu[m]<<"  ";
                                            // }
                                            // std::cout<<std::endl;

                                            // 将中间状态写入DRAM
                                            //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                            int num_total_write_request = 0;
                                            bool only = true;
                                            while(true){
                                                while(only){
                                                    for(int m=0; m<rows; m++){
                                                        int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                                        int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                                        int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                                        num_total_write_request += current_total_write_request;
                                                        // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                                        // assert(false);
                                                        // 该行的基地址
                                                        int write_addr_output = output_offset + ((times*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                                        int write_addr_neuron_state = neuron_state_offset + ((times*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                                        while(current_total_write_request!=0){
                                                            if(current_num_neuron_state_write_request!=0){
                                                                std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                                                this->write_request_fifo->push(write_request);
                                                                write_addr_neuron_state += addr_offset;
                                                                current_num_neuron_state_write_request--;
                    
                                                            } else if(current_num_output_write_request!=0){
                                                                std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                                                this->write_request_fifo->push(write_request);
                                                                write_addr_output += addr_offset;
                                                                current_num_output_write_request--;
                                                            }
                    
                                                            this->dram_instance->run();
                                                            this->n_cycles++; // 一个周期发送一个内存请求事务
                                                            current_total_write_request--;
                                                        }
                                                        // 模拟将片上buffer数据写入DRAM
                                                        // 将当前行中的每一列数据写入DRAM
                                                        for(int p=0; p<cols; p++){
                                                            this->ofmap[(times*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                                            this->nfmap[(times*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                                        }
                                                    }
                                                    only = false;
                                                }
                    
                                                // std::cout<<"num : "<<num<<std::endl;
                                                // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
                    
                                                // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
                    
                                                if(completed_writes == num_total_write_request){
                                                    completed_writes = 0;
                                                    //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                                    
                                                    break;
                                                }
                    
                                                this->dram_instance->run();
                                                this->n_cycles++;
                                            }

                                            // 重置数据，开始下一tile计算

                                            rows=0;
                                            input_arranged.clear();
                                            delete stonne_instance;
                                            delete[] this->spikes;
                                            delete[] this->output_buffer;
                                            delete[] this->output_buffer_cpu;
                                            delete[] this->neuron_state_buffer;
                                            delete[] this->neuron_state_buffer_cpu;
                                        }

                                    }
                                }
                            }

                            k += (reused_times-1);

                        } else if(k==num_input_buffer_fold-reused_times){
                            
                            for(int times=0; times<reused_times; times++){  // 重复使用读到buffer中的数据，在不同循环中底部加不同数量的0
                                num_tile = 0;
                                //std::cout<<" ******************************* k=="<<k<<",  times ******************************** : "<<times<<std::endl;

                                // 在底部加times行0

                                for(int n=0; n<Y_; n++){
                                    if(n==0){
                                        // 在底部加times行0
                                        for(int p=0; p<(this->layers[i].R-times); p++){  // 这些行是真实的数据
                                            int base_addr = (p+times)*this->layers[i].C*this->layers[i].Y;
                                            //std::cout<<"base_addr : "<<base_addr<<std::endl;
                                            for(int q=0; q<this->layers[i].C*this->layers[i].P; q++){
                                                im2col_bank[p*this->layers[i].C*this->layers[i].S + q] = 0; 
                                            }
                                            for(int q=this->layers[i].C*this->layers[i].P; q<this->layers[i].C*this->layers[i].S; q++){
                                                im2col_bank[p*this->layers[i].C*this->layers[i].S + q] = this->input_buffer[base_addr+(q-this->layers[i].C*this->layers[i].P)];
                                                //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr+(q-this->layers[i].C*this->layers[i].P))<<"    to   im2col_bank location : "<<(p * this->layers[i].C * this->layers[i].S + q)<<std::endl;
                                                this->n_cycles++;
                                            }
                                        }
                                        for(int p=(this->layers[i].R-times); p<this->layers[i].R; p++){  // 这些行是0
                                            for(int q=0; q<this->layers[i].S*this->layers[i].C; q++){
                                                im2col_bank[p * this->layers[i].C * this->layers[i].S + q] = 0;
                                            }
                                        }

                                        // std::cout<<"The sorted data is : "<<std::endl;
                                        // for(int p=0; p<this->bankSize; p++){
                                        //     std::cout<<im2col_bank[p]<<"   ";
                                        // }
                                        // std::cout<<std::endl;

                                        this->input_arranged.push_back(im2col_bank);
                                        rows++;

                                    } else if(this->layers[i].P==2 && n==Y_-2){
                                        // 移位 和 补零
                                        for(int p=0; p<this->layers[i].R; p++){
                                            for(int q=0; q<(this->layers[i].S-1)*this->layers[i].C; q++){
                                                im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = im2col_bank[p*this->layers[i].S*this->layers[i].C + q + this->layers[i].C]; 
                                            }
                                            for(int q=(this->layers[i].S-1)*this->layers[i].C; q<this->layers[i].S*this->layers[i].C; q++){
                                                im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = 0;
                                            }
                                        }

                                        // std::cout<<"The sorted data is : "<<std::endl;
                                        // for(int p=0; p<this->bankSize; p++){
                                        //     std::cout<<im2col_bank[p]<<"   ";
                                        // }
                                        // std::cout<<std::endl;

                                        this->input_arranged.push_back(im2col_bank);

                                        rows++;
                                        assert(rows < this->numBanks); //如果此时刚好可以用于计算了，需要进行计算

                                    } else if(n==Y_-1){

                                        // 移位 和 补零
                                        for(int p=0; p<this->layers[i].R; p++){
                                            for(int q=0; q<(this->layers[i].S-1)*this->layers[i].C; q++){
                                                im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = im2col_bank[p*this->layers[i].S*this->layers[i].C + q + this->layers[i].C]; 
                                            }
                                            for(int q=(this->layers[i].S-1)*this->layers[i].C; q<this->layers[i].S*this->layers[i].C; q++){
                                                im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = 0;
                                            }
                                        }

                                        // std::cout<<"The sorted data is : "<<std::endl;
                                        // for(int p=0; p<this->bankSize; p++){
                                        //     std::cout<<im2col_bank[p]<<"   ";
                                        // }
                                        // std::cout<<std::endl;

                                        this->input_arranged.push_back(im2col_bank);

                                        rows++;

                                        //if(rows == this->numBanks){
                                        // std::cout<<"current tile is : ============================================================ "<<num_tile<<std::endl;
                                        // std::cout<<"current tile rows is :     "<<rows<<std::endl;
                                        num_tile+=1;

                                        // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                        int total = rows * this->bankSize;
                                        this->spikes = new int[total];
                                        int index = 0;
                                        for(const auto& row : input_arranged){
                                            for(int val : row){
                                                this->spikes[index++] = val;
                                            }
                                        }

                                        // 根据需要例化存储输出结果的buffer
                                        unsigned int num_output_buffer_need = rows * cols;
                                        unsigned int num_neuron_state_buffer_need = rows * cols;
                                        assert(num_output_buffer_need <= num_output);
                                        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                        this->output_buffer = new int[num_output_buffer_need]();
                                        this->output_buffer_cpu = new int[num_output_buffer_need]();
                                        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                        this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                        Stonne* stonne_instance = new Stonne(stonne_cfg);
                                        matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                        stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                        stonne_instance->loadGEMMTile(cols,1,rows);
                                        stonne_instance->run();

                                        this->n_cycles += stonne_instance->n_cycles;
                                        // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                        // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                        for(int m = 0; m<num_output_buffer_need; m++){
                                            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                            if(difference>0){
                                                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                                // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                                // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                                assert(false);
                                            }
                                        }
                                        //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                        // 将中间状态写入DRAM
                                        //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                        int num_total_write_request = 0;
                                        bool only = true;
                                        while(true){
                                            while(only){
                                                for(int m=0; m<rows; m++){
                                                    int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                                    int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                                    int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                                    num_total_write_request += current_total_write_request;
                                                    // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                                    // assert(false);
                                                    // 该行的基地址
                                                    int write_addr_output = output_offset + (((k + times)*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                                    int write_addr_neuron_state = neuron_state_offset + (((k + times)*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                                    while(current_total_write_request!=0){
                                                        if(current_num_neuron_state_write_request!=0){
                                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                                            this->write_request_fifo->push(write_request);
                                                            write_addr_neuron_state += addr_offset;
                                                            current_num_neuron_state_write_request--;
                
                                                        } else if(current_num_output_write_request!=0){
                                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                                            this->write_request_fifo->push(write_request);
                                                            write_addr_output += addr_offset;
                                                            current_num_output_write_request--;
                                                        }
                
                                                        this->dram_instance->run();
                                                        this->n_cycles++; // 一个周期发送一个内存请求事务
                                                        current_total_write_request--;
                                                    }
                                                    // 模拟将片上buffer数据写入DRAM
                                                    // 将当前行中的每一列数据写入DRAM
                                                    for(int p=0; p<cols; p++){
                                                        //std::cout<<" write data in output_buffer loaction : "<<(m*cols + p)<<"   to DRAM location : "<<(((k + times)*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p )<<std::endl;
                                                        
                                                        this->ofmap[((k + times)*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                                        this->nfmap[((k + times)*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                                    }
                                                }
                                                only = false;
                                            }
                
                                            // std::cout<<"num : "<<num<<std::endl;
                                            // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
                
                                            // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
                
                                            if(completed_writes == num_total_write_request){
                                                completed_writes = 0;
                                                //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                                break;
                                            }
                
                                            this->dram_instance->run();
                                            this->n_cycles++;
                                        }

                                        // 重置数据，开始下一tile计算
                                        rows=0;
                                        input_arranged.clear();
                                        delete stonne_instance;
                                        delete[] this->spikes;
                                        delete[] this->output_buffer;
                                        delete[] this->output_buffer_cpu;
                                        delete[] this->neuron_state_buffer;
                                        delete[] this->neuron_state_buffer_cpu;

                                    } else {
                                        for(int p=0; p<(this->layers[i].R-times); p++){  // 这些行是有效值
                                            int base_addr = ((p+times)*this->layers[i].Y + (this->layers[i].S-this->layers[i].P) + (n-1)) *this->layers[i].C;
                                            //std::cout<<"base_addr : "<<base_addr<<std::endl;
                                            for(int q=0; q<this->layers[i].C; q++){
                                                if(q==0){
                                                    // 移位
                                                    for(int num=0; num<(this->layers[i].S-1)*this->layers[i].C; num++){
                                                        im2col_bank[p * this->layers[i].C * this->layers[i].S + num] = im2col_bank[p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C]; 
                                                        //std::cout<<"Data movement process (in im2col location) : "<<(p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C)<<"<"<<im2col_bank[p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C]<<"> "<<"    to   "<<(p * this->layers[i].C * this->layers[i].S + num)<<std::endl;
                                                    }
                                                    // 取数据
                                                    im2col_bank[p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q] = this->input_buffer[base_addr+q];
                                                    //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr+q)<<"    to   im2col_bank location : "<<(p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)<<std::endl;
                                                    this->n_cycles++;
                                                    
                                                } else{
                                                    im2col_bank[p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q] = this->input_buffer[base_addr+q];
                                                    //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr+q)<<"    to   im2col_bank location : "<<(p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)<<std::endl;
                                                    this->n_cycles++;
                                                }
                                            }
                                        }
                                        for(int p=(this->layers[i].R-times); p<this->layers[i].R; p++){  // 这些行是0
                                            for(int q=0; q<this->layers[i].C*this->layers[i].S; q++){
                                                im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = 0;
                                                //std::cout<<"set 0 in location : "<<(p*this->layers[i].S*this->layers[i].C + q)<<std::endl;
                        
                                            }
                                        }

                                        // std::cout<<"The sorted data is : "<<std::endl;
                                        // for(int p=0; p<this->bankSize; p++){
                                        //     std::cout<<im2col_bank[p]<<"   ";
                                        // }
                                        // std::cout<<std::endl;
                                        // std::cout<<std::endl;

                                        this->input_arranged.push_back(im2col_bank);
                                        rows++;

                                        if(rows == this->numBanks){
                                            // std::cout<<"current tile is : ============================================================ "<<num_tile<<std::endl;
                                            // std::cout<<"current tile rows is :     "<<rows<<std::endl;
                                            num_tile+=1;

                                            // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                            int total = rows * this->bankSize;
                                            this->spikes = new int[total];
                                            int index = 0;
                                            for(const auto& row : input_arranged){
                                                for(int val : row){
                                                    this->spikes[index++] = val;
                                                }
                                            }

                                            // 根据需要例化存储输出结果的buffer
                                            unsigned int num_output_buffer_need = rows * cols;
                                            unsigned int num_neuron_state_buffer_need = rows * cols;
                                            assert(num_output_buffer_need <= num_output);
                                            assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                            this->output_buffer = new int[num_output_buffer_need]();
                                            this->output_buffer_cpu = new int[num_output_buffer_need]();
                                            this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                            this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                            Stonne* stonne_instance = new Stonne(stonne_cfg);
                                            matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                            //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                            stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                            stonne_instance->loadGEMMTile(cols,1,rows);
                                            stonne_instance->run();

                                            this->n_cycles += stonne_instance->n_cycles;
                                            // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                            // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                            for(int m = 0; m<num_output_buffer_need; m++){
                                                float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                                if(difference>0){
                                                    std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                                    std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                                    std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                                    // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                                    // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                                    assert(false);
                                                }
                                            }
                                            //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                            // 将中间状态写入DRAM
                                            //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                            int num_total_write_request = 0;
                                            bool only = true;
                                            while(true){
                                                while(only){
                                                    for(int m=0; m<rows; m++){
                                                        int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                                        int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                                        int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                                        num_total_write_request += current_total_write_request;
                                                        // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                                        // assert(false);
                                                        // 该行的基地址
                                                        int write_addr_output = output_offset + (((k+times)*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                                        int write_addr_neuron_state = neuron_state_offset + (((k+times)*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                                        while(current_total_write_request!=0){
                                                            if(current_num_neuron_state_write_request!=0){
                                                                std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                                                this->write_request_fifo->push(write_request);
                                                                write_addr_neuron_state += addr_offset;
                                                                current_num_neuron_state_write_request--;
                    
                                                            } else if(current_num_output_write_request!=0){
                                                                std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                                                this->write_request_fifo->push(write_request);
                                                                write_addr_output += addr_offset;
                                                                current_num_output_write_request--;
                                                            }
                    
                                                            this->dram_instance->run();
                                                            this->n_cycles++; // 一个周期发送一个内存请求事务
                                                            current_total_write_request--;
                                                        }
                                                        // 模拟将片上buffer数据写入DRAM
                                                        // 将当前行中的每一列数据写入DRAM
                                                        for(int p=0; p<cols; p++){
                                                            this->ofmap[((k+times)*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                                            this->nfmap[((k+times)*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                                        }
                                                    }
                                                    only = false;
                                                }
                    
                                                // std::cout<<"num : "<<num<<std::endl;
                                                // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
                    
                                                // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
                    
                                                if(completed_writes == num_total_write_request){
                                                    completed_writes = 0;
                                                    //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                                    break;
                                                }
                    
                                                this->dram_instance->run();
                                                this->n_cycles++;
                                            }

                                            // 重置数据，开始下一tile计算
                                            rows=0;
                                            input_arranged.clear();
                                            delete stonne_instance;
                                            delete[] this->spikes;
                                            delete[] this->output_buffer;
                                            delete[] this->output_buffer_cpu;
                                            delete[] this->neuron_state_buffer;
                                            delete[] this->neuron_state_buffer_cpu;
                                        }

                                    }
                                }
                                
                            }


                            k += (reused_times-1);

                        } else{
                            num_tile = 0;
                            // 读到buffer中的数据只用一次，只需考虑两侧加一定数量的0
                            for(int n=0; n<Y_; n++){
                                if(n==0){  // 第一个卷积窗口，考虑其左侧有P列的0

                                    for(int p=0; p<this->layers[i].R; p++){
                                        int base_addr = p * this->layers[i].C * this->layers[i].Y;
                                        //std::cout<<"base_addr : "<<base_addr<<std::endl;
                                        for(int q=0; q<this->layers[i].C*this->layers[i].P; q++){
                                            im2col_bank[p*this->layers[i].C*this->layers[i].S + q] = 0;
                                            //std::cout<<"set 0 in im2col_bank location : "<<(p*this->layers[i].C*this->layers[i].S + q)<<std::endl;
                                        }
                                        for(int q=this->layers[i].C*this->layers[i].P; q<this->layers[i].C*this->layers[i].S; q++){
                                            im2col_bank[p*this->layers[i].C*this->layers[i].S + q] = this->input_buffer[base_addr+(q-this->layers[i].C*this->layers[i].P)];
                                            //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr+(q-this->layers[i].C*this->layers[i].P))<<"    to   im2col_bank location : "<<(p*this->layers[i].C*this->layers[i].S + q)<<std::endl;
                                                    
                                            this->n_cycles++;
                                        }
                                    }

                                    // std::cout<<"The sorted data is : "<<std::endl;
                                    // for(int p=0; p<this->bankSize; p++){
                                    //     std::cout<<im2col_bank[p]<<"   ";
                                    // }
                                    // std::cout<<std::endl;

                                    this->input_arranged.push_back(im2col_bank);
                                    rows++;

                                } else if(this->layers[i].P==2 && n==Y_-2){  // 移位，补零
                                    for(int p=0; p<this->layers[i].R; p++){
                                        for(int q=0; q<(this->layers[i].S-1)*this->layers[i].C; q++){
                                            im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = im2col_bank[p*this->layers[i].S*this->layers[i].C + q + this->layers[i].C]; 
                                        }
                                        for(int q=(this->layers[i].S-1)*this->layers[i].C; q<this->layers[i].S*this->layers[i].C; q++){
                                            im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = 0;
                                        }
                                    }

                                    // std::cout<<"The sorted data is : "<<std::endl;
                                    // for(int p=0; p<this->bankSize; p++){
                                    //     std::cout<<im2col_bank[p]<<"   ";
                                    // }
                                    // std::cout<<std::endl;

                                    this->input_arranged.push_back(im2col_bank);

                                    rows++;
                                    assert(rows < this->numBanks); //如果此时刚好可以用于计算了，需要进行计算
 
                                } else if(n==Y_-1){  // 移位，补零

                                    for(int p=0; p<this->layers[i].R; p++){
                                        for(int q=0; q<(this->layers[i].S-1)*this->layers[i].C; q++){
                                            im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = im2col_bank[p*this->layers[i].S*this->layers[i].C + q + this->layers[i].C]; 
                                        }
                                        for(int q=(this->layers[i].S-1)*this->layers[i].C; q<this->layers[i].S*this->layers[i].C; q++){
                                            im2col_bank[p*this->layers[i].S*this->layers[i].C + q] = 0;
                                        }
                                    }

                                    // std::cout<<"The sorted data is : "<<std::endl;
                                    // for(int p=0; p<this->bankSize; p++){
                                    //     std::cout<<im2col_bank[p]<<"   ";
                                    // }
                                    // std::cout<<std::endl;

                                    this->input_arranged.push_back(im2col_bank);

                                    rows++;

                                    //if(rows == this->numBanks){
                                    // std::cout<<"current tile is : ============================================================ "<<num_tile<<std::endl;
                                    // std::cout<<"current tile rows is :     "<<rows<<std::endl;
                                    num_tile+=1;

                                    // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                    int total = rows * this->bankSize;
                                    this->spikes = new int[total];
                                    int index = 0;
                                    for(const auto& row : input_arranged){
                                        for(int val : row){
                                            this->spikes[index++] = val;
                                        }
                                    }

                                    // 根据需要例化存储输出结果的buffer
                                    unsigned int num_output_buffer_need = rows * cols;
                                    unsigned int num_neuron_state_buffer_need = rows * cols;
                                    assert(num_output_buffer_need <= num_output);
                                    assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                    this->output_buffer = new int[num_output_buffer_need]();
                                    this->output_buffer_cpu = new int[num_output_buffer_need]();
                                    this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                    this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                    Stonne* stonne_instance = new Stonne(stonne_cfg);
                                    matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                    //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                    stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                    stonne_instance->loadGEMMTile(cols,1,rows);
                                    stonne_instance->run();

                                    this->n_cycles += stonne_instance->n_cycles;
                                    // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                    // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                    for(int m = 0; m<num_output_buffer_need; m++){
                                        float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                        if(difference>0){
                                            std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                            std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                            std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                            // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                            // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                            assert(false);
                                        }
                                    }
                                    //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                    // 将中间状态写入DRAM
                                    //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                    int num_total_write_request = 0;
                                    bool only = true;
                                    while(true){
                                        while(only){
                                            for(int m=0; m<rows; m++){
                                                int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                                int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                                int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                                num_total_write_request += current_total_write_request;
                                                // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                                // assert(false);
                                                // 该行的基地址
                                                int write_addr_output = output_offset + ((k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                                int write_addr_neuron_state = neuron_state_offset + ((k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                                while(current_total_write_request!=0){
                                                    if(current_num_neuron_state_write_request!=0){
                                                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                                        this->write_request_fifo->push(write_request);
                                                        write_addr_neuron_state += addr_offset;
                                                        current_num_neuron_state_write_request--;
            
                                                    } else if(current_num_output_write_request!=0){
                                                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                                        this->write_request_fifo->push(write_request);
                                                        write_addr_output += addr_offset;
                                                        current_num_output_write_request--;
                                                    }
            
                                                    this->dram_instance->run();
                                                    this->n_cycles++; // 一个周期发送一个内存请求事务
                                                    current_total_write_request--;
                                                }
                                                // 模拟将片上buffer数据写入DRAM
                                                // 将当前行中的每一列数据写入DRAM
                                                for(int p=0; p<cols; p++){
                                                    this->ofmap[(k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                                    this->nfmap[(k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                                }
                                            }
                                            only = false;
                                        }
            
                                        // std::cout<<"num : "<<num<<std::endl;
                                        // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
            
                                        // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
            
                                        if(completed_writes == num_total_write_request){
                                            completed_writes = 0;
                                            //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                            break;
                                        }
            
                                        this->dram_instance->run();
                                        this->n_cycles++;
                                    }

                                    // 重置数据，开始下一tile计算
                                    rows=0;
                                    input_arranged.clear();
                                    delete stonne_instance;
                                    delete[] this->spikes;
                                    delete[] this->output_buffer;
                                    delete[] this->output_buffer_cpu;
                                    delete[] this->neuron_state_buffer;
                                    delete[] this->neuron_state_buffer_cpu;

                                } else {
                                    // 中间窗口，正常的移位和取数据
                                    for(int p=0; p<this->layers[i].R; p++){
                                        int base_addr = (p * this->layers[i].Y + (this->layers[i].S-this->layers[i].P) + (n-1)) * this->layers[i].C;
                                        //std::cout<<"base_addr : "<<base_addr<<std::endl;
                                        for(int q=0; q<this->layers[i].C; q++){
                                            if(q==0){
                                                // 移位
                                                for(int num=0; num<(this->layers[i].S-1)*this->layers[i].C; num++){
                                                    im2col_bank[p * this->layers[i].C * this->layers[i].S + num] = im2col_bank[p * this->layers[i].C * this->layers[i].S + num + this->layers[i].C]; 
                                                }
                                                // 取数据
                                                im2col_bank[p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q] = this->input_buffer[base_addr+q];
                                                //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr+q)<<"    to   im2col_bank location : "<<(p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)<<std::endl;
                                                this->n_cycles++;
                                            } else{
                                                im2col_bank[p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q] = this->input_buffer[base_addr+q];
                                                //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr+q)<<"    to   im2col_bank location : "<<(p * this->layers[i].C * this->layers[i].S + (this->layers[i].S-1)*this->layers[i].C + q)<<std::endl;
                                                this->n_cycles++;
                                            }
                                        }
            
                                    }
                                    // std::cout<<"The sorted data is : "<<std::endl;

                                    // for(int p=0; p<this->bankSize; p++){
                                    //     std::cout<<im2col_bank[p]<<"   ";
                                    // }
                                    // std::cout<<std::endl;

                                    this->input_arranged.push_back(im2col_bank);
                                    rows++;

                                    if(rows == this->numBanks){
                                        // std::cout<<"current tile is : ============================================================ "<<num_tile<<std::endl;
                                        // std::cout<<"current tile rows is :     "<<rows<<std::endl;
                                        num_tile+=1;

                                        // 将this->input_arranged转化为数组，然后送入脉动阵列计算
                                        int total = rows * this->bankSize;
                                        this->spikes = new int[total];
                                        int index = 0;
                                        for(const auto& row : input_arranged){
                                            for(int val : row){
                                                this->spikes[index++] = val;
                                            }
                                        }

                                        // 根据需要例化存储输出结果的buffer
                                        unsigned int num_output_buffer_need = rows * cols;
                                        unsigned int num_neuron_state_buffer_need = rows * cols;
                                        assert(num_output_buffer_need <= num_output);
                                        assert(num_neuron_state_buffer_need <= num_neuron_state_buffer_need);
                                        this->output_buffer = new int[num_output_buffer_need]();
                                        this->output_buffer_cpu = new int[num_output_buffer_need]();
                                        this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                                        this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                                        Stonne* stonne_instance = new Stonne(stonne_cfg);
                                        matrixMultiply(rows, this->bankSize, cols, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                                        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                                        stonne_instance->loadDenseGEMM(layer_name,cols,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                                        stonne_instance->loadGEMMTile(cols,1,rows);
                                        stonne_instance->run();

                                        this->n_cycles += stonne_instance->n_cycles;
                                        // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                                        // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                                        for(int m = 0; m<num_output_buffer_need; m++){
                                            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                                            if(difference>0){
                                                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                                // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                                // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                                assert(false);
                                            }
                                        }
                                        //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                                        // 将中间状态写入DRAM
                                        //std::cout<<"begin write neuron_state =========================================================="<<std::endl;
                                        int num_total_write_request = 0;
                                        bool only = true;
                                        while(true){
                                            while(only){
                                                for(int m=0; m<rows; m++){
                                                    int current_num_neuron_state_write_request = std::ceil(cols*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)dram_instance->dram->GetBusBits());
                                                    int current_num_output_write_request = std::ceil(cols / (float)dram_instance->dram->GetBusBits());
                                                    int current_total_write_request = current_num_output_write_request + current_num_neuron_state_write_request;
                                                    num_total_write_request += current_total_write_request;
                                                    // std::cout<<"current_total_write_request : "<<current_total_write_request<<std::endl;
                                                    // assert(false);
                                                    // 该行的基地址
                                                    int write_addr_output = output_offset + ((k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) / 8;
                                                    int write_addr_neuron_state = neuron_state_offset + ((k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols) * (std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;
                                                    while(current_total_write_request!=0){
                                                        if(current_num_neuron_state_write_request!=0){
                                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_neuron_state,true);
                                                            this->write_request_fifo->push(write_request);
                                                            write_addr_neuron_state += addr_offset;
                                                            current_num_neuron_state_write_request--;
                
                                                        } else if(current_num_output_write_request!=0){
                                                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr_output,true);
                                                            this->write_request_fifo->push(write_request);
                                                            write_addr_output += addr_offset;
                                                            current_num_output_write_request--;
                                                        }
                
                                                        this->dram_instance->run();
                                                        this->n_cycles++; // 一个周期发送一个内存请求事务
                                                        current_total_write_request--;
                                                    }
                                                    // 模拟将片上buffer数据写入DRAM
                                                    // 将当前行中的每一列数据写入DRAM
                                                    for(int p=0; p<cols; p++){
                                                        //std::cout<<" write data in output_buffer loaction : "<<(m*cols + p)<<"   to DRAM location : "<<((k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p )<<std::endl;
                                                        
                                                        this->ofmap[(k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->output_buffer[m*cols + p]; 
                                                        this->nfmap[(k*Y_ + (num_tile-1)*this->stonne_cfg.m_MSNetworkCfg.ms_rows + m)*this->layers[i].K + j*this->stonne_cfg.m_MSNetworkCfg.ms_cols + p ] = this->neuron_state_buffer[m*cols + p]; 
                                                    }
                                                }
                                                only = false;
                                            }
                
                                            // std::cout<<"num : "<<num<<std::endl;
                                            // std::cout<<"num_neuron_state_write_request : "<<num_total_write_request<<std::endl;
                
                                            // std::cout<<"completed_writes : "<<completed_writes<<std::endl;
                
                                            if(completed_writes == num_total_write_request){
                                                completed_writes = 0;
                                                //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                                break;
                                            }
                
                                            this->dram_instance->run();
                                            this->n_cycles++;
                                        }

                                        // 重置数据，开始下一tile计算
                                        rows=0;
                                        input_arranged.clear();
                                        delete stonne_instance;
                                        delete[] this->spikes;
                                        delete[] this->output_buffer;
                                        delete[] this->output_buffer_cpu;
                                        delete[] this->neuron_state_buffer;
                                        delete[] this->neuron_state_buffer_cpu;
                                    }


                                }
                            }

                        }

                        //std::cout<<"k : ==========================================="<<k<<std::endl;
                    }
                }

                delete[] this->input_buffer;
                delete[] this->weight_buffer;
            }
 
            // 计算完毕，验证写到DRAM的结果是否正确
            conv_compute(this->layers[i].R, this->layers[i].S, this->layers[i].C, this->layers[i].K, this->layers[i].P, this->layers[i].stride, this->layers[i].X, this->layers[i].Y, this->ifmap, this->filter, this->ofmap_cpu, this->nfmap_cpu, this->stonne_cfg.V_th);

            for(int j=0; j<X_*Y_*this->layers[i].K; j++){
                //float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
                float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]) + fabs(this->nfmap[j]-this->nfmap_cpu[j]);
                if(difference>0){
                    std::cout<<"ERROR position : "<<j<<"  value ofmap simlator : "<<this->ofmap[j]<<"  value ofmap cpu : "<<this->ofmap_cpu[j]<<std::endl;
                    std::cout<<"ERROR position : "<<j<<"  value nfmap simlator : "<<this->nfmap[j]<<"  value nfmap cpu : "<<this->nfmap_cpu[j]<<std::endl;
                    assert(false);
                }
            }

            std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

        } else if(layers[i].type == "fc"){
            //std::cout<<"current layer is : fc"<<std::endl;
            // 全连接层
            // 权重数据从阵列水平方向传入，输入数据从阵列垂直方向传入
            this->stonne_cfg.layer_type = FC;
            this->pooling_enabled = false;
            this->n_fc++;
            this->layer_name = "fc" + std::to_string(this->n_fc);

            int output_layer = this->layers[i].output_neuron;
            int input_layer = this->layers[i].input_neuron;

            // 建模存储真实的数据
            this->ifmap = new int[input_layer];
            this->filter = new int[output_layer * input_layer];
            this->ofmap = new int[output_layer]();
            this->ofmap_cpu = new int[output_layer]();
            this->nfmap = new int[output_layer]();
            this->nfmap_cpu = new int[output_layer]();

            for(int n=0; n<input_layer; n++){
                this->ifmap[n] = rand()%2;
            }

            for(int n=0; n<output_layer*input_layer; n++){
                this->filter[n] = rand()%((this->max_weight-this->min_weight+1)+this->min_weight);
            }

            // 权重需要重复从DRAM取出的次数
            // 输入一次读完，且只需要读一次
            int num_weight_buffer_fold = std::ceil(output_layer / (float)this->stonne_cfg.m_MSNetworkCfg.ms_rows);

            std::cout<<"\033[1m\033[33m Start simulation layer :  \033[0m"<<this->layer_name<<std::endl;

            // 例化dram
            this->dram_instance = new Dram(read_callback, write_callback);
            this->dram_instance->set_read_request_fifo(this->read_request_fifo);
            this->dram_instance->set_write_request_fifo(this->write_request_fifo);
            
            // 各类数据在DRAM中的起始地址
            uint64_t input_offset = 0x0000;  // 输入数据在DRAM中的起始地址
            uint64_t weight_offset = this->input_dram_size*1024*1024; // 权重数据在DRAM中的起始地址
            uint64_t weight_offset_copy = weight_offset;
            uint64_t output_offset = (this->input_dram_size + this->weight_dram_size)*1024*1024;  // 输出数据在DRAM中的起始地址
            uint64_t output_offset_copy = output_offset;
            uint64_t neuron_state_offset = (this->input_dram_size + this->weight_dram_size + this->output_dram_size)*1024*1024; // 神经元状态在DRAM中的起始地址
            uint64_t neuron_state_offset_copy = neuron_state_offset;
            uint64_t addr_offset = 8;  // 连续读取时，每次地址加8，因为一次可以读出64bit数据

            // 记录已经从片外取出的权重数
            int num_weight_obtained = 0;

            // 读输入到输入buffer，只需要读这一次
            unsigned int num_input_buffer_need = input_layer;
            assert(num_input_buffer_need <= num_input);
            this->input_buffer = new int[num_input_buffer_need];

            int num_input_data = input_layer; 
            int num_input_read_request = std::ceil(num_input_data / (float)dram_instance->dram->GetBusBits());
            int num_input_read_request_copy = num_input_read_request;

            while(true){

                if(num_input_read_request!=0){
                    std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(input_offset,false);
                    this->read_request_fifo->push(read_request);
                    input_offset += addr_offset;
                    num_input_read_request--;
                }

                if(completed_reads == num_input_read_request_copy) {
                    for(int k=0; k<input_layer; k++){
                        this->input_buffer[k] = this->ifmap[k];
                    }
                    completed_reads = 0;
                    break;
                }

                this->dram_instance->run();
                this->n_cycles++;
            }

            int num_output_written = 0;
            int num_neuron_state_written = 0;

            for(int j=0; j<num_weight_buffer_fold; j++){
                //std::cout<<"=====================j============================= : "<<j<<std::endl;
                
                // 需要从片外取出的数据个数、需要向DRAM发送的读请求个数
                int num_weight_data;
                int num_weight_read_request;
                int num_weight_read_request_copy;

                // 读取权重数据
                int start_row = j*this->stonne_cfg.m_MSNetworkCfg.ms_rows;
                int end_row = std::min<int>(start_row+this->stonne_cfg.m_MSNetworkCfg.ms_rows, output_layer);
                int delta_weight = this->stonne_cfg.m_MSNetworkCfg.ms_rows - (end_row-start_row);

                // 根据需要实例化片上buffer，input_layer行，列数是脉动阵列的列数
                
                unsigned int num_weight_buffer_need = (end_row-start_row) * input_layer;
                unsigned int num_output_buffer_need = end_row-start_row;
                unsigned int num_neuron_state_buffer_need = end_row-start_row;
                assert(num_weight_buffer_need <= num_weight);
                assert(num_output_buffer_need <= num_output);
                assert(num_neuron_state_buffer_need <= num_neuron_state);
                this->weight_buffer = new int[num_weight_buffer_need];            
                this->output_buffer = new int[num_output_buffer_need]();
                this->neuron_state_buffer = new int[num_neuron_state_buffer_need]();
                this->output_buffer_cpu = new int[num_output_buffer_need]();
                this->neuron_state_buffer_cpu = new int[num_neuron_state_buffer_need]();

                if(delta_weight>0) {
                    num_weight_data = input_layer * (end_row-start_row);
                } else{
                    num_weight_data = input_layer * this->stonne_cfg.m_MSNetworkCfg.ms_rows;
                }

                num_weight_read_request = std::ceil(num_weight_data*this->weight_width /(float)dram_instance->dram->GetBusBits());
                num_weight_read_request_copy = num_weight_read_request;

                // std::cout<<"num_total_read_request : =================================="<<num_total_read_request<<std::endl;
                // std::cout<<"num_input_read_request : "<<num_input_read_request<<std::endl;
                // std::cout<<"num_weight_read_request: "<<num_weight_read_request<<std::endl;
                // std::cout<<"this->weight_width : "<<this->weight_width<<std::endl;
                // std::cout<<"num_weight_data : "<<num_weight_data<<std::endl;

                while(true){  // 发送DRAM内存事务请求

                    if(num_weight_read_request!=0){
                        std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(weight_offset,false);
                        this->read_request_fifo->push(read_request);
                        weight_offset += addr_offset; 
                        num_weight_read_request--;
                    }

                    if(completed_reads == num_weight_read_request_copy){  // 内存读取事务响应完毕
                        // 将片外的数据存储片上buffer
                        // 权重数据
                        for(int k=0; k<num_weight_data; k++){
                            this->weight_buffer[k] = this->filter[num_weight_obtained+k];
                        }

                        num_weight_obtained += num_weight_data;
                        weight_offset = weight_offset_copy + num_weight_obtained*this->weight_width/8;  // 下一轮取数据的基地址
                        completed_reads = 0;
                        //std::cout<<"All read requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                        break;
                    }
                    this->dram_instance->run();
                    this->n_cycles++;
                }

                // 得到数据之后，送入计算单元计算
                Stonne* stonne_instance = new Stonne(stonne_cfg);
                matrixMultiply((end_row-start_row),input_layer,1, this->weight_buffer, this->input_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->pooling_enabled);
                //sequential_layer(1, input_layer, 1, this->stonne_cfg.m_MSNetworkCfg.ms_rows, 1, 1, this->stonne_cfg.m_MSNetworkCfg.ms_cols, input_layer, 1, this->weight_buffer, this->input_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, this->stonne_cfg.V_th, this->Timestamp, this->pooling_enabled);
                //std::cout<<"debug"<<std::endl;
                stonne_instance->loadDenseGEMM(layer_name, 1, input_layer, (end_row-start_row), this->weight_buffer, this->input_buffer, this->output_buffer, this->neuron_state_buffer, CNN_DATAFLOW);
                //std::cout<<"debug"<<std::endl;
                stonne_instance->loadGEMMTile(1, 1,(end_row-start_row));
                stonne_instance->run();

                this->n_cycles += stonne_instance->n_cycles;
                // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                // 对比结果
                // 输出
                // std::cout<<"the output of sim: -------------------------------------"<<std::endl;
                // for(int k=0; k<num_output_buffer_need; k++){
                //     std::cout<<this->output_buffer[k]<<"    ";
                // }
                // std::cout<<std::endl;
                // std::cout<<"the output of cpu: -------------------------------------"<<std::endl;
                // for(int k=0; k<num_output_buffer_need; k++){
                //     std::cout<<this->output_buffer_cpu[k]<<"    ";
                // }
                // std::cout<<std::endl;
                // std::cout<<"the neuron state of sim: -------------------------------------"<<std::endl;
                // for(int k=0; k<num_neuron_state_buffer_need; k++){
                //     std::cout<<this->neuron_state_buffer[k]<<"    ";
                // }
                // std::cout<<std::endl;
                // std::cout<<"the neuron state of cpu: -------------------------------------"<<std::endl;
                // for(int k=0; k<num_neuron_state_buffer_need; k++){
                //     std::cout<<this->neuron_state_buffer_cpu[k]<<"    ";
                // }
                // std::cout<<std::endl;

                // std::cout<<j<<std::endl;
                for(int k=0; k<num_output_buffer_need; k++){
                    float difference = fabs(this->output_buffer[k] - this->output_buffer_cpu[k]) + fabs(this->neuron_state_buffer[k] - this->neuron_state_buffer_cpu[k]);
                    if(difference>0){
                        std::cout << "ERROR position " << k <<  ": Value ofmap simulator: " << this->output_buffer[k] << ". Value ofmap CPU: " << this->output_buffer_cpu[k] << std::endl;
                        std::cout << "ERROR position " << k <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[k] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[k] << std::endl;
                        std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                        assert(false);
                    }
                }

                //std::cout << "\033[1;32mTest passed correctly \033[0m" << std::endl << std::endl;


                // 计算结束之后，将输出buffer和中间数据buffer中的内容写入DRAM
                // 输出数据个数：num_output_buffer_need
                // 神经元状态数据：num_neuron_state_buffer_need
                // 计算出所需要发送的写请求个数
                int num_output_write_request = std::ceil(num_output_buffer_need / (float)this->dram_instance->dram->GetBusBits());
                int num_neuron_state_write_request = std::ceil(num_neuron_state_buffer_need*(std::ceil(std::log2(this->stonne_cfg.V_th))) / (float)this->dram_instance->dram->GetBusBits());
                int num_total_write_requset = num_output_write_request + num_neuron_state_write_request;
                int num_total_write_request_copy = num_total_write_requset;
                
                // 输出一下总请求个数
                // std::cout<<"num_output_write_request : "<<num_output_write_request<<std::endl;
                // std::cout<<"num_neuron_state_write_request : "<<num_neuron_state_write_request<<std::endl;
                // std::cout<<"num_total_write_requset : "<<num_total_write_requset<<std::endl;
                while(true){  // 发送DRAM内存事务请求
                    if(num_total_write_requset!=0){
                        if(num_output_write_request!=0){
                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(output_offset,true);
                            this->write_request_fifo->push(write_request);
                            output_offset += addr_offset;
                            num_output_write_request--;
                        }else if(num_neuron_state_write_request!=0){
                            std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(output_offset,true);
                            this->write_request_fifo->push(write_request);
                            neuron_state_offset += addr_offset;
                            num_neuron_state_write_request--;
                        }
                        num_total_write_requset--;
                    }
                    //std::cout<<"debug"<<std::endl;
                    if(completed_writes == num_total_write_request_copy){ // 请求响应完之后，将片上buffer数据写入DRAM

                        for(int k=0; k<num_output_buffer_need; k++){
                            this->ofmap[num_output_written+k] = this->output_buffer[k];
                        }

                        for(int k=0; k<num_neuron_state_buffer_need; k++){
                            this->nfmap[num_neuron_state_written+k] = this->neuron_state_buffer[k];
                        }

                        num_output_written += num_output_buffer_need;
                        num_neuron_state_written += num_neuron_state_buffer_need;

                        output_offset = output_offset_copy + num_output_written / 8;
                        neuron_state_offset = neuron_state_offset_copy + num_neuron_state_written*(std::ceil(std::log2(this->stonne_cfg.V_th))) / 8;

                        completed_writes = 0;
                        //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                        break;
                    }

                    this->dram_instance->run();
                    this->n_cycles++;
                }
                delete stonne_instance;
                delete[] this->weight_buffer;            
                delete[] this->output_buffer;
                delete[] this->neuron_state_buffer;
                delete[] this->output_buffer_cpu;
                delete[] this->neuron_state_buffer_cpu;
            }

            // 计算完毕，验证写到DRAM的结果是否正确
            matrixMultiply(output_layer, input_layer, 1, this->filter, this->ifmap, this->ofmap_cpu, this->nfmap_cpu, this->stonne_cfg.V_th, this->pooling_enabled);

            for(int j=0; j<output_layer; j++){
                float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]) + fabs(this->nfmap[j]-this->nfmap_cpu[j]);
                if(difference>0){
                    std::cout<<"ERROR position : "<<j<<"  value ofmap simlator : "<<this->ofmap[j]<<"  value ofmap cpu : "<<this->ofmap_cpu[j]<<std::endl;
                    std::cout<<"ERROR position : "<<j<<"  value nfmap simlator : "<<this->nfmap[j]<<"  value nfmap cpu : "<<this->nfmap_cpu[j]<<std::endl;
                    assert(false);
                }
            }

            std::cout << "\033[1;32mTest passed correctly ****************** \033[0m" << std::endl << std::endl;

            delete[] this->input_buffer;

        } else if(layers[i].type == "pooling"){
            // 池化层
            this->stonne_cfg.layer_type = POOL;
            this->stonne_cfg.V_th = 1;  // 最大池化的阈值是1
            this->n_pooling++;
            this->layer_name = "pooling"+std::to_string(this->n_pooling);

            // 计算输出特征图维度，只支持步长为2的池化
            unsigned int X_ = (this->layers[i].X + 2*this->layers[i].P - this->layers[i].R) / this->layers[i].stride + 1;
            unsigned int Y_ = (this->layers[i].Y + 2*this->layers[i].P - this->layers[i].S) / this->layers[i].stride + 1;

            // 建模存储真实的数据
            // 池化层没有权重数据，也没有神经元状态数据
            unsigned int ifmap_size = this->layers[i].X * this->layers[i].Y * this->layers[i].C;
            unsigned int ofmap_size = X_ * Y_ * this->layers[i].K;
            this->ifmap = new int[ifmap_size];
            this->ofmap = new int[ofmap_size]();
            this->ofmap_cpu = new int[ofmap_size]();

            for(int j=0; j<ifmap_size; j++){
                this->ifmap[j] = rand()%2;
            }

            // std::cout<<"Below are the input feature map data for each channel : "<<std::endl;
            // // for(int num=0; num<ifmap_size; num++){
            // //     std::cout<<this->ifmap[num]<<"  ";
            // // }

            // std::cout<<std::endl;
            // for(int c=0; c<this->layers[i].C; c++){  // 遍历每个通道，输出每个通道的数据
            //     std::cout<<"channel : "<<c<<std::endl;
            //     for(int m=0; m<this->layers[i].X; m++){
            //         for(int n=0; n<this->layers[i].Y; n++){
            //             int index = c + this->layers[i].C * (m*this->layers[i].Y+n);
            //             std::cout<<this->ifmap[index]<<"  ";
            //         }
            //         std::cout<<std::endl;
            //     }
            // }

            // 开始仿真
            std::cout<<"\033[1m\033[33m Start simulation layer :  \033[0m"<<this->layer_name<<std::endl;

            // 例化dram
            this->dram_instance = new Dram(read_callback, write_callback);
            this->dram_instance->set_read_request_fifo(this->read_request_fifo);
            this->dram_instance->set_write_request_fifo(this->write_request_fifo);

            // 各类数据在DRAM中的起始位置
            // 在DRAM中依次存放：输入、权重、输出、神经元状态
            uint64_t input_offset = 0x0000;  // 输入数据在DRAM中的起始地址
            uint64_t output_offset = (this->input_dram_size + this->weight_dram_size)*1024*1024;  // 输出数据在DRAM中的起始地址
            uint64_t addr_offset = 8;  // 连续读取时，每次地址加8，因为一次可以读出64bit数据

            // 实例化片上输入buffer
            unsigned int num_input_buffer_need = this->layers[i].R * this->layers[i].Y * this->layers[i].C;
            assert(num_input_buffer_need <= num_input);
            this->input_buffer = new int[num_input_buffer_need];

            //int num_input_buffer_fold = this->layers[i].X / 2; // 输入数据需要从片外取多少次
            int num_input_buffer_fold = X_; // 考虑池化窗口不是2*2的情况
            int num_input_arranged_fold = std::ceil(Y_ / (float) this->stonne_cfg.m_MSNetworkCfg.ms_rows); // 片上input buffer一个平面需要复用脉动阵列的次数
            int num_input_channel_fold = std::ceil(this->layers[i].C / (float) this->stonne_cfg.m_MSNetworkCfg.ms_cols); // 输入通道需要复用脉动阵列的次数

            int num_input_obtained = 0;

            for(int j=0; j<num_input_buffer_fold; j++){
                // 第一次取R*Y*C个数据
                // 后续每次取stride*Y*C个数据
                int num_input_data;
                if(j==0){
                    num_input_data = this->layers[i].R * this->layers[i].Y * this->layers[i].C;
                } else{
                    num_input_data = this->layers[i].stride * this->layers[i].Y * this->layers[i].C;
                }
                int num_input_read_request = std::ceil(num_input_data / (float)dram_instance->dram->GetBusBits());
                int num_input_read_request_copy = num_input_read_request;

                while(true){
                    
                    if(num_input_read_request!=0){
                        std::shared_ptr<RequestPackage> read_request = std::make_shared<RequestPackage>(input_offset,false);  // （地址，事件类型）false表示读请求，true表示写请求
                        this->read_request_fifo->push(read_request); // 将请求推入fifo
                        input_offset += addr_offset;  // 下一个读取地址
                        num_input_read_request--;
                    }

                    if(completed_reads == num_input_read_request_copy){

                        // 模拟将DRAM中的数据写入片上buffer
                        if(j==0){
                            for(int k=0; k<num_input_data; k++){
                                this->input_buffer[k] = this->ifmap[num_input_obtained+k];
                            }
                        } else {
                            // 移位
                            for(int k=0; k<(this->layers[i].R-this->layers[i].stride)*this->layers[i].C*this->layers[i].Y; k++){
                                this->input_buffer[k] = this->input_buffer[k + this->layers[i].stride*this->layers[i].C*this->layers[i].Y];
                            }
                            // 取数据
                            for(int k=0; k<this->layers[i].stride*this->layers[i].C*this->layers[i].Y; k++){
                                this->input_buffer[k + (this->layers[i].R-this->layers[i].stride)*this->layers[i].C*this->layers[i].Y] = this->ifmap[num_input_obtained+k];
                                
                            }
                        }

                        num_input_obtained += num_input_data;
                        input_offset = num_input_obtained / 8;

                        completed_reads = 0;
                        //std::cout<<"All read requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                        break;
                    }

                    this->dram_instance->run();
                    this->n_cycles++;
                }

                // std::cout<<"Below are the input buffer data : "<<std::endl;
                // for(int num=0; num<ifmap_size; num++){
                //     std::cout<<this->ifmap[num]<<"  ";
                // }
                // std::cout<<std::endl;
                // for(int num=0; num<num_input_buffer_need; num++){
                //     std::cout<<this->input_buffer[num]<<"  ";
                // }
                // std::cout<<std::endl;
                // std::cout<<"Below are the input buffer data for each channel : "<<std::endl;
                // for(int c=0; c<this->layers[i].C; c++){
                //     std::cout<<"channel : "<<c<<std::endl;
                //     for(int m=0; m<this->layers[i].R; m++){
                //         for(int n=0; n<this->layers[i].Y; n++){
                //             int index = c + this->layers[i].C * (m*this->layers[i].Y +n);
                //             std::cout<<this->input_buffer[index]<<"  ";
                //         }
                //         std::cout<<std::endl;
                //     }
                // }

                // DRAM中的数据取到片上之后W，开始将buffer中的数据排列成满足脉动阵列计算的格式
                for(int k=0; k<num_input_channel_fold; k++){
                    int start_channel = k * this->stonne_cfg.m_MSNetworkCfg.ms_cols;
                    int end_channel = std::min<int>(start_channel+this->stonne_cfg.m_MSNetworkCfg.ms_cols, this->layers[i].C);
                    int channels = end_channel-start_channel;

                    // 生成用于卷积操作的权重值
                    this->weight_buffer = new int[channels * this->layers[i].R * this->layers[i].S * channels]();
                    for(int m=0; m<channels; m++){
                        for(int n=0; n<this->layers[i].R*this->layers[i].S; n++){
                            this->weight_buffer[m*channels*this->layers[i].R*this->layers[i].S + m*this->layers[i].R*this->layers[i].S + n] = 1;
                        }
                    }

                    // std::cout<<"---------------weight----------------------"<<std::endl;
                    // for(int m=0; m<channels; m++){
                    //     for(int n=0; n<channels*this->layers[i].R*this->layers[i].S; n++){
                    //         std::cout<<this->weight_buffer[m*channels*this->layers[i].R*this->layers[i].S + n];
                    //     }
                    //     std::cout<<std::endl;
                    // }

                    this->bankSize = channels*this->layers[i].R*this->layers[i].S; 
                    std::vector<int> im2col_bank(this->bankSize,0);

                    for(int n=0; n<num_input_arranged_fold; n++){
                        //std::cout<<" ========================== num_input_arranged_fold :  "<<n<<std::endl;
                        int start_rows = n * this->stonne_cfg.m_MSNetworkCfg.ms_rows;
                        int end_rows = std::min<int>(start_rows+this->stonne_cfg.m_MSNetworkCfg.ms_rows, Y_);
                        int rows = end_rows - start_rows;

                        // 根据需要例化片上输出buffer
                        unsigned int num_output_buffer_need = rows * channels;
                        unsigned int num_neuron_buffer_need = rows * channels;
                        assert(num_neuron_buffer_need <= num_neuron_state); 
                        assert(num_output_buffer_need <= num_output);
                        this->output_buffer = new int[num_output_buffer_need]();
                        this->output_buffer_cpu = new int[num_output_buffer_need](); 
                        this->neuron_state_buffer = new int[num_neuron_buffer_need](); // 无计算需求
                        this->neuron_state_buffer_cpu = new int[num_neuron_buffer_need]();

                        if(n==0){  // 第一块单独考虑，因为基地址计算不同
                            for(int m=0; m<rows; m++){  // 取出的每一行中有channels个R*S大小的窗口
                                if(m==0){  // 取出 R * S * channels 个数据
                                    for(int c=0; c<channels; c++){  
                                        for(int p=0; p<this->layers[i].R; p++){
                                            int base_addr = (p * this->layers[i].Y) * this->layers[i].C + (k * this->stonne_cfg.m_MSNetworkCfg.ms_cols) + c;
                                            //std::cout<<"base_addr : "<<base_addr<<std::endl;
                                            for(int q=0; q<this->layers[i].S; q++){
                                                im2col_bank[(c * this->layers[i].R *this->layers[i].S) + (p*this->layers[i].S) + q] = this->input_buffer[base_addr + q*this->layers[i].C];
                                                //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr + q*this->layers[i].C)<<"    to   im2col_bank location : "<<((c * this->layers[i].R *this->layers[i].S) + (p*this->layers[i].S) + q)<<std::endl;
                                                this->n_cycles++;
                                            }
                                        }
                                    }

                                    // std::cout<<"The sorted data is : "<<std::endl;
                                    // for(int p=0; p<this->bankSize; p++){
                                    //     std::cout<<im2col_bank[p]<<"   ";
                                    // }
                                    // std::cout<<std::endl;
                            
                                    this->input_arranged.push_back(im2col_bank);

                                } else {   // 取出 R * strides * channels 个数据
                                    // 每个通道内，取（stride*R）个数据，im2col内移动（（S-stride）*R）个数据
                                    for(int c=0; c<channels; c++){
                                        for(int p=0; p<this->layers[i].R; p++){
                                            int base_addr = (p*this->layers[i].Y + this->layers[i].S + (m-1)*this->layers[i].stride) * this->layers[i].C + (k * this->stonne_cfg.m_MSNetworkCfg.ms_cols) + c;
                                            //std::cout<<"base_addr : "<<base_addr<<std::endl;

                                            for(int q=0; q<this->layers[i].stride; q++){
                                                if(q==0){
                                                    // 移位
                                                    for(int num=0; num<(this->layers[i].S-this->layers[i].stride); num++){
                                                        im2col_bank[(c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + num] = im2col_bank[(c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + num + this->layers[i].stride];
                                                        //std::cout<<"Data movement from im2col in location : "<<((c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + num + this->layers[i].stride)<<"  to  : "<<((c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + num)<<std::endl;
                                                    }
                                                    
                                                    // 取数据
                                                    im2col_bank[(c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + (this->layers[i].S-this->layers[i].stride) + q] = this->input_buffer[base_addr + q*this->layers[i].C];
                                                    //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr + q*this->layers[i].C)<<"    to   im2col_bank location : "<<((c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + (this->layers[i].S-this->layers[i].stride) + q)<<std::endl;
                                                    this->n_cycles++;
                                                } else{
                                                    im2col_bank[(c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + (this->layers[i].S-this->layers[i].stride) + q] = this->input_buffer[base_addr + q*this->layers[i].C];
                                                    //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr + q*this->layers[i].C)<<"    to   im2col_bank location : "<<((c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + (this->layers[i].S-this->layers[i].stride) + q)<<std::endl;
                                                    this->n_cycles++;
                                                }

                                            }
                                        }
                                    }

                                    // std::cout<<"The sorted data is : "<<std::endl;
                                    // for(int p=0; p<this->bankSize; p++){
                                    //     std::cout<<im2col_bank[p]<<"   ";
                                    // }
                                    // std::cout<<std::endl;
                            
                                    this->input_arranged.push_back(im2col_bank);
                                }
                            }
                        } else {  // 每次取出R * strides * channels 个数据，移位R * (S-strides) * channels 个数据
                            for(int m=0; m<rows; m++){
                                for(int c=0; c<channels; c++){
                                    for(int p=0; p<this->layers[i].R; p++){
                                        int base_addr = (p*this->layers[i].Y + (this->layers[i].S + n*this->stonne_cfg.m_MSNetworkCfg.ms_rows*this->layers[i].stride - this->layers[i].stride) + m*this->layers[i].stride) * this->layers[i].C + k*this->stonne_cfg.m_MSNetworkCfg.ms_cols + c;
                                        //std::cout<<"base_addr : "<<base_addr<<std::endl;

                                        for(int q=0; q<this->layers[i].stride; q++){
                                            if(q==0){
                                                // 移位 
                                                for(int num=0; num<(this->layers[i].S-this->layers[i].stride); num++){
                                                    im2col_bank[(c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + num] = im2col_bank[(c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + num + this->layers[i].stride];
                                                    //std::cout<<"Data movement from im2col in location : "<<((c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + num + this->layers[i].stride)<<"  to  : "<<((c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + num)<<std::endl;
                                                }
                                                
                                                // 取数据
                                                im2col_bank[(c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + (this->layers[i].S-this->layers[i].stride) + q] = this->input_buffer[base_addr + q*this->layers[i].C];
                                                //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr + q*this->layers[i].C)<<"    to   im2col_bank location : "<<((c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + (this->layers[i].S-this->layers[i].stride) + q)<<std::endl;
                                                this->n_cycles++;
                                            } else {
                                                // 取数据
                                                im2col_bank[(c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S +(this->layers[i].S-this->layers[i].stride) + q] = this->input_buffer[base_addr + q*this->layers[i].C];
                                                //std::cout<<"Data fetching process(in input_buffer location) : "<<(base_addr + q*this->layers[i].C)<<"    to   im2col_bank location : "<<((c*this->layers[i].R*this->layers[i].S) + p*this->layers[i].S + (this->layers[i].S-this->layers[i].stride) + q)<<std::endl;
                                                this->n_cycles++;
                                            }
                                        }
                                    }
                                }

                                // std::cout<<"The sorted data is : "<<std::endl;
                                // for(int p=0; p<this->bankSize; p++){
                                //     std::cout<<im2col_bank[p]<<"   ";
                                // }
                                // std::cout<<std::endl;
                        
                                this->input_arranged.push_back(im2col_bank);

                            }
                        }

                        int total = rows * this->bankSize;
                        this->spikes = new int[total];

                        int index = 0;
                        for(const auto& row : input_arranged){
                            for(int val : row){
                                this->spikes[index++] = val;
                            }
                        }
                        

                        // 输入数据：this->spikes
                        // 权重数据：this->weight_buffer

                        Stonne* stonne_instance = new Stonne(stonne_cfg);
                        matrixMultiply(rows, this->bankSize, channels, this->spikes, this->weight_buffer, this->output_buffer_cpu, this->neuron_state_buffer_cpu, 1, false);
                        //sequential_layer(1,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,rows,1,1,cols,this->layers[i].R * this->layers[i].S * this->layers[i].C,1,this->spikes,this->weight_buffer,this->output_buffer_cpu,this->neuron_state_buffer_cpu,this->stonne_cfg.V_th,this->Timestamp,this->pooling_enabled);
                        stonne_instance->loadDenseGEMM(layer_name,channels,this->bankSize,rows,this->spikes,this->weight_buffer,this->output_buffer,this->neuron_state_buffer,CNN_DATAFLOW);
                        stonne_instance->loadGEMMTile(channels,1,rows);
                        stonne_instance->run();

                        this->n_cycles += stonne_instance->n_cycles;
                        // std::cout<<"stonne_instance->n_cycles : "<<stonne_instance->n_cycles<<std::endl;
                        // std::cout<<"total cycles : "<<this->n_cycles<<std::endl;

                        for(int m = 0; m<num_output_buffer_need; m++){
                            float difference = fabs(this->output_buffer[m]-this->output_buffer_cpu[m]) + fabs(this->neuron_state_buffer[m]-this->neuron_state_buffer_cpu[m]);
                            if(difference>0){
                                std::cout << "ERROR position " << m <<  ": Value ofmap simulator: " << this->output_buffer[m] << ". Value ofmap CPU: " << this->output_buffer_cpu[m] << std::endl;
                                std::cout << "ERROR position " << m <<  ": Value neuron_state simulator: " << this->neuron_state_buffer[m] << ". Value neuron_state CPU: " << this->neuron_state_buffer_cpu[m] << std::endl;
                                std::cout << "\033[1;31mT test not passed\033[0m" << std::endl;
                                // std::cout<<"j : "<<j<<"   num_weight_buffer_fold : "<<num_weight_buffer_fold<<std::endl;
                                // std::cout<<"k : "<<k<<"   num_input_buffer_fold  "<<num_input_buffer_fold<<std::endl;
                                assert(false);
                            }
                        }
                        //std::cout << "\033[1;32mTest passed correctly\033[0m" << std::endl << std::endl;

                        // std::cout<<"input data ---------------------: "<<std::endl;
                        // for(int m=0;m<rows;m++){
                        //     for(int a=0;a<this->layers[i].R * this->layers[i].S * channels; a++){
                        //         std::cout<<this->spikes[m*this->layers[i].R * this->layers[i].S * channels + a]<<"   ";
                        //     }
                        //     std::cout<<std::endl;
                        // }

                        // std::cout<<"the output of sim : ----------------------------"<<std::endl;
                        // for(int n=0; n<rows; n++){
                        //     for(int p=0; p<channels; p++){
                        //         std::cout<<this->output_buffer[n*channels+p]<<" ";
                        //     }
                        //     std::cout<<std::endl;
                        // }

                        // std::cout<<"the output of cpu : ----------------------------"<<std::endl;
                        // for(int n=0; n<rows; n++){
                        //     for(int p=0; p<channels; p++){
                        //         std::cout<<this->output_buffer_cpu[n*channels+p]<<" ";
                        //     }
                        //     std::cout<<std::endl;
                        // }

                        // 将计算结果写入DRAM
                        // 计算结果的维度：rows行，channels列，逐行将输出结果写入DRAM，行与行之后的地址不一定连续
                        int num_output_write_request = 0;
                        bool only = true;
                        while(true){
                            while(only){
                                for(int r=0; r<rows; r++){
                                    int current_num_output_write_request = std::ceil(channels / (float)dram_instance->dram->GetBusBits());
                                    num_output_write_request += current_num_output_write_request;

                                    int write_addr = output_offset + ((j*Y_ + n*this->stonne_cfg.m_MSNetworkCfg.ms_rows + r)*this->layers[i].C + k*this->stonne_cfg.m_MSNetworkCfg.ms_cols)/8;
                                    while(current_num_output_write_request != 0){
                                        std::shared_ptr<RequestPackage> write_request = std::make_shared<RequestPackage>(write_addr,true);
                                        this->write_request_fifo->push(write_request);
                                        write_addr += addr_offset;
                                        current_num_output_write_request--;

                                        this->dram_instance->run();
                                        this->n_cycles++; // 一个周期发送一个内存请求事务
                                    }

                                    // 将该行数据写入DRAM
                                    for(int p=0; p<channels; p++){
                                        this->ofmap[(j*Y_ + n*this->stonne_cfg.m_MSNetworkCfg.ms_rows + r)*this->layers[i].C + k*this->stonne_cfg.m_MSNetworkCfg.ms_cols +p] = this->output_buffer[r*channels+p];
                                    }

                                }

                                only = false;
                            }

                            if(completed_writes == num_output_write_request){
                                completed_writes = 0;
                                //std::cout<<"All write requests are completed in clock cycle: "<<std::dec<<this->n_cycles<<std::endl;
                                break;
                            }

                            this->dram_instance->run();
                            this->n_cycles++;
                        }

                        delete stonne_instance;
                        this->input_arranged.clear();
                        delete[] this->spikes;
                        delete[] this->output_buffer;
                        delete[] this->output_buffer_cpu;
                        delete[] this->neuron_state_buffer;
                        delete[] this->neuron_state_buffer_cpu;
                    }

                    delete[] this->weight_buffer;

                }

            }

            // 计算完毕，验证写到DRAM的结果是否正确
            pooling_compute(this->layers[i].X, this->layers[i].Y, this->layers[i].C, this->layers[i].R, this->layers[i].S, this->layers[i].stride, this->ifmap, this->ofmap_cpu);
            // std::cout<<"debug"<<std::endl;

            for(int j=0; j<X_*Y_*this->layers[i].C; j++){
                float difference = fabs(this->ofmap[j]-this->ofmap_cpu[j]);
                if(difference>0){
                    std::cout<<"ERROR position : "<<j<<"  value ofmap simlator : "<<this->ofmap[j]<<"  value ofmap cpu : "<<this->ofmap_cpu[j]<<std::endl;
                    assert(false);
                }
            }

            std::cout << "\033[1;32mTest passed correctly \033[0m" << std::endl << std::endl;

        } else {
            std::cout<<"Unsupported layer types!"<<std::endl;
            assert(false);
        }
            
        delete[] this->ifmap;
        //delete[] this->filter;
        delete[] this->ofmap;
        delete[] this->ofmap_cpu;
        //delete[] this->nfmap;
        // this->MK_matrix = new int[this->output_size];
        // std::memcpy(this->MK_matrix, this->output, this->output_size*sizeof(int));
        // delete[] this->input_buffer;
        // delete[] this->weight_buffer;
        // delete[] this->output_buffer;
        // delete[] this->output_buffer_cpu;
        // delete[] this->neuron_state_buffer;
        // delete[] this->neuron_state_buffer_cpu;

        if(this->layers[i].type != "pooling"){
            delete[] this->filter;
            delete[] this->nfmap;
            delete[] this->nfmap_cpu;
        }

        delete this->dram_instance;

        std::cout<<"The number of cycles of this layer runing : "<<this->n_cycles<<std::endl;
        // std::cout << "\033[1;32mSimulation completed \033[0m" << std::endl;
        std::cout<<std::endl;
    }

    // std::cout << std::endl;
    std::cout <<"============================ End simulation ===================================="<<std::endl;
    // std::cout << std::endl;
    std::cout << "Number of cycles running: " << this->n_cycles << std::endl;
    // std::cout << "Time mem: " << time_mem/100000 << std::endl;
    // std::cout << "Time pooling: "<<time_pooling/100000 <<std::endl;
    // std::cout << "Time update: " << time_update/100000 <<std::endl;
    // std::cout << "Time as: " << time_as/100000 << std::endl;
    // std::cout << "Time ms: " << time_ms/100000 << std::endl;

    // // If the code does not stop then the TEST is correct
    // std::cout<<std::endl;
    // std::cout << "\033[1;32mTest passed correctly \033[0m" << std::endl << std::endl;

}