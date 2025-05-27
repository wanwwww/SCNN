
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

    // // DRAM中各类参数存储的空间
    // this->input_dram_size = this->stonne_cfg.m_DRAMCfg.input_dram_size;
    // this->weight_dram_size = this->stonne_cfg.m_DRAMCfg.weight_dram_size;
    // this->output_dram_size = this->stonne_cfg.m_DRAMCfg.output_dram_size;

    this->weight_width = stonne_cfg.weight_width;
    this->max_weight = stonne_cfg.max_weight;
    this->min_weight = stonne_cfg.min_weight;

    // DRAM中一次存放：输入、权重、输出和神经元状态
    this->input_offset = 0x0000; // 输入数据在DRAM中的起始地址 （单位是字节）
    this->weight_offset = 2*1024*1024;
    this->output_offset = (2 + 3) * 1024 *1024;
    this->neuron_state_offset = (2 + 3 + 2) * 1024 * 1024;
    this->addr_offset = 8; // 一次可以从DRAM中读取8个字节的数据

    // 实例化数组当作片上buffer
    this->input_buffer_size = this->stonne_cfg.m_BufferCfg.input_buffer_size;
    this->weight_buffer_size = this->stonne_cfg.m_BufferCfg.weight_buffer_size;
    this->output_buffer_size = this->stonne_cfg.m_BufferCfg.output_buffer_size;
    this->neuron_state_buffer_size = this->stonne_cfg.m_BufferCfg.neuron_state_buffer_size;

    // std::cout<<"weight_buffer_size : "<<this->weight_buffer_size<<std::endl;

    // 片上buffer能够存储的各种数据的个数，但实际不需要存储这么多
    this->num_input = this->input_buffer_size*1024*8;  // 存储输入脉冲的个数
    this->num_weight = this->weight_buffer_size*1024*8/this->weight_width;  // 存储权重的个数
    this->num_output = this->output_buffer_size*1024*8;  // 存储输入脉冲的个数
    this->num_neuron_state = this->neuron_state_buffer_size*1024*8/(std::ceil(std::log2(this->stonne_cfg.V_th)));  // 存储膜电位的个数

    // std::cout<<"this->num_weight : "<<this->num_weight<<std::endl;

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

// 在process_conv_2的基础上，进一步将input_arranged_buffer设置为双buffer
int Controller::process_conv_3(int layer_id, int i, int j, int cols, layer_topology layer_parameters){
    //std::cout<<"call process_conv_3"<<std::endl;
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
    // std::cout<<"call process_conv_and_pooling"<<std::endl;
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

// 输入和权重双buffer，模块化，input_buffer容量有限
std::tuple<int*, int*, int*, int*> Controller::runFC_2(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters) {
    // 全连接层控制逻辑
    // std::cout<<std::endl;
    // std::cout<<"****************************************************************************************************************"<<std::endl;
    // std::cout<<"                                                 Call runFC_2 function                                          "<<std::endl;
    // std::cout<<"****************************************************************************************************************"<<std::endl;
    // std::cout<<std::endl;

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

// input_arranged_buffer 和 weight_buffer 和 input_buffer都设置为双buffer，
// weight_buffer 取下一次计算所需数据的时间是：input_buffer第二次取数据之后（较简单的控制逻辑）
std::tuple<int*, int*, int*, int*> Controller::runConv_DataFlow_3(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    // std::cout<<std::endl;
    // std::cout<<"****************************************************************************************************************"<<std::endl;
    // std::cout<<"                                   Call runConv_DataFlow_3 function                                             "<<std::endl;
    // std::cout<<"****************************************************************************************************************"<<std::endl;
    // std::cout<<std::endl;

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

// 三个双buffer，加padding操作在step1阶段实现
std::tuple<int*, int*, int*, int*> Controller::runConvandPooling_DataFlow_2(int layer_id, int* ifmap, int* filter, int* ofmap, int* nfmap, layer_topology layer_parameters){
    // 卷积层的控制逻辑
    // 考虑是否池化，这里只支持池化为0、1、2，步长为1的卷积

    // std::cout<<std::endl;
    // std::cout<<"****************************************************************************************************************"<<std::endl;
    // std::cout<<"                                  Call runConvandPooling_DataFlow_2 function                                    "<<std::endl;
    // std::cout<<"****************************************************************************************************************"<<std::endl;
    // std::cout<<std::endl;

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
    // std::cout<<"num_weight_buffer_need : "<<num_weight_buffer_need<<std::endl;
    // std::cout<<"this->num_weight : "<<this->num_weight<<std::endl;
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
    std::cout<<"load the first weights cycles : "<<n_cycles_load_first_weight<<"          and current cycles is : "<<this->n_cycles<<std::endl;
    
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
        std::cout<<"load the first input cycles : "<<n_cycles_load_first_input<<"          and current cycles is : "<<this->n_cycles<<std::endl;
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

        // std::cout<<"current weight loop begin cycles : "<<this->n_cycles<<std::endl;
        int count_rows = 0; // 用于累加卷积的输出到片上SRAM计数
        for(int j=0; j<X_; j++){
            std::cout<<"======================================== input loop ========================================= "<<j<<"/"<<X_-1<<std::endl;
            std::cout<<"current input loop begin cycles : "<<this->n_cycles<<"          and current cycles is : "<<this->n_cycles<<std::endl;
            // *** 加载下一块输入数据，与上一块数据的计算并行
            int n_cycles_load_next_input;
            if(j+1<X_){
                n_cycles_load_next_input = load_input_data_step1_ppbuffer(layer_id, ifmap, this->dram_instance, j+1, layer_parameters);
                std::cout<<"load next input cycles : "<<n_cycles_load_next_input<<std::endl;
            } else {
                n_cycles_load_next_input = 0;
                std::cout<<"load next input cycles : "<<n_cycles_load_next_input<<std::endl;
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
                std::cout<<"load next weight cycles : "<<n_cycles_load_next_weight<<std::endl;
            } else {
                n_cycles_load_next_weight = 0;
                std::cout<<"load next weight cycles : "<<n_cycles_load_next_weight<<std::endl;
            }

            // *** 重排和计算
            int n_cycles_process = process_conv_and_pooling(layer_id, i, j, cols, count_rows, layer_parameters);
            count_rows++;
            std::cout<<"process_conv_and_pooling cycles : "<<n_cycles_process<<std::endl;
            //this->n_cycles += n_cycles_process;

            // *** 将计算结果写入DRAM
            // 1. 将神经元状态写入DRAM
            int n_cycles_write_neuron_state = store_neuron_state(nfmap, this->dram_instance, i, j, cols, layer_parameters);
            std::cout<<"write neuron_state cycles : "<<n_cycles_write_neuron_state<<std::endl;
            // this->n_cycles += n_cycles_write_neuron_state;

            // 2. 对累积的结果进行池化，将池化结果写入DRAM
            //int n_cycles_pooling_and_write_output;
            int n_cycles_process_pooling;
            int n_cycles_write_output;
            if(j>0 && j%2!=0){

                //std::cout<<"need pooling "<<std::endl;
                count_rows = 0;
                // 调用池化模块
                n_cycles_process_pooling = process_pooling(i, j, cols, layer_parameters);
                std::cout<<"process_pooling cycles : "<<n_cycles_process_pooling<<std::endl;
                //this->n_cycles += n_cycle_process_pooling;

                // 将池化结果写入DRAM
                n_cycles_write_output = store_output(ofmap, this->dram_instance, i, j, cols, layer_parameters);
                std::cout<<"write output cycles : "<<n_cycles_write_output<<std::endl;
                // std::cout<<"debug"<<std::endl;
                //this->n_cycles += n_cycles_write_output;

                //n_cycles_pooling_and_write_output = n_cycles_process_pooling + n_cycles_write_output;
                //std::cout<<"process_pooling and write output cycles : "<<n_cycles_pooling_and_write_output<<std::endl;
            } else {
                // std::cout<<" without pooling "<<std::endl;
                n_cycles_process_pooling = 0;
                n_cycles_write_output = 0;
                std::cout<<"process_pooling cycles : "<<n_cycles_process_pooling<<std::endl;
                std::cout<<"write output cycles : "<<n_cycles_write_output<<std::endl;
                //n_cycles_pooling_and_write_output = 0;
                //std::cout<<"process_pooling and write output cycles : "<<n_cycles_pooling_and_write_output<<std::endl;
            }

            if(i==num_weight_buffer_fold-1 && j==X_-1){
                // 最后一次计算结束，加上输出写入DRAM所需周期
                // std::cout<<"add write output"<<std::endl;
                this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process+n_cycles_write_neuron_state+n_cycles_process_pooling+n_cycles_write_output);
                // std::cout<<std::endl;
                std::cout<<"Global cycles : "<<this->n_cycles<<std::endl;
            } else {
                // std::cout<<"without write output"<<std::endl;
                this->n_cycles += std::max(n_cycles_load_next_input+n_cycles_load_next_weight, n_cycles_process+n_cycles_process_pooling);
                // std::cout<<std::endl;
                std::cout<<"Global cycles : "<<this->n_cycles<<std::endl;
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
