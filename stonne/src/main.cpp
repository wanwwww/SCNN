#include <iostream>
#include "STONNEModel.h"
#include <chrono>
#include <assert.h>
#include "testbench.h"
#include <string>
#include <math.h>
#include <utility.h>
#include <cstdlib>
#include <filesystem>


#include "Controller.h"
#include "MYPOOL.h"


// 函数声明
std::vector<layer_topology> readCSV_layers(std::string& filename); // 读取网络层参数
std::string get_model_name(const std::string& filepath);

int main(int argc, char *argv[]) {

    std::string layer_topology_path;
    std::string hardware_cfg_path;
    //hardware_cfg h_cfg;

    // 从命令行读取的参数
    if(argc!=3){
        // std::cerr 是用于输出错误信息的输出流。与std::cout类似，但std::cerr通常不经过缓冲区。
        std::cerr<<"Error : please provide topology file path and configuration file path!"<<std::endl;
        return EXIT_FAILURE; // 用于返回一个非零值，表示程序异常终止或运行失败。EXIT_FAILURE是<cstdlib>中定义的宏，一般代表一个非零整数（通常为1）
    } else {
        layer_topology_path = argv[1];
        hardware_cfg_path = argv[2];
        if(!std::filesystem::exists(layer_topology_path)){
            std::cerr<<"Error : the topology file path does not exist!"<<std::endl;
            return EXIT_FAILURE;
        }
        if(!std::filesystem::exists(hardware_cfg_path)){
            std::cerr<<"Error : the configuration file path does not exist!"<<std::endl;
            return EXIT_FAILURE;
        }
    }

     // 例化config，加载硬件参数
    Config stonne_cfg;
    stonne_cfg.loadFile(hardware_cfg_path);
    stonne_cfg.max_weight = (1 << (stonne_cfg.weight_width-1))-1;
    stonne_cfg.min_weight = -(1 << (stonne_cfg.weight_width-1));

    // 网络拓扑结构
    std::vector<layer_topology> layers;
    layers = readCSV_layers(layer_topology_path);

    // 根据网络拓扑，确定DRAM中各类数据存储的起始地址，DRAM中存储各层数据时，采用固定位置的方法
    // DRAM中各层数据存储顺序：输入、权重、输出、膜电位
    // 遍历每一层参数，计算各类数据所需的存储空间

    int input_max_in_byte = 0;
    int weight_max_in_byte = 0;
    int output_max_in_byte = 0;

    int input_num_in_byte = 0; 
    int output_num_in_byte = 0;
    int weight_num_in_byte = 0; 


    for(int i=0; i<layers.size(); i++){
        if(layers[i].type == "fc"){  // 全连接层
            int input_neuron = layers[i].input_neuron;  
            int output_neuron = layers[i].output_neuron;  
            
            input_num_in_byte = static_cast<int>(std::ceil(input_neuron / (float)8));  // 向上取整
            output_num_in_byte = static_cast<int>(std::ceil(output_neuron / (float)8));
            weight_num_in_byte = static_cast<int>(std::ceil(input_neuron*output_neuron*stonne_cfg.weight_width / (float)8)); 

        } else {  // 卷积层
            int R = layers[i].R;
            int S = layers[i].S;
            int C = layers[i].C;
            int K = layers[i].K;
            int X = layers[i].X;
            int Y = layers[i].Y;
            int P = layers[i].P;
            int stride = layers[i].stride;
            
            int X_ = (X + 2*P - R)/stride + 1;
            int Y_ = (Y + 2*P - S)/stride + 1;

            if(layers[i].type == "conv_pooling"){  // 卷积层后跟池化
                X_ = X_/2;
                Y_ = Y_/2;
            } 

            input_num_in_byte = static_cast<int>(std::ceil(X*Y*C / (float)8));
            output_num_in_byte = static_cast<int>(std::ceil(X_*Y_*K / (float)8));
            weight_num_in_byte = static_cast<int>(std::ceil(K*R*S*C*stonne_cfg.weight_width / (float)8));
        }

        if(input_num_in_byte >= input_max_in_byte){
            input_max_in_byte = input_num_in_byte;
        }

        if(output_num_in_byte >= output_max_in_byte){
            output_max_in_byte = output_num_in_byte;
        }

        if(weight_num_in_byte >= weight_max_in_byte){
            weight_max_in_byte = weight_num_in_byte;
        }
    }

    // 设置DRAM参数
    stonne_cfg.m_DRAMCfg.input_offset = 0;
    stonne_cfg.m_DRAMCfg.weight_offset = input_max_in_byte;
    stonne_cfg.m_DRAMCfg.output_offset = input_max_in_byte + weight_max_in_byte;
    stonne_cfg.m_DRAMCfg.neuron_state_offset = input_max_in_byte + weight_max_in_byte + output_max_in_byte;
    // std::cout<<"input_offset : "<<stonne_cfg.m_DRAMCfg.input_offset<<std::endl;
    // std::cout<<"weight_offset : "<<stonne_cfg.m_DRAMCfg.weight_offset<<std::endl;
    // std::cout<<"output_offset : "<<stonne_cfg.m_DRAMCfg.output_offset<<std::endl;
    // std::cout<<"neuron_state_offset : "<<stonne_cfg.m_DRAMCfg.neuron_state_offset<<std::endl;

    // h_cfg = readCSV_cfg(hardware_cfg_path);
    // std::cout<<" the layers topo "<<std::endl;
    // for(int i=0;i<layers.size();i++){
    //     std::cout<<layers[i].type<<" ";
    //     std::cout<<layers[i].R<<" ";
    //     std::cout<<layers[i].S<<" ";
    //     std::cout<<layers[i].C <<" ";
    //     std::cout<<layers[i].K <<" ";
    //     std::cout<<layers[i].X <<" ";
    //     std::cout<<layers[i].Y <<" ";
    //     std::cout<<layers[i].P <<" ";
    //     std::cout<<layers[i].stride<<" ";
    //     std::cout<<layers[i].pooling_size<<" ";
    //     std::cout<<layers[i].pooling_stride<<" "; 
    //     std::cout<<layers[i].input_neuron<<" ";
    //     std::cout<<layers[i].output_neuron<<" ";
    //     std::cout<<layers[i].batch<<std::endl;
    //     std::cout<<std::endl;
    // }

    int* ifmap;
    int* filter;
    int* ofmap;
    int* nfmap;

    // 实例化控制器
    Controller* control = new Controller(stonne_cfg, layers);
    //std::tie(ifmap, filter, ofmap, nfmap) = control->runConvandPooling_DataFlow(1,ifmap,filter,ofmap,nfmap,layers[0]);
    //control->run();

    //std::cout<<"begin cycles : "<<control->n_cycles<<std::endl;

    for(int i=0; i<layers.size(); i++){

        int layer_id = i+1;

        if(i==0){  // 对于第一层，输入是输入，输出是输出
            if(layers[i].type == "conv_pooling") {
                std::tie(ifmap, filter, ofmap, nfmap) = control->runConvandPooling_DataFlow_2(layer_id, ifmap, filter, ofmap, nfmap, layers[i]);
                delete[] ifmap;
                delete[] filter;
                delete[] nfmap;
            } else if(layers[i].type == "conv") {
                std::tie(ifmap, filter, ofmap, nfmap) = control->runConv_DataFlow_3(layer_id, ifmap, filter, ofmap, nfmap, layers[i]);
                delete[] ifmap;
                delete[] filter;
                delete[] nfmap;
            } else if(layers[i].type == "fc") {
                std::tie(ifmap, filter, ofmap, nfmap) = control->runFC_2(layer_id, ifmap, filter, ofmap, nfmap, layers[i]);
                delete[] ifmap;
                delete[] filter;
                delete[] nfmap;
            } else {
                std::cout<<"\033[1;31m"<<"Unsupported layer types !"<<"\033[0m"<<std::endl;
                assert(false);
            }
        } else {  // 对于后面的层，输入是上一层的输出
            if(layers[i].type == "conv_pooling") {
                std::tie(ifmap, filter, ofmap, nfmap) = control->runConvandPooling_DataFlow_2(layer_id, ofmap, filter, ifmap, nfmap, layers[i]);
                delete[] ifmap;
                delete[] filter;
                delete[] nfmap;
            } else if(layers[i].type == "conv") {
                std::tie(ifmap, filter, ofmap, nfmap) = control->runConv_DataFlow_3(layer_id, ofmap, filter, ifmap, nfmap, layers[i]);
                delete[] ifmap;
                delete[] filter;
                delete[] nfmap;
            } else if(layers[i].type == "fc") {
                std::tie(ifmap, filter, ofmap, nfmap) = control->runFC_2(layer_id, ofmap, filter, ifmap, nfmap, layers[i]);
                delete[] ifmap;
                delete[] filter;
                delete[] nfmap;
            } else {
                std::cout<<"\033[1;31m"<<"Unsupported layer types !"<<"\033[0m"<<std::endl;
                assert(false);
            }
        }
    }

    std::string model_name = get_model_name(layer_topology_path);

    std::cout<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<"      The "<<model_name<<" simulation ends, and the total number of running cycles is: "<<control->n_cycles<<std::endl;
    std::cout<<"****************************************************************************************************************"<<std::endl;
    std::cout<<std::endl;
    
    delete control;

    // // 测试池化模块代码=============================================================================
    // // 输入数组：channels * 2 * Y_
    // // 输出数组：channels*Y_/2
    // int channels = 2;
    // int Y_ = 28;
    // int* on_chip_sram = new int[channels*2*Y_];
    // int* output_regs = new int[channels*Y_/2]();
    // int* output_regs_cpu = new int[channels*Y_/2]();

    // // 生成随机脉冲
    // for(int i=0; i<channels*2*Y_; i++){
    //     on_chip_sram[i] = rand()%2;
    // }

    // //std::cout<<"begin sim : "<<std::endl;
    // MYPOOL* pooling_instance = new MYPOOL(stonne_cfg);
    // //std::cout<<"debug1"<<std::endl;
    // pooling_instance->loadPOOLLayer(Y_, channels, on_chip_sram, output_regs);
    // //std::cout<<"debug2"<<std::endl;
    // pooling_instance->run();
    // //std::cout<<"debug3"<<std::endl;


    // // 输出
    // std::cout<<"cycles : "<<pooling_instance->n_cycle<<std::endl;
    // std::cout<<"------------- input ----------------------"<<std::endl;
    // for(int i=0; i<channels; i++){
    //     std::cout<<"channels : "<<i<<std::endl;
    //     for(int j=0; j<2; j++){
    //         for(int k=0; k<Y_; k++){
    //             std::cout<<on_chip_sram[i*2*Y_ + j*Y_ + k]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    // }

    // std::cout<<"------------- sim output ----------------------"<<std::endl;
    // for(int i=0; i<channels; i++){
    //     std::cout<<"channels : "<<i<<std::endl;
    //     for(int j=0; j<Y_/2; j++){
    //         std::cout<<output_regs[i*Y_/2 + j]<<"  ";
    //     }
    //     std::cout<<std::endl;
    // }

    // pool2x2(on_chip_sram, output_regs_cpu, Y_, channels);

    // std::cout<<"------------- cpu output ----------------------"<<std::endl;
    // for(int i=0; i<channels; i++){
    //     std::cout<<"channels : "<<i<<std::endl;
    //     for(int j=0; j<Y_/2; j++){
    //         std::cout<<output_regs_cpu[i*Y_/2 + j]<<"  ";
    //     }
    //     std::cout<<std::endl;
    // }

    // // 对比模拟器和CPU的
    // for(int i=0; i<channels*Y_/2; i++){
    //     float difference = fabs(output_regs[i] - output_regs_cpu[i]);
    //     if(difference>0){
    //         std::cout<<"error location : "<<i<<std::endl;
    //         assert(false);
    //     }
    // }

}

std::string get_model_name(const std::string& filepath) {
    std::filesystem::path p(filepath);
    std::string filename = p.stem().string();  // 去除扩展名后的文件名
    return filename;
}

// 读取csv文件函数
std::vector<layer_topology> readCSV_layers(std::string& filename){

    std::vector<layer_topology> layers;
    std::ifstream file(filename); // 打开csv文件

    if(!file){
        std::cout<<"failed to open file"<<std::endl;
        assert(1==0);
    }

    std::string line;
    std::getline(file,line); // 从文件中取出一行，这里是跳过CSV文件的首行

    while(std::getline(file,line)){
        std::stringstream ss(line);  // 将line转化为std::stringstream，可以像操作流一样逐个提取数据
        std::string type; // 存储层类型
        int R;
        int S;
        int C;
        int K;
        int X;
        int Y;
        int P;
        int stride;
        int pooling_size;
        int pooling_stride;
        int input_neuron;
        int output_neuron;
        int batch;

        std::getline(ss, type, ',');  // 取出文件中的层类型字符串，存储在type中
        ss >> R;
        ss.ignore(); // 忽略逗号
        ss >> S;
        ss.ignore();
        ss >> C;
        ss.ignore();
        ss >> K;
        ss.ignore();
        ss >> X;
        ss.ignore();
        ss >> Y;
        ss.ignore();
        ss >> P;
        ss.ignore();
        ss >> stride;
        ss.ignore();
        ss >> pooling_size;
        ss.ignore();
        ss >> pooling_stride;
        ss.ignore();
        ss >> input_neuron;
        ss.ignore();
        ss >> output_neuron;
        ss.ignore();
        ss >> batch;

        layer_topology layer;
        layer.type = type;
        layer.R = R;
        layer.S = S;
        layer.C = C;
        layer.K = K;
        layer.X = X;
        layer.Y = Y;
        layer.P = P;
        layer.stride = stride;
        layer.pooling_size = pooling_size;
        layer.pooling_stride = pooling_stride;
        layer.input_neuron = input_neuron;
        layer.output_neuron = output_neuron;
        layer.batch = batch;

        layers.push_back(layer);
    }

    return layers;
}