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

int main(int argc, char *argv[]) {

    std::string layer_topology_path;
    std::string hardware_cfg_path;

    std::vector<layer_topology> layers;
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

    layers = readCSV_layers(layer_topology_path);
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

    // 例化config，加载硬件参数
    Config stonne_cfg;
    stonne_cfg.loadFile(hardware_cfg_path);

    // 实例化控制器
    Controller* control = new Controller(stonne_cfg, layers);
    //std::tie(ifmap, filter, ofmap, nfmap) = control->runConvandPooling_DataFlow(1,ifmap,filter,ofmap,nfmap,layers[0]);
    //control->run();

    for(int i=0; i<layers.size(); i=i+2){
        int layer_id = i+1;
        // 奇数层的输入在ifmap，输出存储在ofmap
        if(layers[i].type == "conv_pooling") {
            std::tie(ifmap, filter, ofmap, nfmap) = control->runConvandPooling_DataFlow(layer_id, ifmap, filter, ofmap, nfmap, layers[i]);
            delete[] ifmap;
            delete[] filter;
            delete[] nfmap;
        } else if(layers[i].type == "conv") {
            std::tie(ifmap, filter, ofmap, nfmap) = control->runConv_DataFlow_2(layer_id, ifmap, filter, ofmap, nfmap, layers[i]);
            delete[] ifmap;
            delete[] filter;
            delete[] nfmap;
        } else if(layers[i].type == "fc") {
            std::tie(ifmap, filter, ofmap, nfmap) = control->runFC(layer_id, ifmap, filter, ofmap, nfmap, layers[i]);
            delete[] ifmap;
            delete[] filter;
            delete[] nfmap;
        } else {
            std::cout<<"\033[1;31m"<<"Unsupported layer types !"<<"\033[0m"<<std::endl;
        }

        // 偶数层的输入为上一层的输出
        if((i+1)<layers.size()){
            layer_id++;
            if(layers[i+1].type == "conv_pooling") {
                std::tie(ifmap, filter, ofmap, nfmap) = control->runConvandPooling_DataFlow(layer_id, ofmap, filter, ifmap, nfmap, layers[i+1]);
                // delete[] ifmap;
                delete[] filter;
                delete[] nfmap;
                delete[] ofmap;
            } else if(layers[i].type == "conv") {
                std::tie(ifmap, filter, ofmap, nfmap) = control->runConv_DataFlow_1(layer_id, ofmap, filter, ofmap, nfmap, layers[i+1]);
                delete[] ofmap;
                delete[] filter;
                delete[] nfmap;
            } else if(layers[i].type == "fc") {
                std::tie(ifmap, filter, ofmap, nfmap) = control->runFC(layer_id, ofmap, filter, ifmap, nfmap, layers[i+1]);
                delete[] ofmap;
                delete[] filter;
                delete[] nfmap;
            } else {
                std::cout<<"\033[1;31m"<<"Unsupported layer types !"<<"\033[0m"<<std::endl;
            }
        } else {
            delete[] ofmap;
        }
    }
    
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