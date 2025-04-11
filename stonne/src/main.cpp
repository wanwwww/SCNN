#include <iostream>
#include "STONNEModel.h"
#include "types.h"
#include <chrono>
#include <assert.h>
#include "testbench.h"
#include <string>
#include <math.h>
#include <utility.h>
#include <cstdlib>
#include <filesystem>

#include "Controller.h"


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
    //h_cfg = readCSV_cfg(hardware_cfg_path);
    // std::cout<<" the layers topo "<<std::endl;
    // for(int i=0;i<layers.size();i++){
    //     std::cout<<layers[i].type<<" ";
    //     std::cout<<layers[i].R<<" ";
    //     std::cout<<layers[i].S<<" ";
    //     std::cout<<layers[i].C <<" ";
    //     std::cout<<layers[i].K <<" ";
    //     std::cout<<layers[i].X <<" ";
    //     std::cout<<layers[i].Y <<" ";
    //     std::cout<<layers[i].input_neuron<<" ";
    //     std::cout<<layers[i].output_neuron<<" ";
    //     std::cout<<layers[i].batch<<std::endl;
    //     std::cout<<std::endl;
    // }

    // 例化config，加载硬件参数
    Config stonne_cfg;
    stonne_cfg.loadFile(hardware_cfg_path);

    // 实例化控制器
    Controller* control = new Controller(stonne_cfg, layers);

    control->run();
    
    delete control;

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
        layer.input_neuron = input_neuron;
        layer.output_neuron = output_neuron;
        layer.batch = batch;

        layers.push_back(layer);
    }

    return layers;
}


// #include <iostream>
// #include <atomic>
// #include "dramsim3.h"

// static uint64_t globalCycle = 0;
// int completed_reads = 0;
// int completed_writes = 0;

// void ReadCallback(uint64_t addr){
//     std::cout << "Read completed at address: 0x" << std::hex << addr
//               << " at cycle " << std::dec << globalCycle << std::endl;
//     completed_reads++;
// }

// void WriteCallback(uint64_t addr){
//     std::cout << "Write completed at address: 0x" << std::hex << addr
//               << " at cycle " << std::dec << globalCycle << std::endl;
//     completed_writes++;
// }


// int main(){
//     std::string config_file = "/home/zww/DRAMsim3/configs/DDR4_8Gb_x8_3200.ini";
//     std::string output_dir = "/home/zww/DRAMsim3/output";

//     // 获取DRAM模拟器示例
//     dramsim3::MemorySystem* dram = dramsim3::GetMemorySystem(config_file,output_dir,ReadCallback,WriteCallback);
//     if (!dram) {
//         std::cerr << "Failed to create DRAM simulation system!" << std::endl;
//         return -1;
//     }

//     // 示例：添加新的读事务
//     const int num_requests=10;   // 请求个数
//     int next_request = 0; // 下一个待发送请求的索引
//     uint64_t base_addr = 0x1000;
//     uint64_t offset = 0x40;

//     // 用于记录最后一个读请求完成的时钟周期
//     int last_completion_cycle = -1;

//     //const int max_cycles=5000; //假设最多模拟max_cycles个周期

//     while(true){

//         // 如果还有请求待发送，则尝试在当前周期发送一个请求
//         if(next_request < num_requests){
//             uint64_t addr = base_addr + next_request * offset;

//             // 检查当前周期是否能接受新事物
//             if (dram->WillAcceptTransaction(addr, false)) { 
//                 if(dram->AddTransaction(addr, false)){
//                     std::cout << "Added read transaction at address: 0x" << std::hex << addr <<" at cycle "<<std::dec<<globalCycle<<std::endl;
//                     // 发送成功后
//                     next_request++;
//                 } else {
//                     std::cout << "Failed to add read transaction!!!  error error error" <<" at cycle "<<std::dec<<globalCycle<<std::endl;
//                 }
//             } else {
//                 std::cout << "DRAM not ready to accept new transaction.--------------" << " at cycle "<<std::dec<<globalCycle<<std::endl;  // 等待下一周期
//             }
        
//         } 

//         dram->ClockTick();

//         // 检查是否所有读请求均已完成
//         if(completed_reads==num_requests && last_completion_cycle < 0 ){
//             last_completion_cycle = globalCycle;
//             std::cout<<"All requests are completed in clock cycle: "<<last_completion_cycle<<std::endl;
//             break;
//         }

//         globalCycle++;
//     }

    
//     // 释放资源
//     delete dram;

//     return 0;

// }