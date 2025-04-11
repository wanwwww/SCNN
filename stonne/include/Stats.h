//Created by Francisco Munoz Martinez on 25/11/2019
/*
This file is created to have a centralized control of all the stat parameters. 
We use a Stat empty class as the parent class of the rest ones. This is so to put in the Component class 
*/

//The function implementations are written in src/Stats.cpp. For every parameter added, this file must be modified


#ifndef __STATS__H__
#define __STATS__H__

#include <iostream>
#include <fstream>
#include "types.h"
#include <vector>

// 创建此文件是为了集中控制所有stat参数
// 提供了统一的接口，可以在不同组件上收集和打印统计信息

// 基类，是一个纯虚类，为所有统计类提供统一接口。定义两种虚函数，reset()重置统计信息，print()将统计信息打印到文件流中
// 派生类，每个派生类对应一个特定的硬件模块或组件，并包含与该模块相关的统计数据

class Stats {
public:
    virtual void reset() = 0;
    virtual void print(std::ofstream& out, unsigned int indent) = 0;
};


// 用于跟踪connection的统计信息
class ConnectionStats : public Stats {
public:
    // 发送数据的次数和接受数据的次数
    counter_t n_sends;          // n times the connection is used to send a data
    counter_t n_receives;       // n times the connection is used to receive a data
    
    ConnectionStats(); // 构造函数调用 reset() 函数 
    void reset();  // 重置函数将 n_sends 和 n_receives置为0
    void print(std::ofstream& out, unsigned int indent);

};


class FifoStats : public Stats {
public:
    counter_t n_pops; // 弹出操作次数
    counter_t n_pushes; // 推入操作次数
    counter_t n_fronts; //Look up without poping 查看队首数据的次数
    counter_t average_occupancy; // 平均占用量
    counter_t max_occupancy; // 最大占用量 

    FifoStats(); //构造函数调用 reset() 函数 
    void reset();
    void print(std::ofstream& out, unsigned int indent);

};


class DSwitchStats : public Stats {
public:
    counter_t total_cycles;
    counter_t n_broadcasts;         //N cycles the ASwitch send the input data through both output connections 广播的次数 
    counter_t n_unicasts;           // N cycles the ASwitch send the input data throgh a single connection (left or right)  单播的次数 
    counter_t n_left_sends;         // N cycles the ASwitch send the input data to the left connection (it does not matter what happens with the right one)
    counter_t n_right_sends;        // N cycles the ASwitch send the input data to the right connection (it does not matter what happens with the left one)
    DSwitchStats();
    void reset();
    void print(std::ofstream& out, unsigned int indent);
    

};


class MSwitchStats : public Stats {
public: 
    counter_t total_cycles;
    counter_t n_multiplications;                       //N cycles the MSwitch performs a multiplication
    counter_t n_input_forwardings_send;                //N cycles the MSwitch sends an input data to the neighbour MSwitch
    counter_t n_input_forwardings_receive;             //N cycles the MSwitch receives an input data from the neighbour MSwitch 
    counter_t n_inputs_receive;                        //N cycles the MSwitch receives a new input from the SDMemory
    counter_t n_weights_receive;                       //N cycles the MSwitch receives a new weight from the SDMemory
    counter_t n_weight_fifo_flush;                     //N cycles the MSwitch has to flush the FIFO that stores the weights. (i.e., it has to process another filter)
    counter_t n_psums_receive;                         //N cycles the MSwitch receives a new psum from the SDMemory
    counter_t n_psum_forwarding_send;                  //N cycles the MSwitch forwards a psum (i.e., there is folding and therefore this MSwitch is forwarding the psums to the ART.
    counter_t n_configurations;                        // N times the MSwitch is configured
    
    MSwitchStats();
    void reset();
    void print(std::ofstream& out, unsigned int indent);

};


class MultiplierOSStats : public Stats {
public:
    counter_t total_cycles;
    counter_t n_multiplications;                     //N cycles the Multiplier performs a multiplication
    counter_t n_bottom_forwardings_send;             //N cycles the Multiplier sends an weight to the bottom neighbour multiplier
    counter_t n_top_forwardings_receive;             //N cycles the Multiplier receives a weight from the top multiplier
    counter_t n_right_forwardings_send;              //N cycles the Multiplier sends an input to the right neighbour multiplier
    counter_t n_left_forwardings_receive;            //N cycles the Multiplier receives an input from the left neighbour multiplier
    counter_t n_configurations;                      // N times the MSwitch is configured

    MultiplierOSStats();
    void reset();
    void print(std::ofstream& out, unsigned int indent);

};


class ASwitchStats: public Stats {
public:
    counter_t total_cycles;                             //N cycles 
    counter_t n_2_1_sums;                               //N cycles the ASwitch performs a 2_1 sum
    counter_t n_2_1_comps;                              //N cycles the ASwitch performs a 2_1 comparision   TODO NOT IMPLEMENTED YET
    counter_t n_3_1_sums;                               //N cycles the ASwitch performs a 3_1 sum
    counter_t n_3_1_comps;                              //N cycles the ASwitch performs a 3_1 comparision   TODO NOT IMPLEMENTED YET
    counter_t n_parent_send;                            //N cycles the ASwitch send a data to the parent
    counter_t n_augmented_link_send;                    //N cycles the ASwitch send a data to the augmented link 向增强链路发送数据的次数
    counter_t n_memory_send;                            //N cycles the ASwitch send a data directly to memory (to the buses)
    counter_t n_configurations;                         //N times the ASwitch has been configured
    ASwitchStats();
    void reset();
    void print(std::ofstream& out, unsigned int indent);

};


class AccumulatorStats: public Stats {
public:
    counter_t total_cycles;                             //N cycles
    counter_t n_adds;                                   //N cycles the Accumulator performs a sum
    counter_t n_memory_send;                            //N cycles the Accumulator sends a data directly to memory (to the buses)
    counter_t n_receives;                               //N cycles the Accumulator receives a data from the ASNetwork
    counter_t n_register_reads;                         //N cycles the Accumulator reads from its local register
    counter_t n_register_writes;                        //N cycles the Accumulator writes to its local register
    counter_t n_configurations;                         //N times the Accumulator has been configured
    AccumulatorStats();
    void reset();
    void print(std::ofstream& out, unsigned int indent);

};


class SDMemoryStats : public Stats {
public:
    counter_t total_cycles;                                      //N cycles 
    counter_t n_SRAM_weight_reads;                               //Number of SRAM weight reads  权重读取次数 
    counter_t n_SRAM_input_reads;                                //Number of SRAM input reads  输入读取次数  
    counter_t n_SRAM_psum_reads;                                 //Number of SRAM psum reads 
    counter_t n_SRAM_psum_writes;                                //Number of SRAM psum writes
    counter_t sta_sparsity;                                      //Exent of sparsity sta matrix. Only for sparse controllers
    counter_t str_sparsity;                                      //Exent of sparsity str matrix. Only for sparse controllers
    counter_t n_sta_vectors_at_once_avg;                         //Number of sta vectors that fits on average at every reconfiguration.
    counter_t n_sta_vectors_at_once_max;                         //Maximum number of vectors mapped at once
    counter_t n_reconfigurations;                                //Number of reconfigurations done in the MN and RN networks
    Dataflow dataflow;                                           //Dataflow to use. Note this is a runtime stat
    std::vector<counter_t> n_SRAM_read_ports_weights_use;        //Number of times each read port is used to deliver a weight
    std::vector<counter_t> n_SRAM_read_ports_inputs_use;         //Number of times each read port is used to deliver an input
    std::vector<counter_t> n_SRAM_read_ports_psums_use;          //Number of times each read port is used to deliver a psum
    std::vector<counter_t> n_SRAM_write_ports_use;               //Number of times each write port is used to write a psum

    SDMemoryStats();
    void reset();
    void print(std::ofstream& out, unsigned int indent);
    
};


// 总线线统计数据
class CollectionBusLineStats : public Stats {
public:
    counter_t total_cycles;                                      //N cycles 
    counter_t n_times_conflicts;                                 //N times there is more than 1 outputs to send in the same bus in the same cycle
    counter_t n_conflicts_average;                               //N inputs that there is on average trying to get access to the output every cycle
    counter_t n_sends;                                           //N times the bus send a data to the output
    std::vector<counter_t> n_inputs_receive;                     //N inputs that have been received by each input line
    CollectionBusLineStats();
    void reset();
    void print(std::ofstream& out, unsigned int indent);

};



#endif

