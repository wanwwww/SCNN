//Created 13/06/2019

#ifndef __types__h__

#define __types__h__

#define STONNE_WORD_SIZE 1
#define IND_SIZE 4
#define timescale 10

#include <string>

typedef unsigned int data_t;
typedef unsigned int bandwidth_t;
typedef unsigned int id_t;
typedef unsigned int cycles_t;
typedef int* address_t;
typedef unsigned int counter_t;
typedef unsigned int latency_t;
typedef unsigned int* metadata_address_t;


enum operand_t {WEIGHT, IACTIVATION, OACTIVATION, PSUM, VTH, SPIKE};
enum traffic_t {BROADCAST, MULTICAST, UNICAST}; // 广播、多播、单播
enum direction_t {LEFT, RIGHT};
//Adder configuration signals
enum fl_t {RECEIVE, SEND, NOT_CONFIGURED}; ///forwarding link type
enum adderconfig_t {ADD_2_1, ADD_3_1, ADD_1_1_PLUS_FW_1_1, FW_2_2, NO_MODE, FOLD}; // To the best of my knowledge, FW_2_2 corresponds with sum left and right and send the result to the FW.
/////
enum Layer_t{CONV, FC, POOL};
//enum ReduceNetwork_t{ASNETWORK, FENETWORK, TEMPORALRN};
//enum MultiplierNetwork_t{LINEAR, OS_MESH};
/////
//enum MemoryController_t{MAERI_DENSE_WORKLOAD, SIGMA_SPARSE_GEMM, MAGMA_SPARSE_DENSE, TPU_OS_DENSE};
//enum SparsityControllerState{CONFIGURING, DIST_STA_MATRIX, DIST_STR_MATRIX, WAITING_FOR_NEXT_STA_ITER, ALL_DATA_SENT};
enum OSMeshControllerState{OS_CONFIGURING, OS_DIST_INPUTS, OS_WAITING_FOR_NEXT_ITER, OS_ALL_DATA_SENT};
enum PoolingMemoryControllerState{POOLINGMODULE_CONFIGURING, POOLING_DATA_DIST, POOLING_ALL_DATA_SENT}; // 池化模块内存控制器，配置状态、数据分发状态、数据发送完毕状态

/////
enum Dataflow{CNN_DATAFLOW, MK_STA_KN_STR, MK_STR_KN_STA, SPARSE_DENSE_DATAFLOW};
enum GENERATION_TYPE{GEN_BY_ROWS, GEN_BY_COLS};
enum WIRE_TYPE{RN_WIRE, MN_WIRE, DN_WIRE};

enum adderoperation_t {ADDER, COMPARATOR, MULTIPLIER, NOP, UPDATE, POOLING};

enum pooling_t {MAXPOOLING, AVERAGEPOOLING};

//Testbench
enum layerTest {TINY, LATE_SYNTHETIC, EARLY_SYNTHETIC, VGG_CONV11, VGG_CONV1};

// 层拓扑
struct layer_topology
{
    std::string type;
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
};

// 乒乓buffer
typedef struct{
    int* current_buffer;  // 用于计算的buffer
    int* next_buffer;   // 用于读取数据的buffer
    bool buffer_toggle;  // 切换
} PingPong_Buffer;

// 用于记录从犯DRAM中读取数据到input_buffer的数据结构
typedef struct{
    //int index; //第几次取数据
    int num_rows; // 取几行
    int start_rows; // 从第几行开始取
    int add_0_above; // 上面加几行0
    int add_0_below; // 下面加几行零
} record;

// 硬件配置
// struct hardware_cfg
// {
//     unsigned int ms_rows;
//     unsigned int ms_cols;

//     unsigned int dn_bw;
//     unsigned int rn_bw;

//     unsigned int  accumulation_buffer_enabled;

//     ReduceNetwork_t rn_type;
//     MultiplierNetwork_t mn_type;
//     MemoryController_t mem_ctrl;

//     // SNN
//     data_t Vth;
//     data_t Timestamp;
//     pooling_t pooling_type;
// };


#endif
