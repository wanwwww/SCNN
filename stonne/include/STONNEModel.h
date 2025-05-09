#ifndef STONNEMODEL_H_
#define STONNEMODEL_H_

#include <string>
//#include "RSNetwork.h"
//#include "MSNetwork.h"
//#include "DSNetworkTop.h"
//#include "ASNetwork.h"
//#include "SDMemory.h"
#include "Connection.h"
//#include "LookupTable.h"
#include "CollectionBus.h"
#include "Config.h"
//#include "CompilerART.h"
//#include "CompilerMSN.h"
#include "ReduceNetwork.h"
//#include "DistributionNetwork.h"
//#include "FENetwork.h"
#include "MemoryController.h"
//#include "SparseSDMemory.h"
//#include "SparseDenseSDMemory.h"
#include "TemporalRN.h"
#include "OSMeshSDMemory.h"
#include "OSMeshMN.h"
//#include "TileGenerator/TileGenerator.h"

#include "NeuronStateUpdater.h"

// Stonne类是整个模拟器的核心，封装了硬件参数、功能模块、连接逻辑和运行流程
class Stonne {
public:
    //Hardware paramenters
    Config stonne_cfg;  // 存储模拟器的配置参数 
    unsigned int ms_size; //Number of multipliers
    unsigned int n_adders; //Number of adders obtained from ms_size
    bool pooling_enabled;

    //DistributionNetwork* dsnet; //Distribution Network
    MultiplierNetwork* msnet; //Multiplier Network
    ReduceNetwork* asnet; //ART Network
    // add
    NeuronStateUpdater* updatenet;

    OSMeshSDMemory* mem; //MemoryController abstraction (e.g., SDMemory from MAERI)

    Bus* collectionBus; //CollectionBus
    
    // connections
    Connection* outputASConnection; //The last connection of the AS and input to the lookuptable 
    //Connection* outputLTConnection; //Output of the lookup table connection and write port to the SDMemory 

    Connection** addersBusConnections; //Array of output connections between the adders and the bus 
    Connection** BusMemoryConnections; //Array of output Connections between the bus and the memory. (Write output ports)

    //Software parameters
    DNNLayer* dnn_layer; // 指向当前加载的网络层 
    Tile* current_tile; // 指向当前的tile
    bool layer_loaded; //Indicates if the function loadDNN  是否加载了网络层
    bool tile_loaded; // 是否加载了当前的tile

    //Connection and cycle functions
    // 这些函数负责连接硬件模块之间的数据流 
    //void connectMemoryandDSN();  // 连接内存和分发网络 
    //void connectMSNandDSN(); // 连接分发网络和乘法网络 Function to connect the multiplieers of the MSN to the last level switches in the DSN.

    void connectMemoryandMSN(); //连接内存和乘法网络
    void connectMSNandASN(); //将乘法器的输出连接到加法器网络 

    // modify
    void connectASNandUpdateNet(); //连接归约网络和膜电位更新网络
    void connectionUpdateNetandBus();  // 连接膜电位更新网络和总线
    void connectBusandMemory(); //Connect the bus and the memory write ports. 将总线的输出连接到内存，完成存储

    void cycle(); //模拟硬件的一个周期 
    void printStats();
    void printEnergy();
    void printGlobalStats(std::ofstream& out, unsigned int indent); //打印整体统计信息到文件 

    //DEBUG functions
    void testDSNetwork(unsigned int num_ms);
    void testTile(unsigned int num_ms);
    void testMemory(unsigned int num_ms);

   
public:

    //Statistics
    unsigned int n_cycles;   // 模拟器运行的总周期数
    // DEBUG PARAMETERS
    unsigned long time_ds;
    unsigned long time_ms;
    unsigned long time_as;
    unsigned long time_lt;
    unsigned long time_mem;

    unsigned long time_update;
    unsigned long time_pooling;


    Stonne (Config stonne_cfg);
    ~Stonne();

    //General constructor. Generates mRNA configuration if enabled
    void loadDNNLayer(Layer_t layer_type, std::string layer_name, unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G,  unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, address_t input_address, address_t filter_address, address_t output_address, address_t neuron_state, Dataflow dataflow);

    //Load CONV Layer. At the end this calls to the general constructor  with all the parameters
    //void loadCONVLayer(std::string layer_name, unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G, unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, address_t input_address, address_t filter_address, address_t output_address);

    //Load FC layer just with the appropiate parameters
    //N = batch size (i.e., number of rows in input matrix); S=number of inputs per batch (i.e., column size in input matrix and weight matrix); K=number of outputs neurons (i.e, number of rows weight matrix)
    //void loadFCLayer(std::string layer_name, unsigned int N, unsigned int S, unsigned int K, address_t input_address, address_t filter_address, address_t output_address); 

    //Load Sparse GEMM onto STONNE according to SIGMA parameter taxonomy. 
    //void loadGEMM(std::string layer_name, unsigned int N, unsigned int K, unsigned int M, address_t MK_matrix, address_t KN_matrix, metadata_address_t MK_metadata, metadata_address_t KN_metadata, address_t output_matrix, metadata_address_t output_metadata, Dataflow dataflow);

    //Load Dense GEMM onto STONNE according to SIGMA parameter taxonomy and tiling according to T_N, T_K and T_M
    void loadDenseGEMM(std::string layer_name, unsigned int N, unsigned int K, unsigned int M, address_t MK_matrix, address_t KN_matrix, address_t output_matrix, address_t neuron_state, Dataflow dataflow);

    //Load sparse-dense GEMM onto STONNE
    //void loadSparseDense(std::string layer_name, unsigned int N, unsigned int K, unsigned int M, address_t MK_matrix, address_t KN_matrix, metadata_address_t MK_metadata_id, metadata_address_t MK_metadata_pointer, address_t output_matrix, unsigned int T_N, unsigned int T_K);

    // Generic method to load a tile
    void loadTile(unsigned int T_R, unsigned int T_S, unsigned int T_C, unsigned int T_K, unsigned int T_G, unsigned int T_N, unsigned int T_X_, unsigned int T_Y_); //Load general and CONV tile

    //Load a Dense GEMM tile to run it using the loadDenseGEMM function
    void loadGEMMTile(unsigned int T_N, unsigned int T_K, unsigned int T_M);

    // Loads a FC tile to run it using the loadFC function
    //void loadFCTile(unsigned int T_S, unsigned int T_N, unsigned int T_K); //VNSize = T_S, NumVNs= T_N*T_K

    // Loads a SparseDense tile to run it using the loadSparseDense function
    //void loadSparseDenseTile(unsigned int T_N, unsigned int T_K);

    //Loads a tile configuration from a file
    //void loadTile(std::string tile_file);

    // Generates a tile configuration using a TileGenerator module
    // void generateTile(TileGenerator::Generator generator = TileGenerator::Generator::CHOOSE_AUTOMATICALLY,
    //                   TileGenerator::Target target = TileGenerator::Target::PERFORMANCE,
    //                   float MK_sparsity = 0.0f);

    void run();

};

#endif
//TO DO add enumerate.
