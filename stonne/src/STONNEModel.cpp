//
// Created by Francisco Munoz-Martinez on 18/06/19.
//
#include "STONNEModel.h"

#include <assert.h>
#include <chrono>
#include "types.h"
#include <vector>
#include "Tile.h"
#include "utility.h"
#include "Config.h"
#include <time.h>
#include <math.h>
#include <unistd.h>
#include "cpptoml.h"

Stonne::Stonne(Config stonne_cfg) {
    
    // std::cout<<"debug in Stonne constructor function"<<std::endl;

    this->stonne_cfg=stonne_cfg;

    this->ms_size = stonne_cfg.m_MSNetworkCfg.ms_rows*stonne_cfg.m_MSNetworkCfg.ms_cols;

    this->layer_loaded=false;
    this->tile_loaded=false;

    this->outputASConnection = new Connection(16);  // 16  归约网络和查找表的连接
    
    //this->outputLTConnection = new Connection(stonne_cfg.m_SDMemoryCfg.port_width);  // 1
    
    this->pooling_enabled = false;
    
    // 乘法器网络
    this->msnet = new OSMeshMN(1, "OSMesh", stonne_cfg);  // id and name 

    // std::cout<<"debug in Stonne constructor function after new msnet"<<std::endl;

    // 实例化分发网络
    //switch(DistributionNetwork). It is possible to create instances of other DistributionNetworks.h
    //this->dsnet = new DSNetworkTop(1, "DSNetworkTop", stonne_cfg);
    
    // 加法网络
    this->asnet = new TemporalRN(2, "TemporalRN", stonne_cfg, outputASConnection); // 传入输出连接 

    // add 
    // 实例化膜电位更新网络
    this->updatenet = new NeuronStateUpdater(3, "NeuronStateUpdater", stonne_cfg);

    // 实例化总线
    this->collectionBus = new Bus(4, "CollectionBus", stonne_cfg); 
    //this->lt = new LookupTable(5, "LookUpTable", stonne_cfg, outputASConnection, outputLTConnection);

    // 内存控制器
    this->mem = new OSMeshSDMemory(0, "OSMeshSDMemory", stonne_cfg);

    //std::cout<<this->mem<<std::endl;

    this->mem->setUpdateNetwork(updatenet);  // add

    // 将加法器网络和乘法器网络添加到内存控制器中 
    //Adding to the memory controller the asnet and msnet to reconfigure them if needed
    this->mem->setReduceNetwork(asnet);
    this->mem->setMultiplierNetwork(msnet); 

    this->updatenet->setMemoryController(this->mem); // add

    // 计算加法器的个数 
    //Calculating n_adders
    this->n_adders=this->ms_size-1; 


    //rsnet
    //this->connectMemoryandDSN(); // 连接内存和分发网络
    //this->connectMSNandDSN(); // 连接乘法器网络和分发网络
    
    this->connectMemoryandMSN();  // 连接内存和乘法网络
    this->connectMSNandASN(); // 连接乘法器网络和加法器网络

    // modify 
    this->connectASNandUpdateNet(); // 连接归约网络和膜电位更新网络
    this->connectionUpdateNetandBus(); // 连接更新网络和总线
    this->connectBusandMemory();  // 连接从总线输出到内存 
  
    //DEBUG PARAMETERS
    //this->time_ds = 0;
    this->time_ms = 0;
    this->time_as = 0;
    //this->time_lt = 0;
    this->time_mem = 0;

    this->time_update = 0;
    this->time_pooling = 0;


    //STATISTICS
    this->n_cycles = 0;

    //std::cout<<" stonne instantiation completed "<<std::endl;

}

Stonne::~Stonne() {
    //delete this->dsnet;
    delete this->msnet;
    delete this->asnet;
    delete this->outputASConnection;
    //delete this->outputLTConnection;
    //delete this->lt;
    delete this->mem;
    delete this->collectionBus;

    delete this->updatenet;

    if(layer_loaded) {
        delete this->dnn_layer;
    }
  
    if(tile_loaded) {
        delete this->current_tile;
    } 
}

// 连接1 ： 连接内存和分发网络，将分发网络的顶层输入连接设置为内存的读连接
//Connecting the DSNetworkTop input ports with the read ports of the memory. These connections have been created
//by the module DSNetworkTop, so we just have to connect them with the memory.
// void Stonne::connectMemoryandDSN() {
//     std::vector<Connection*> DSconnections = this->dsnet->getTopConnections();
//     // std::cout<<std::endl;
//     // std::cout<<"dsnet.getTopConnections num : "<<DSconnections.size()<<std::endl;
//     // std::cout<<std::endl;
//     //Connecting with the memory
//     this->mem->setReadConnections(DSconnections);
// }

// // 连接2 ： 连接乘法网络和分发网络，将DN的最后一级连接设置为MN的输入连接 
// // 注意分发网络的最后一级节点个数要匹配乘法器的个数 
// //Connecting the multipliers of the mSN to the last level switches of the DSN. In order to do this link correct, the number of 
// //connections in the last level of the DSN (output connections of the last level switches) must match the number of multipliers. 
// //The multipliers are then connected to those connections, setting a link between them. 
// void Stonne::connectMSNandDSN() {
//     std::map<int, Connection*> DNConnections = this->dsnet->getLastLevelConnections(); //Map with the DS connections
//     // std::cout<<std::endl;
//     // std::cout<<"dsnet.getLastLevelConnections num : "<<DNConnections.size()<<std::endl;
//     // std::cout<<std::endl;
//     this->msnet->setInputConnections(DNConnections);
     
// }

// 直接连接内存和乘法网络
void Stonne::connectMemoryandMSN(){
    std::map<int, Connection*> MemReadconnections = this->mem->getReadConnections(); // 得到内存的读连接
    this->msnet->setInputConnections(MemReadconnections); // 将其设置为乘法网络的输入连接
}


// 连接3 ： 连接乘法器网络和归约网络，将归约网络的输入连接设置为乘法器网络的输出连接 
//Connect the multiplier switches with the Adder switches. Note the number of ASs connection connectionss and MSs must be the identical
void Stonne::connectMSNandASN() {
    // 得到的RNConnections是归约网络中累加缓冲区中多个累加节点的输入连接的映射
    std::map<int, Connection*> RNConnections = this->asnet->getLastLevelConnections(); //Map with the AS connections
    // 为MN设置输出连接，即为每个MS设置到累积缓冲区的连接，每一个ms有到上下左右和累积缓冲区五个连接
    this->msnet->setOutputConnections(RNConnections);

}

// 连接4 ： 连接归约网络和总线 
// 得到总线的输入连接，将其设置为归约网络的与内存的连接 
// void Stonne::connectASNandBus() { 
//     // 得到每条总线线的多个端口的输入连接（在TPU中每条总线线只对应一个数据端口）
//     std::vector<std::vector<Connection*>> connectionsBus = this->collectionBus->getInputConnections(); //Getting the CollectionBus Connections
//     this->asnet->setMemoryConnections(connectionsBus); //Send the connections to the ReduceNetwork to be connected according to its algorithm
// }

// 连接4 : 连接归约网络和膜电位更新网络
void Stonne::connectASNandUpdateNet(){
    std::vector<Connection*> connectionUpdater = this->updatenet->getInputConnections(); // 得到膜电位更新网络的输入连接 
    this->asnet->setMemoryConnections(connectionUpdater);
}

// 连接5 : 连接膜电位更新网络和总线
void Stonne::connectionUpdateNetandBus(){
    // 得到总线的输入连接，也就是每条总线线的所对应的输入连接
    std::vector<std::vector<Connection*>> connectionsBus = this->collectionBus->getInputConnections();
    // 将其设置为膜电位更新模块的输出连接 
    this->updatenet->setOutputConnections(connectionsBus);
}

// 连接 ： 连接总线和内存 
// 得到每条总线线的输出连接，将其设置为内存的写连接 
void Stonne::connectBusandMemory() {
    std::vector<Connection*> write_port_connections = this->collectionBus->getOutputConnections();
    this->mem->setWriteConnections(write_port_connections);      
}

// 
void Stonne::loadDNNLayer(Layer_t layer_type, std::string layer_name, unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G, unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, address_t input_address, address_t filter_address, address_t output_address, address_t neuron_state, Dataflow dataflow) {
    //std::cout<<"calling loadDNNLayer begin"<<std::endl;
    
    assert((C % G)==0); //G must be multiple of C
    assert((K % G)==0); //G must be multiple of K
    assert(X>=R);
    assert(Y>=S);
    if((layer_type==FC)) {
        //assert((R==1) && (C==1) && (G==1) && (Y==S) && (X==M)); //Ensure the mapping is correct
    } 
    this->dnn_layer = new DNNLayer(layer_type, layer_name, R,S, C, K, G, N, X, Y, strides);   
    this->layer_loaded = true;
    //std::cout<<"debug"<<std::endl;
    //std::cout << "this->mem: " << this->mem << std::endl;
    
    this->mem->setLayer(this->dnn_layer, input_address, filter_address, output_address, neuron_state, dataflow);
    //std::cout<<"debug out"<<std::endl;
    //std::cout<<"calling loadDNNLayer end"<<std::endl;
}

void Stonne::loadDenseGEMM(std::string layer_name, unsigned int N, unsigned int K, unsigned int M, address_t MK_matrix, address_t KN_matrix, address_t output_matrix, address_t neuron_state, Dataflow dataflow) {
    //Setting GEMM (from SIGMA) parameters onto CNN parameters:
    //input_matrix = MK
    //filter_matrix = KN
    //std::cout<<"debug in loadDenseGEMM"<<std::endl;
    loadDNNLayer(CONV, layer_name, 1, K, 1, N, 1, 1, M, K, 1, MK_matrix, KN_matrix, output_matrix, neuron_state, dataflow);
    //std::cout << "Loading a GEMM into STONNE" << std::endl;
}


//To dense CNNs and GEMMs 
void Stonne::loadTile(unsigned int T_R, unsigned int T_S, unsigned int T_C, unsigned int T_K, unsigned int T_G, unsigned int T_N, unsigned int T_X_, unsigned int T_Y_) {

    assert(this->layer_loaded);

    // 检查硬件资源是否足够 
    // ============================================认为这里可以只考虑输出维度 ???????????????????????????????????????????????????????
    assert((this->stonne_cfg.m_MSNetworkCfg.ms_rows*this->stonne_cfg.m_MSNetworkCfg.ms_cols) >= (T_R*T_S*T_C*T_K*T_G*T_N*T_X_*T_Y_));

    //Checking if the dimensions fit the DNN layer. i.e., the tile is able to calculate the whole layer.
    //std::cout << "Loading Tile: <T_R=" << T_R << ", T_S=" << T_S << ", T_C=" << T_C << ", T_K=" << T_K << ", T_G=" << T_G << ", T_N=" << T_N << ", T_X'=" << T_X_ << ", T_Y'=" << T_Y_ << ">" << std::endl; 
 

    //End check
    // 计算折叠次数，也就是完成计算需要重复的次数。ceil是向上取整函数。
    // =======================================这里只考虑RSC三个维度，可能是因为计算一个输出结果需要执行这么多次乘累加？？？？？？？？？？？？？？？
    unsigned int n_folding = ceil(this->dnn_layer->get_R() / (float) T_R)*ceil(this->dnn_layer->get_S() / (float)T_S) * ceil(this->dnn_layer->get_C() / (float)T_C) ;
    // std::cout<<std::endl;
    // std::cout<<"n_folding (STONNDModel.cpp) : "<<n_folding<<std::endl;
    // std::cout<<ceil(this->dnn_layer->get_R() / (float) T_R)<<" * "<<ceil(this->dnn_layer->get_S() / (float)T_S)<<" * "<<ceil(this->dnn_layer->get_C() / (float)T_C)<<std::endl;
    // std::cout<<std::endl;

    bool folding_enabled = false; //Condition to use extra multiplier. Note that if folding is enabled but some type of accumulation buffer is needed this is false as no fw ms is needed. 
    //std::cout<<"this->stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled : "<<this->stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled<<std::endl;
    // 如果存在折叠，但是RN无法自行累积，则要使用额外的MS来累积 
    if((n_folding > 1) && (this->stonne_cfg.m_ASNetworkCfg.accumulation_buffer_enabled==0) && 1 ) { //If there is folding and the RN is not able to acumulate itself, we have to use an extra MS to accumulate
        folding_enabled = true; 
        //When there is folding we leave one MS free per VN aiming at suming the psums. In next line we check if there are
        // enough mswitches in the array to support the folding. 
        assert(this->ms_size >= ((T_R*T_S*T_C*T_K*T_G*T_N*T_X_*T_Y_) + (T_K*T_G*T_N*T_X_*T_Y_))); //We sum one mswitch per VN 为每一个VN加一个mswitch
    }
    //std::cout<<"folding_enabled : "<<folding_enabled<<std::endl;

    this->current_tile = new Tile(T_R, T_S, T_C, T_K, T_G, T_N, T_X_, T_Y_, folding_enabled); // TPU情况下，folding_enabled = false，即不需要额外的累加器 
    
    //Setting the signals to the corresponding networks

    // 在TPU中，信号的配置是在内控控制器中完成的
    //If stride > 1 then all the signals of ms_fwreceive_enabled and ms_fwsend_enabled must be disabled since no reuse between MSwitches can be done. In order to not to incorporate stride
    //as a tile parameter, we leave the class Tile not aware of the stride. Then, if stride exists, here the possible enabled signals (since tile does not know about tile) are disabled.
    this->tile_loaded = true;
    this->mem->setTile(this->current_tile);
}

void Stonne::loadGEMMTile(unsigned int T_N, unsigned int T_K, unsigned int T_M)  {
    //loadTile(1, T_K, 1, T_M, 1, T_N, 1, 1);
    //std::cout << "Loading a GEMM tile" << std::endl;
    loadTile(1, T_K, 1, T_N, 1, 1, T_M, 1);
    //assert(this->layer_loaded && (this->dnn_layer->get_layer_type() == GEMM));   //Force to have the right layer with the GEMM parameters)
}


void Stonne::run() {
    //Execute the cycles
    this->cycle();
}


void Stonne::cycle() {
    //this->testDSNetwork(this->ms_size);
    //this->testTile(this->ms_size);
    //this->printStats();
    bool execution_finished=false;
    while(!execution_finished) {
        // std::cout<<std::endl;
        // std::cout<<"cycle : "<<this->n_cycles<<std::endl;
        // std::cout<<"mem state : "<<this->mem->getCurrentState()<<std::endl;
        auto start = std::chrono::steady_clock::now();
        this->mem->cycle();
        auto end = std::chrono::steady_clock::now();
        this->time_mem+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        // std::cout<<"mem completed"<<std::endl;

        this->collectionBus->cycle(); 

        // add
        start = std::chrono::steady_clock::now();
        this->updatenet->cycle();
        end = std::chrono::steady_clock::now();
        this->time_update+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //std::cout<<"updatenet completed"<<std::endl;


        start = std::chrono::steady_clock::now();
        this->asnet->cycle();
        //this->lt->cycle();
//        this->collectionBus->cycle(); //This order since these are connections that have to be seen in next cycle
        end = std::chrono::steady_clock::now();
        this->time_as+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::steady_clock::now();
        this->msnet->cycle();
        end = std::chrono::steady_clock::now();
        this->time_ms+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        start = std::chrono::steady_clock::now();
        //this->dsnet->cycle();
        end = std::chrono::steady_clock::now();
        this->time_ds+=std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        execution_finished = this->mem->isExecutionFinished();
        
        this->n_cycles++;

        //if(this->n_cycles > 25) { assert(1==0);}
    }

    // if(this->stonne_cfg.print_stats_enabled) { //If sats printing is enable
    //     this->printStats();
    //     this->printEnergy();
    // }

}

//General function to print all the STATS
void Stonne::printStats() {
    //std::cout << "Printing stats" << std::endl;

    std::ofstream out; 
    unsigned int num_ms = this->stonne_cfg.m_MSNetworkCfg.ms_rows*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    unsigned int dn_bw = this->stonne_cfg.m_SDMemoryCfg.n_read_ports;
    unsigned int rn_bw = this->stonne_cfg.m_SDMemoryCfg.n_write_ports;
    const char* output_directory=std::getenv("OUTPUT_DIR");
    std::string output_directory_str="";
    if(output_directory!=NULL) {
        std::string env_output_dir(output_directory);
        output_directory_str+=env_output_dir+"/";
    }

    out.open(output_directory_str+"output_stats_layer_"+this->dnn_layer->get_name()+"_architecture_MSes_"+std::to_string(num_ms)+"_dnbw_"+std::to_string(dn_bw)+"_"+"rn_bw_"+std::to_string(rn_bw)+"timestamp_"+std::to_string((int)time(NULL))+".txt"); //TODO Modify name somehow
    unsigned int indent=IND_SIZE;
    out << "{" << std::endl;

        //Printing input parameters
        this->stonne_cfg.printConfiguration(out, indent);
        out << "," << std::endl;

        //Printing layer configuration parameters
        this->dnn_layer->printConfiguration(out, indent);
        out << "," << std::endl;

        //Printing tile configuration parameters
        if (tile_loaded) {
            this->current_tile->printConfiguration(out, indent);
            out << "," << std::endl;
        }
        
        //Printing ASNetwork configuration parameters (i.e., ASwitches configuration for these VNs, flags, etc)
        this->asnet->printConfiguration(out, indent);
        out << "," << std::endl;
  
        this->msnet->printConfiguration(out, indent);
        out << "," << std::endl;

        //Printing global statistics
        this->printGlobalStats(out, indent);
        out << "," << std::endl;        

        //Printing all the components
        // this->dsnet->printStats(out, indent);  //DSNetworkTop //DSNetworks //DSwitches
        // out << "," << std::endl;
        this->msnet->printStats(out, indent);
        out << "," << std::endl;
        this->asnet->printStats(out, indent);
        out << "," << std::endl;
        this->mem->printStats(out, indent);
        out << "," << std::endl;
        this->collectionBus->printStats(out, indent);
        out << std::endl;
    
    out << "}" << std::endl;
    out.close();
}

void Stonne::printEnergy() {
    std::ofstream out;

    unsigned int num_ms = this->stonne_cfg.m_MSNetworkCfg.ms_rows*this->stonne_cfg.m_MSNetworkCfg.ms_cols;
    unsigned int dn_bw = this->stonne_cfg.m_SDMemoryCfg.n_read_ports;
    unsigned int rn_bw = this->stonne_cfg.m_SDMemoryCfg.n_write_ports;

    const char* output_directory=std::getenv("OUTPUT_DIR");
    std::string output_directory_str="";
    if(output_directory!=NULL) {
        std::string env_output_dir(output_directory);
        output_directory_str+=env_output_dir+"/";
    }

    out.open(output_directory_str+"output_stats_layer_"+this->dnn_layer->get_name()+"_architecture_MSes_"+std::to_string(num_ms)+"_dnbw_"+std::to_string(dn_bw)+"_"+"rn_bw_"+std::to_string(rn_bw)+"timestamp_"+std::to_string((int)time(NULL))+".counters"); //TODO Modify name somehow
    unsigned int indent=0;
    out << "CYCLES=" <<  this->n_cycles << std::endl; //This is to calculate the static energy
    out << "[MSNetwork]" << std::endl;
    this->msnet->printEnergy(out, indent);
    out << "[ReduceNetwork]" << std::endl;
    this->asnet->printEnergy(out, indent);
    out << "[GlobalBuffer]" << std::endl;
    this->mem->printEnergy(out, indent);
    out << "[CollectionBus]" << std::endl;
    this->collectionBus->printEnergy(out, indent);
    out << std::endl;
    out.close();
}

//Local function to the accelerator to print the globalStats
void Stonne::printGlobalStats(std::ofstream& out, unsigned int indent) {
    //unsigned int n_mswitches_used=this->current_tile->get_VN_Size()*this->current_tile->get_Num_VNs();
    //float percentage_mswitches_used = (float)n_mswitches_used / (float)this->stonne_cfg.m_MSNetworkCfg.ms_size;
    out << ind(indent) << "\"GlobalStats\" : {" << std::endl; //TODO put ID
    //out << ind(indent+IND_SIZE) << "\"N_mswitches_used\" : " << n_mswitches_used << "," << std::endl;
    //out << ind(indent+IND_SIZE) << "\"Percentage_mswitches_used\" : " << percentage_mswitches_used << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"N_cycles\" : " << this->n_cycles << std::endl;
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability

}

void Stonne::testMemory(unsigned int num_ms) {
   for(int i=0; i<20; i++) {
    this->mem->cycle();
    //this->dsnet->cycle();
    this->msnet->cycle();
   }
    
}

void Stonne::testTile(unsigned int num_ms) {
    Tile* tile = new  Tile(3,1,1,2,1,1,1,1, false);
    //Tile* tile = new Tile(CONV, 2,2,1,2,1,2,2,1);
    //tile->generate_signals(num_ms);
    std::map<std::pair<int,int>, adderconfig_t> switches_configuration;// tile->get_switches_configuration();
    for(auto it=switches_configuration.begin(); it != switches_configuration.end(); ++it) {
        std::pair<int,int> current_node (it->first);
        adderconfig_t conf = it->second;
        std::cout << "Switch " << std::get<0>(current_node) << ":" << std::get<1>(current_node) << " --> " << get_string_adder_configuration(it->second) <<  std::endl;
    }
}

void Stonne::testDSNetwork(unsigned int num_ms) {
    //BRoadcast test
     /*
    std::shared_ptr<DataPackage> data_to_send = std::make_shared<DataPackage>(32, 1, IACTIVATION, 0, BROADCAST);
    std::vector<std::shared_ptr<DataPackage>> vector_to_send;
    vector_to_send.push_back(data_to_send);
    this->inputConnection->send(vector_to_send);
    */

    //Unicast test
    /* 
    std::shared_ptr<DataPackage> data_to_send = std::make_shared<DataPackage>(32, 500, IACTIVATION, 0, UNICAST, 6);
    std::vector<std::shared_ptr<DataPackage>> vector_to_send;
    vector_to_send.push_back(data_to_send);
    this->inputConnection->send(vector_to_send);
    */

    //Multicast test 
    
    bool* dests = new bool[num_ms]; //16 MSs
    for(int i=0;i<num_ms; i++) {
        dests[i]=false;
    }
    
    //Enabling Destinations 
    for(int i=0; i<6; i++)
        dests[i]=true;

    std::shared_ptr<DataPackage> data_to_send = std::make_shared<DataPackage>(32, 1, IACTIVATION, 0, MULTICAST, dests, num_ms);
    std::vector<std::shared_ptr<DataPackage>> vector_to_send;
    vector_to_send.push_back(data_to_send);
    //this->inputDSConnection->send(vector_to_send);
    
    //Configuring the adders
    //First test
    std::map<std::pair<int,int>, adderconfig_t> switches_configuration; //Adders configuration
    std::map<std::pair<int,int>, fl_t> fwlinks_configuration;
    std::pair<int,int> switch0 (0,0);
    switches_configuration[switch0]=FW_2_2;

    std::pair<int,int> switch1(2,1);
    switches_configuration[switch1]=ADD_1_1_PLUS_FW_1_1;
    fwlinks_configuration[switch1]=SEND;

    std::pair<int,int> switch2(2,2);
    switches_configuration[switch2]=ADD_3_1;
    fwlinks_configuration[switch2]=RECEIVE;

//    asnet->addersConfiguration(switches_configuration);
 //   asnet->forwardingConfiguration(fwlinks_configuration);


 
    //this->dsnet->cycle(); //TODO REVERSE THE ORDER!!!
    this->msnet->cycle();
    for(int i=0; i<7; i++) {
       //this->lt->cycle();
       this->asnet->cycle(); // 2 to 1
    }
    
    delete[] dests;

}

