//
// Created by Francisco Munoz on 19/06/19.
//
#include "TemporalRN.h"
#include <assert.h>
#include "utility.h"
#include <math.h>

//TODO Conectar los enlaces intermedios de forwarding
//This Constructor creates the reduction tree similar to the one shown in the paper
TemporalRN::TemporalRN(id_t id, std::string name, Config stonne_cfg, Connection* outputConnection) : ReduceNetwork(id, name) {
    // Collecting the parameters from configuration file
    this->stonne_cfg = stonne_cfg;
    this->port_width=stonne_cfg.m_ASwitchCfg.port_width; //16
    //End collecting the parameters from the configuration file
    //
    //Calculating the number of accumulators based on the shape of the multiplier network
    this->accumulation_buffer_size = stonne_cfg.m_MSNetworkCfg.ms_rows*stonne_cfg.m_MSNetworkCfg.ms_cols;
    
    // 实例化累积缓冲区
    this->accumulationBuffer = new AccumulationBuffer(0, "AccumulationBuffer", this->stonne_cfg, this->accumulation_buffer_size);
    
    //Creating the input connections
    // 累积缓冲区的个数等于乘法器的数量，下面创建累积缓冲区的输入连接
    for(int i=0; i<this->accumulation_buffer_size; i++) {
        Connection* input_connection = new Connection(this->port_width);
        inputconnectiontable.push_back(input_connection);
    }
    
    this->accumulationBuffer->setInputConnections(inputconnectiontable);
      
}

TemporalRN::~TemporalRN() {
   delete this->accumulationBuffer;
   for(int i=0; i < inputconnectiontable.size(); i++) {
       delete this->inputconnectiontable[i];
   }

}

// 设置与内存的连接，实际上是与总线的接口，归约网络的输出数据通过总线写入内存 
void TemporalRN::setMemoryConnections(std::vector<Connection*> memoryConnections) {

    this->outputconnectiontable = memoryConnections;
    this->accumulationBuffer->setMemoryConnections(outputconnectiontable);

}

// 返回输入连接，从向量转换为映射 
std::map<int, Connection*> TemporalRN::getLastLevelConnections() {
    //Converting from vector to map for questions of compatibility with the rest of the code
    std::map<int, Connection*> last_level_connections; 
    for(int i=0; i<inputconnectiontable.size(); i++) {
        last_level_connections[i]=inputconnectiontable[i]; 
    }
    return last_level_connections;
}


void TemporalRN::resetSignals() {
    this->accumulationBuffer->resetSignals();
}


void TemporalRN::configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding) {
    this->accumulationBuffer->configureSignals(current_tile, dnn_layer, ms_size, n_folding);
} 


void TemporalRN::configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int ms_size) {
    assert(false);
}


//TODO Implementar esto bien
void TemporalRN::cycle() {
    this->accumulationBuffer->cycle();
}

//Print configuration of the TemporalRN 
void TemporalRN::printConfiguration(std::ofstream& out, unsigned int indent) {

    out << ind(indent) << "\"ASNetworkConfiguration\" : {" << std::endl;
    out << ind(indent) << "}";
    
}

//Printing stats
void TemporalRN::printStats(std::ofstream& out, unsigned int indent) {
     out << ind(indent) << "\"ASNetworkStats\" : {" << std::endl;
     this->accumulationBuffer->printStats(out, indent+IND_SIZE);
     out << ind(indent) << "}";
}

void TemporalRN::printEnergy(std::ofstream& out, unsigned int indent) {
     /*
      The TemporalRN component prints the counters for the next subcomponents:
          - Accumulators
          - Input wires are not printed as we consider this accumulator as inside the multiplier (it is an entire PE)

      Note that the wires that connect with memory are not taken into account in this component. This is done in the CollectionBus.

     */
     //Printing the accumulator stats
     this->accumulationBuffer->printEnergy(out, indent);

     

}
