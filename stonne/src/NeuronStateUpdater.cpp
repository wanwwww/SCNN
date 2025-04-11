
#include "NeuronStateUpdater.h"
#include <math.h>


NeuronStateUpdater::NeuronStateUpdater(id_t id, std::string name, Config stonne_cfg) : Unit(id,name){
    this->stonne_cfg = stonne_cfg;
    this->port_width = stonne_cfg.m_ASwitchCfg.port_width;
    this->n_updaters = stonne_cfg.m_MSNetworkCfg.ms_rows*stonne_cfg.m_MSNetworkCfg.ms_cols;

    std::string name_str = "stateUpdate";

    for(int i=0;i<this->n_updaters;i++){
        std::string name_SU = name_str+=i;
        Connection* input_connection = new Connection(this->port_width);
        NeuronStateUpdaterSwitch* updater = new NeuronStateUpdaterSwitch(i, name_SU, stonne_cfg, i, input_connection);
        this->inputconnectiontable.push_back(input_connection);
        this->updatertable[i] = updater;
    }

}

NeuronStateUpdater::~NeuronStateUpdater(){
    for(int i=0;i<this->n_updaters;i++){
        delete this->inputconnectiontable[i];
        delete this->updatertable[i];
    }
}

void NeuronStateUpdater::setOutputConnections(std::vector<std::vector<Connection*>> memoryconnections){
    // 先将memoryconnections转化为向量形式，存在outputconnectiontable中
    unsigned int n_bus_lines = memoryconnections.size(); // n 条总线线
    for(int i=0;i<this->n_updaters;i++){
        unsigned int inputID = (i / n_bus_lines);
        unsigned int busID = i % n_bus_lines;
        Connection* mem_conn = memoryconnections[busID][inputID];
        outputconnectiontable.push_back(mem_conn);
        updatertable[i]->setOutputConnection(mem_conn);
    }
}

void NeuronStateUpdater::setPoolingconnections(std::vector<Connection*> poolingconnections){
    //std::cout<<"setPoolingconnections"<<std::endl;
    for(int i=0;i<this->n_updaters;i++){
        //std::cout<<i<<std::endl;
        this->poolingconnectiontable.push_back(poolingconnections[i]);
        //std::cout<<"save the pooling connection to table"<<std::endl;
        updatertable[i]->setPoolingConnection(poolingconnections[i]);
    }
}

std::vector<Connection*> NeuronStateUpdater::getInputConnections(){
    return this->inputconnectiontable;
}

void NeuronStateUpdater::setMemoryController(OSMeshSDMemory* mem){
    for(int i=0;i<this->n_updaters;i++){
        updatertable[i]->setMemController(mem);
    }
}

void NeuronStateUpdater::cycle(){
    for(int i=0;i<this->n_updaters;i++){
        updatertable[i]->cycle();
    }
}


void NeuronStateUpdater::printStats(std::ofstream& out, unsigned int indent){

}

void NeuronStateUpdater::printEnergy(std::ofstream& out, unsigned int indent){

}

void NeuronStateUpdater::setConfiguration(Config cfg){
    
}