//
// Created by Francisco Munoz on 19/06/19.
//

#ifndef __REDUCENETWORK__H__
#define __REDUCENETWORK__H__

#include <iostream>
#include <fstream>
#include <map>
#include "types.h"
#include "Config.h"
#include "CompilerComponent.h"
#include "Connection.h"
#include "Unit.h"
#include "Tile.h"
#include "DNNLayer.h"

// 一个归约网络的基类，负责将计算结果累加到输出缓冲区中

class ReduceNetwork : public Unit{
    
public:
    ReduceNetwork(id_t id, std::string name)  : Unit(id, name) {}
    virtual ~ReduceNetwork() {}
    // 配置网络中的内存连接，将所有内存端口从总线（busID,lineID）连接到其对应的节点
    virtual void setMemoryConnections(std::vector<Connection*> memoryConnections) {assert(false);} //Connect all the memory ports from buses (busID, lineID) to its corresponding switches
    // 获取最后一级连接，通常是输出连接
    virtual std::map<int, Connection*> getLastLevelConnections() {assert(false);}
    // 设置输出连接。用于将规约结果发送到预取缓冲区
    virtual void setOutputConnection(Connection* outputConnection)  {assert(false);} //This function set the outputConnection with the Prefetch buffer
    // 配置归约网络的信号
    virtual void configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_size, unsigned int n_folding) {assert(false);}
    virtual void configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int ms_size) {assert(false);}
    virtual void resetSignals() {assert(false);}


    //Cycle function
    virtual void cycle(){}
    
    virtual void printConfiguration(std::ofstream& out, unsigned int indent) {}  //This function prints the configuration of the ASNetwork (i.e., ASwitches configuration such as ADD_2_1, ADD_3_1, etc)
    virtual void printStats(std::ofstream& out, unsigned int indent) {} //This functions prints the statistics obtained during the execution. 
    virtual void printEnergy(std::ofstream& out, unsigned int indent){}
    

};

#endif 
