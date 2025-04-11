// Created by Francisco Munoz on 28/02/2019 


#ifndef _BUS_CPP
#define _BUS_CPP

#include <iostream>
#include <vector>
#include "Unit.h"
#include "Connection.h"
#include "CollectionBusLine.h"
#include "Unit.h"
#include "Config.h"


// 一个总线组件，负责管理输入连接和输出连接，并控制数据在总线内的流动 
// 总线可以看作是一个整体，由多条“总线线”组成，每条线负责特定类型的数据或信号传输
// 总线的三种主要类型：数据总线、地址总线、控制总线 

class Bus : public Unit { 

private:
    unsigned int n_bus_lines;  //Number of outputs from the bus  总线中的“总线线”数量 
    unsigned int input_ports_bus_line;  // 每条总线线的输入端口数 
    unsigned int connection_width; // 数据传输的位宽 
    unsigned int fifo_size;  // 每条总线线的缓冲区大小 
    std::vector<CollectionBusLine*> collection_bus_lines; // 包含所有 CollectionBusLine 对象，用于管理输入和输出的交互。
    
public:
    
    Bus(id_t id, std::string name, Config stonne_cfg);
    
    // 获取总线中总线线的个数 
    unsigned int getNBusLines()    {return this->n_bus_lines;}
    unsigned int getInputPortsBusLine()   {return this->input_ports_bus_line;}

    std::vector<std::vector<Connection*>> getInputConnections();  // 返回总线的所有输入连接 
    std::vector<Connection*> getOutputConnections(); //Get the output connections of all the lines
    // 获取特定的连接 
    Connection* getInputConnectionFromBusLine(unsigned int busID, unsigned int inputID); //Get a specific inpur from a specific bus line


    void cycle(); //Get the inputs and send as many as posssible to the outputs
  
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
    ~Bus();
   
};

#endif
