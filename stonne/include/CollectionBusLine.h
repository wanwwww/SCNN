// Created the 4th of november of 2019 by Francisco Munoz Martinez

#ifndef __CollectionBusLine__h__
#define __CollectionBusLine__h__

#include "Fifo.h"
#include "Connection.h"
#include <iostream>
#include "Unit.h"
#include "Stats.h"

// 表示总线中的一条总线线
// 每条总线线负责从多个输入端口接收数据，并通过一个输出端口将数据传输到目标设备
// 每条总线线负责管理多个输入端口及其对应的 FIFO 队列，并通过其唯一的输出端口将数据传输出去

class CollectionBusLine : public Unit {

private: 
    // 表示总线线的输入端口数
    unsigned int input_ports;  //Number of input connections that correspond with input_connections.size() and input_fifos.size()
    // 存储与总线线相连的所有输入连接
    std::vector<Connection*> input_connections; //Every input connection for this bus line
    // 每个输入连接都有一个对应的FIFO队列，用于暂存输入数据 
    std::vector<Fifo*> input_fifos; //Every fifo corresponds with an inputConnection for this busLine
    // 表示总线线的输出连接，用于将数据发送到下一级组件 
    Connection* output_port;      //Output connection with memory
    // 记录下次从哪个输入端口读取数据,使用轮询（Round-Robin）策略选择输入
    unsigned int next_input_selected; //Using RR policy
    // 该总线线的编号 
    unsigned int busID;  //Output port ID of this line

    // 负责从输入连接接收数据，并将其存储在对应的FIFO中 
    void receive();

    // 用于记录统计信息的对象 
    CollectionBusLineStats collectionbuslineStats; //To track information

public: 
    //Getters useful to make the connections with the ART switches and the memory
    std::vector<Connection*> getInputConnections() {return this->input_connections;} 
    Connection* getOutputPort() {return this->output_port;}
    Connection* getInputPort(unsigned int inputID); // 返回特定端口的输入连接
    
    //Creates the input_connections, the input_fifos and the output_port
    CollectionBusLine(id_t id, std::string name, unsigned int busID, unsigned int input_ports_bus_line, unsigned int connection_width, unsigned int fifo_size);
    ~CollectionBusLine(); //Destroy connection, fifos, and output connection

    // 模拟一个时钟周期。从当前选中的输入端口中读取数据，并通过输出端口发送数据 
    void cycle(); //Select one input and send it trough the output

    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
   
    

};





#endif
