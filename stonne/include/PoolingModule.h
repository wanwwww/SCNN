
#ifndef __PoolingModule__h
#define __PoolingModule__h

#include "Unit.h"
#include "Connection.h"
#include "PoolingUnit.h"
#include <map>


class PoolingModule : public Unit {

public: 

    unsigned int pooling_num; // 有多少个池化单元
    std::map<int,PoolingUnit*> poolingtable; // 存储每个位置的池化单元
    Connection* input_connection; // 用于接收从SRAM中读取的数据

    std::vector<Connection*> outputconnectiontable; // 池化模块的输出连接，与内存相连

    PoolingModule(id_t id, std::string name, Config stonne_cfg);  // 构造函数
    ~PoolingModule();  

    Connection* getInputConnection() {return this->input_connection;};
    std::map<int, Connection*> getOutputConnection();

    void cycle();
    void receive();

    void resetSignals();
    
};

#endif
