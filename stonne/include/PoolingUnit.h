
#ifndef __PoolingUnit__h
#define __PoolingUnit__h

#include "Fifo.h"
#include "Config.h"
#include "Unit.h"
#include "Connection.h"


class PoolingUnit : public Unit {
    
public:

    int location; // 该池化单元在池化模块中的位置
    int operate_num;
    int current_num;
    bool result; // 存储池化的结果

    Fifo* input_fifo;
    Fifo* output_fifo;

    Connection* output_connection; // 与输出寄存器组的连接（即与内存的连接）


    PoolingUnit(id_t id, std::string name, int location, Config stonne_cfg, Connection* output_connection);
    ~PoolingUnit();

    void cycle();
    void resetSignals();


};

#endif