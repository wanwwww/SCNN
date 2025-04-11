
#ifndef __POOLINGSWITCH__H__
#define __POOLINGSWITCH__H__

#include "Connection.h"
#include "Fifo.h"
#include "Unit.h"
#include "DataPackage.h"

class PoolingSwitch : Unit{
private:
    Connection* inputconnection;
    Connection* outputconnection;

    unsigned int n_pooling;
    unsigned int port_width;

    Config stonne_cfg;

    bool received_flag; // 该池化单元是否接收到脉冲 


public:

    Fifo* input_fifo;
    Fifo* output_fifo;

    PoolingSwitch(id_t id, std::string name, Config stonne_cfg, unsigned int n_pooling, Connection* inputconnection);
    ~PoolingSwitch();

    void setOutputConnection(Connection* outputconnection);

    bool pool_enable();

    void receive();

    void send();

    void cycle();
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
    void setConfiguration(Config cfg);
};

#endif
