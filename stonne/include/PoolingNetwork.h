
#ifndef __POOLINGNETWORK__H__
#define __POOLINGNETWORK__H__


#include "Unit.h"
#include "Connection.h"
#include "types.h"
#include "Config.h"
#include "PoolingSwitch.h"
#include <map>
#include "Tile.h"

class PoolingNetwork : public Unit{
private:
    unsigned int port_width;
    unsigned int n_poolings;

    std::vector<Connection*> inputconnectiontable;   
    std::vector<Connection*> outputconnectiontable;  // 连接总线 

    Config stonne_cfg;

    std::map<int,PoolingSwitch*> poolingtable; // 存储池化单元 

    unsigned row_used;
    unsigned col_used;

    unsigned pool_rows;
    unsigned pool_cols;

    adderoperation_t operation_mode;


public:
    PoolingNetwork(id_t id, std::string name, Config stonne_cfg);
    ~PoolingNetwork();

    void setOutputConnection(std::vector<std::vector<Connection*>> memoryconnections); // 与总线的连接 

    std::vector<Connection*> getInputConnections();

    void receive();

    void compute();

    void send();

    void cycle();

    void configureSignals(Tile* current_tile, unsigned int ms_rows, unsigned int ms_cols);

    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
    void setConfiguration(Config cfg);

};

#endif
