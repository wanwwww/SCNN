
#ifndef __NEURONSTATEUPDATE__H__
#define __NEURONSTATEUPDATE__H__


#include "Unit.h"
#include "Connection.h"
#include "types.h"
#include "Config.h"
#include "Tile.h"
#include "DNNLayer.h"
#include "NeuronStateUpdaterSwitch.h"

#include "MemoryController.h"
#include "OSMeshSDMemory.h"



class NeuronStateUpdater : public Unit {

private:
    unsigned int port_width;
    unsigned int n_updaters; // 所需更新开关的个数

    std::vector<Connection*> inputconnectiontable;   
    std::vector<Connection*> outputconnectiontable;  // 不用实例化，设置为总线的输入连接

    std::vector<Connection*> poolingconnectiontable; // 不用例化，设置为池化模块的输入连接

    std::map<int, NeuronStateUpdaterSwitch* > updatertable; 

    Config stonne_cfg;


public:
    NeuronStateUpdater(id_t id, std::string name, Config stonne_cfg);
    ~NeuronStateUpdater();

    void setOutputConnections(std::vector<std::vector<Connection*>> memoryconnections);

    void setPoolingconnections(std::vector<Connection*> poolingconnections);

    std::vector<Connection*> getInputConnections();

    void setMemoryController(OSMeshSDMemory* mem);

    void cycle();

    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
    void setConfiguration(Config cfg);
};

#endif