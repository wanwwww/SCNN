
#ifndef __NEURONSTATEUPDATERSWITCH__H__
#define __NEURONSTATEUPDATERSWITCH__H__

#include "Fifo.h"
#include "DataPackage.h"
#include "Unit.h"
#include "Connection.h"
#include "types.h"

#include "OSMeshSDMemory.h"


class NeuronStateUpdaterSwitch : public Unit {
private:

    Config stonne_cfg;
    Connection* inputconnection;
    Connection* outputconnection;

    Connection* poolingconnection;

    Layer_t layer_type;

    Fifo* input_fifo;
    Fifo* output_fifo;
    Fifo* output_pool_fifo;

    unsigned int n_updater;
    unsigned int port_width;
    int V_th;
    
    bool pooling_enabled;

    adderoperation_t operation_mode;

    OSMeshSDMemory* mem;


public:
    NeuronStateUpdaterSwitch(id_t id, std::string name, Config stonne_cfg, unsigned int n_updater, Connection* inputconnection);
    ~NeuronStateUpdaterSwitch();

    void setOutputConnection(Connection* outputconnection);
    void setPoolingConnection(Connection* poolingconnection);

    void setMemController(OSMeshSDMemory* mem);

    unsigned int computeAddress(std::shared_ptr<DataPackage> pck_received, OSMeshSDMemory* mem);
    
    void receive();
    void compute();
    void send();

    void cycle();

    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
    void setConfiguration(Config cfg);

};




#endif
