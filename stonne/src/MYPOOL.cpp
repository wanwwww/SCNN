
#include "MYPOOL.h"

MYPOOL::MYPOOL(Config stonne_cfg){
    this->stonne_cfg = stonne_cfg;

    // PoolingMemory(id_t id, std::string name, Config stonne_cfg);
    this->mem = new PoolingMemory(0, "PoolingMem", stonne_cfg);

    this->poolingnet = new PoolingModule(1,"PoolingNet", stonne_cfg);

    this->layer_loaded = false;
    this->n_cycle = 0;

    this->mem->setPoolingModule(poolingnet);

    this->connectMemoryandPooling();
}

MYPOOL::~MYPOOL(){
    delete this->mem;
    delete this->poolingnet;
}

void MYPOOL::loadPOOLLayer(int Y_, int channels, address_t input_data, address_t output_data){
    this->mem->setLayer(Y_, channels, input_data, output_data);
    this->layer_loaded = true;
}

void MYPOOL::connectMemoryandPooling(){
    Connection* MemReadConnection = this->poolingnet->getInputConnection();
    std::map<int, Connection*> MemWriteConnection = this->poolingnet->getOutputConnection();

    this->mem->setReadConnection(MemReadConnection);
    this->mem->setWriteConnection(MemWriteConnection);
}

void MYPOOL::run(){
    this->cycle();
}

void MYPOOL::cycle(){
    //std::cout<<"run()"<<std::endl;
    bool exection_finished = false;
    while(!exection_finished){
        //std::cout<<"-------------------"<<this->n_cycle<<"-------------------"<<std::endl;
        // std::cout<<std::endl;
        //std::cout<<"cycle: "<<this->n_cycle<<std::endl;
        // std::cout<<"current state : "<<this->mem->current_state<<std::endl;
        this->mem->cycle();
        //std::cout<<"mem over"<<std::endl;
        this->poolingnet->cycle();
        //std::cout<<"pool over"<<std::endl;

        exection_finished = this->mem->execution_finished;
        // std::cout<<"exection_finished : "<<exection_finished<<std::endl;
        this->n_cycle++;
    }
}

