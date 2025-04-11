
#include "PoolingSwitch.h"


PoolingSwitch::PoolingSwitch(id_t id, std::string name, Config stonne_cfg, unsigned int n_pooling, Connection* inputconnection) : Unit(id, name){
    
    this->n_pooling = n_pooling;
    this->port_width = stonne_cfg.m_MSwitchCfg.port_width;
    this->stonne_cfg = stonne_cfg;
    
    this->inputconnection = inputconnection;

    this->input_fifo = new Fifo(this->port_width);
    this->output_fifo = new Fifo(this->port_width);

    this->received_flag = !input_fifo->isEmpty();  // 如果池化单元的input_fifo不空的话，就说明该池化单元已经接收到了脉冲数据

}

PoolingSwitch::~PoolingSwitch(){
    delete this->input_fifo;
    delete this->output_fifo;
}

void PoolingSwitch::setOutputConnection(Connection* outputconnection){
    this->outputconnection = outputconnection;
}

bool PoolingSwitch::pool_enable(){
    this->received_flag = !input_fifo->isEmpty();
    return this->received_flag;
}

// 接收输入连接中的数据，将其推送到input_fifo
void PoolingSwitch::receive(){
    if(this->inputconnection->existPendingData()){
        std::vector<DataPackage*> data_received = this->inputconnection->receive();
        for(int i=0;i<data_received.size();i++){
            input_fifo->push(data_received[i]);
        }
    }
}

// 将output_fifo中的数据发送到输出连接 
void PoolingSwitch::send(){
    if(!output_fifo->isEmpty()){
        std::vector<DataPackage*> vector_to_send;
        while(!output_fifo->isEmpty()){
            DataPackage* pck = output_fifo->pop();
            vector_to_send.push_back(pck);
        }
        this->outputconnection->send(vector_to_send);
    }
}

void PoolingSwitch::cycle(){

}

void PoolingSwitch::printStats(std::ofstream& out, unsigned int indent){

}

void PoolingSwitch::printEnergy(std::ofstream& out, unsigned int indent){

}

void PoolingSwitch::setConfiguration(Config cfg){

}