
#include "NeuronStateUpdaterSwitch.h"

NeuronStateUpdaterSwitch::NeuronStateUpdaterSwitch(id_t id, std::string name, Config stonne_cfg, unsigned int n_updater, Connection* inputconnection) : Unit(id, name){
    
    this->stonne_cfg = stonne_cfg;
    this->port_width = stonne_cfg.m_MSwitchCfg.port_width;
    this->inputconnection = inputconnection;
    this->n_updater = n_updater;
    this->V_th = stonne_cfg.V_th;

    this->input_fifo = new Fifo(this->port_width);
    this->output_fifo = new Fifo(this->port_width);
    this->output_pool_fifo = new Fifo(this->port_width);

    this->operation_mode = UPDATE;

    this->layer_type = stonne_cfg.layer_type;

    this->pooling_enabled = false;
}

NeuronStateUpdaterSwitch::~NeuronStateUpdaterSwitch(){
    delete this->input_fifo;
    delete this->output_fifo;
    delete this->output_pool_fifo;
}

void NeuronStateUpdaterSwitch::setOutputConnection(Connection* outputconnection){
    this->outputconnection = outputconnection;
}

void NeuronStateUpdaterSwitch::setPoolingConnection(Connection* poolingconnection){
    this->poolingconnection = poolingconnection;
}

// void NeuronStateUpdaterSwitch::setMemConnection(Connection* memconnection){
//     this->memconnection = memconnection;
// }

void NeuronStateUpdaterSwitch::setMemController(OSMeshSDMemory* mem){
    this->mem = mem;
}

unsigned int NeuronStateUpdaterSwitch::computeAddress(std::shared_ptr<DataPackage> pck_received, OSMeshSDMemory* mem){
    unsigned int vn = pck_received->get_vn();
    // unsigned int current_tile_M_pointer = (mem->getN_iterations_completed() / mem->getIter_N()) * mem->getT_M();
    unsigned int current_tile_M_pointer = (mem->getN_iterations_completed() / mem->getIter_N()) * this->stonne_cfg.m_MSNetworkCfg.ms_rows;
    unsigned int current_tile_N_pointer = (mem->getN_iterations_completed() % mem->getIter_N()) * mem->getT_N();
    unsigned int vn_M_pointer = vn / mem->getCols_used();
    unsigned int vn_N_pointer = vn % mem->getCols_used();
    unsigned int addr = (current_tile_M_pointer+vn_M_pointer)*mem->getN() + (current_tile_N_pointer+vn_N_pointer);
    return addr;
}

void NeuronStateUpdaterSwitch::receive(){
    if(this->inputconnection->existPendingData()){
        std::vector<std::shared_ptr<DataPackage>> data_received = this->inputconnection->receive();
        for(int i=0;i<data_received.size();i++){
            input_fifo->push(data_received[i]);
        }
    }
}

// 更新膜电位，并将结果推入到输出fifo
void NeuronStateUpdaterSwitch::compute(){
    std::shared_ptr<DataPackage> pck_received;
    if(!input_fifo->isEmpty()){

        pck_received = input_fifo->pop();
        unsigned int addr = computeAddress(pck_received,this->mem);
        data_t Vth_Increment =  pck_received->get_data();  // 膜电位增量 
        data_t Vth_pre = this->mem->get_Vth(addr);  // 上一个时间步的膜电位 
        data_t Vth_current = Vth_pre + Vth_Increment;
        
        // std::cout<<"Membrane potential address : "<<addr<<std::endl;
        // std::cout<<"Membrane potential at the previous time step : "<<Vth_pre<<std::endl;
        // std::cout<<"The membrane potential increment at the current time step : "<<Vth_Increment<<std::endl;
        // std::cout<<"The accumulated membrane potential : "<<Vth_current<<std::endl;

        data_t result_V;
        data_t result_S;

        if(Vth_current >= this->V_th){
            result_V = 0;  // 重置膜电位 
            result_S = 1;  // 生成脉冲 
        }
        else{
            result_V = Vth_current;   // 更新后的膜电位 
            result_S = 0;
        }

        std::shared_ptr<DataPackage> result_Vth = std::make_shared<DataPackage>(sizeof(data_t), result_V, VTH, 0, pck_received->get_vn(), this->operation_mode);
        std::shared_ptr<DataPackage> result_Spike = std::make_shared<DataPackage>(sizeof(data_t), result_S, SPIKE, 0, pck_received->get_vn(), this->operation_mode);

        
        this->output_fifo->push(result_Vth);
        //std::cout<<"Push a Vth into the output_fifo"<<std::endl;
        
        //std::cout<<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<this->layer_type<<std::endl;
        // 在这里根据层的类型将脉冲数据推入到不同的输出fifo
        if(this->pooling_enabled){
            this->output_pool_fifo->push(result_Spike);
            //std::cout<<"Push a result into the output_pool_fifo (pooling)"<<std::endl;
        }
        else{
            this->output_fifo->push(result_Spike);
            //std::cout<<"Push a result into the output_fifo (without pooling)"<<std::endl;
        }

        // if(this->n_updater == 0){
        //     std::cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<std::endl;
        //     std::cout<<"the first updater, receive data is : "<<Vth_Increment<<std::endl;
        //     std::cout<<"the result output is : "<<result_S<<std::endl;
        //     std::cout<<"the result neuron state is : "<<result_V<<std::endl;
        //     std::cout<<"the Vth_pre is : "<<Vth_pre<<std::endl;
        //     std::cout<<"the Vth_pre addr is : "<<addr<<std::endl;
        // }

        // delete pck_received;  // 释放从归约网络接收到的数据 ?????????????????????????????????
        // pck_received = nullptr;
    }

}

// 当不执行池化操作时，脉冲数据和膜电位数据都被推入output_fifo中，然后由send()方法发送到输出连接（与总线的连接）
// 当执行池化操作时，膜电位数据被推入output_fifo中，然后由send()方法发送到输出连接（与总线的连接）；
//  脉冲数据被推入output_pool_fifo中，然后由send()方法发送到输出连接（与池化模块的连接） 
void NeuronStateUpdaterSwitch::send(){
    if(!output_fifo->isEmpty()){
        std::vector<std::shared_ptr<DataPackage>> vector_to_send_mem;
        while(!output_fifo->isEmpty()){
            std::shared_ptr<DataPackage> pck = output_fifo->pop();
            vector_to_send_mem.push_back(pck);
        }

        this->outputconnection->send(vector_to_send_mem);    

        //std::cout<<"the number of datapackage sent(NeuronStaterSwitch.cpp) : "<<vector_to_send.size()<<std::endl;
    }

    if(!output_pool_fifo->isEmpty()){
        std::vector<std::shared_ptr<DataPackage>> vector_to_send_pool;
        while(!output_pool_fifo->isEmpty()){
            std::shared_ptr<DataPackage> pck = output_pool_fifo->pop();
            vector_to_send_pool.push_back(pck);
        }

        this->poolingconnection->send(vector_to_send_pool);

    }
}

void NeuronStateUpdaterSwitch::cycle(){
    this->receive();
    this->compute();
    this->send();
}



void NeuronStateUpdaterSwitch::printStats(std::ofstream& out, unsigned int indent){

}

void NeuronStateUpdaterSwitch::printEnergy(std::ofstream& out, unsigned int indent){

}

void NeuronStateUpdaterSwitch::setConfiguration(Config cfg){

}
