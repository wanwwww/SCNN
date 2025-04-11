
#include "PoolingNetwork.h"
#include <assert.h>

PoolingNetwork::PoolingNetwork(id_t id, std::string name, Config stonne_cfg) : Unit(id, name){
    
    this->stonne_cfg = stonne_cfg;
    this->port_width = stonne_cfg.m_MSwitchCfg.port_width;
    this->n_poolings = stonne_cfg.m_MSNetworkCfg.ms_rows*stonne_cfg.m_MSNetworkCfg.ms_cols;

    this->operation_mode = POOLING;

    std::string name_str = "pooling";

    for(int i=0;i<this->n_poolings;i++){
        std::string name_P = name_str+=i;
        Connection* input_connection = new Connection(this->port_width);
        PoolingSwitch* pooler = new PoolingSwitch(i, name_P, this->stonne_cfg, i, input_connection);
        this->inputconnectiontable.push_back(input_connection);
        this->poolingtable[i] = pooler;
    }
    
}

PoolingNetwork::~PoolingNetwork(){
    for(int i=0;i<this->n_poolings;i++){
        delete this->inputconnectiontable[i];
        delete this->poolingtable[i];
    }
}

void PoolingNetwork::setOutputConnection(std::vector<std::vector<Connection*>> memoryconnections){
    // 先将memoryconnections转化为向量形式，存在outputconnectiontable中
    unsigned int n_bus_lines = memoryconnections.size(); // n 条总线线
    for(int i=0;i<this->n_poolings;i++){
        unsigned int inputID = (i / n_bus_lines);
        unsigned int busID = i % n_bus_lines;
        Connection* mem_conn = memoryconnections[busID][inputID];
        outputconnectiontable.push_back(mem_conn);
        poolingtable[i]->setOutputConnection(mem_conn);
    }
}

std::vector<Connection*> PoolingNetwork::getInputConnections(){
    return this->inputconnectiontable;
}

void PoolingNetwork::configureSignals(Tile* current_tile, unsigned int ms_rows, unsigned int ms_cols){
    this->row_used = current_tile->get_T_X_() * current_tile->get_T_Y_();
    this->col_used = current_tile->get_T_K();
    assert(this->row_used <= ms_rows);
    assert(this->col_used <= ms_cols);
    this->pool_rows = ms_rows;
    this->pool_cols = ms_cols;
}

// 接收所有池化单元输入连接中的数据包
void PoolingNetwork::receive(){
    for(int i=0;i<n_poolings;i++){
        poolingtable[i]->receive();
    }
}

// 遍历每个池化域右下角的池化单元，判断池化所需的数据是否足够，如果足够则进行最大池化计算，将计算结果推入到output_fifo
void PoolingNetwork::compute(){
    //std::cout<<"calling the compute()"<<std::endl;
    //std::cout<<"this->row_used is "<<this->row_used<<std::endl;
    //std::cout<<"this->col_used is"<<this->col_used<<std::endl;
    for(int i=1;i<this->row_used;i=i+2){
        for(int j=1;j<this->col_used;j=j+2){
            unsigned num = i*this->pool_cols + j;
            //std::cout<<"the num is "<<num<<std::endl;
            if(poolingtable[num]->pool_enable() && poolingtable[num-1]->pool_enable() && poolingtable[num-pool_cols]->pool_enable() && poolingtable[num-pool_cols-1]->pool_enable()){
                //std::cout<<"pooling"<<std::endl;
                DataPackage* pck3 = poolingtable[num]->input_fifo->pop();
                DataPackage* pck2 = poolingtable[num-1]->input_fifo->pop();
                DataPackage* pck1 = poolingtable[num-pool_cols]->input_fifo->pop();
                DataPackage* pck0 = poolingtable[num-pool_cols-1]->input_fifo->pop();
                data_t result = pck0->get_data() + pck1->get_data() + pck2->get_data() + pck3->get_data();
                //std::cout<<"the add result is "<<result<<std::endl;
                if(this->stonne_cfg.pooling_type == MAXPOOLING){
                    assert(result>=0);
                    if(result > 0) {result = 1;}
                }
                else {  // 平均池化，设置阈值为2，因为目前只支持2*2的池化
                    if(result > 2) {result = 1;}
                    else {result = 0;}
                }
                DataPackage* result_spike = new DataPackage(sizeof(data_t), result, SPIKE, 0, pck3->get_vn(), this->operation_mode);
                poolingtable[num]->output_fifo->push(result_spike);
            }
            else {
                //std::cout<<"Not enough spike data"<<std::endl;
            }
        }
    }
}

void PoolingNetwork::send(){
    for(int i=0;i<n_poolings;i++){
        poolingtable[i]->send();
    }
}

void PoolingNetwork::cycle(){
    //std::cout<<"calling poolingnet cycle()"<<std::endl;
    this->receive();
    //std::cout<<"received"<<std::endl;
    this->compute();
    //std::cout<<"computed"<<std::endl;
    this->send();
    //std::cout<<"sent"<<std::endl;
}


void PoolingNetwork::printStats(std::ofstream& out, unsigned int indent){

}

void PoolingNetwork::printEnergy(std::ofstream& out, unsigned int indent){

}

void PoolingNetwork::setConfiguration(Config cfg){

}