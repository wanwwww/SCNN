
#include "PoolingModule.h"
#include "PoolingUnit.h"
#include <assert.h>

PoolingModule::PoolingModule(id_t id, std::string name, Config stonne_cfg) : Unit(id,name){

    this->pooling_num = stonne_cfg.m_MSNetworkCfg.ms_cols/2; // 池化单元个数

    this->input_connection = new Connection(1);

    for(int i=0; i<this->pooling_num; i++){  // 创建各池化单元
        std::string pooling_str = "poolingUnit"+std::to_string(i);
        Connection* output_connection = new Connection(1);
        PoolingUnit* poolingUnit = new PoolingUnit(i, pooling_str, i, stonne_cfg, output_connection);
        this->poolingtable[i] = poolingUnit;
        this->outputconnectiontable.push_back(output_connection);
    }

}

PoolingModule::~PoolingModule() {

    delete this->input_connection;

    for(int i=0; i<this->pooling_num; i++){
        delete this->poolingtable[i];
        delete this->outputconnectiontable[i];
    }
}

// 接收数据,将数据发送给各个池化单元
// 只支持池化窗口为2，步长为2的卷积
void PoolingModule::receive(){
    //std::cout<<"pooling module receive data : "<<std::endl;
    if(input_connection->existPendingData()){
        //std::cout<<"yes"<<std::endl;
        std::vector<std::shared_ptr<DataPackage>> data_received = input_connection->receive();
        assert(data_received.size() == 1);
        std::vector<bool> data = data_received[0]->get_data_vector(); // 从SRAM中读取到的数据
        std::vector<bool> unpacking_data;
        // 根据池化窗口的大小将data中的数据送入到各个池化单元 （根据数据量发送给池化单元，在一些情况下，不是所有的池化单元都接收到数据）
        for(int i=0; i<data.size()/2; i++){
            unpacking_data.push_back(data[i*2]);
            unpacking_data.push_back(data[i*2+1]);
            //std::cout<<"pooling unit : "<<i<<"      data is : "<<data[i*2]<<" and "<<data[i*2+1]<<std::endl;
            std::shared_ptr<DataPackage> unpacked_data = std::make_shared<DataPackage>(unpacking_data, data_received[0]->channel_num, data_received[0]->retrieve_num); // 发送给池化单元的数据包
            poolingtable[i]->input_fifo->push(unpacked_data);
            unpacking_data.clear();
        }
    }
    //std::cout<<"pooling module receive data over"<<std::endl;
}

// 先接收数据，调用各个池化单元的cycle()方法
void PoolingModule::cycle(){
    this->receive();
    //std::cout<<this->pooling_num<<std::endl;
    for(int i=0; i<this->pooling_num; i++){
        poolingtable[i]->cycle();
    }
}

std::map<int, Connection*> PoolingModule::getOutputConnection(){

    std::map<int, Connection*> memConnection;

    for(int i=0; i<this->outputconnectiontable.size(); i++){
        memConnection[i] = outputconnectiontable[i];
    }

    return memConnection;
}

void PoolingModule::resetSignals(){
    for(int i=0; i<this->pooling_num; i++){
        this->poolingtable[i]->resetSignals();
    }
}