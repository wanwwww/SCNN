
#include "PoolingUnit.h"
#include <assert.h>

PoolingUnit::PoolingUnit(id_t id, std::string name, int location, Config stonne_cfg, Connection* output_connection) : Unit(id,name){

    this->location = location; 

    this->operate_num = 2;
    this->current_num = 0;

    this->result = 0;

    this->input_fifo = new Fifo(1);
    this->output_fifo = new Fifo(1);

    this->output_connection = output_connection;

}

PoolingUnit::~PoolingUnit() {
    delete this->input_fifo;
    delete this->output_fifo;
}

void PoolingUnit::resetSignals(){
    this->current_num = 0;
    this->operate_num = 2;
    this->result = 0;
}

// 1. 接收数据 （在上层模块中接收数据，然后将数据解包发送给池化单元）
// 2. 执行OR操作
// 3. 发送数据
void PoolingUnit::cycle() {
    //std::cout<<"poolingUnit cycle()"<<std::endl;
    // 1. 接收数据
    std::shared_ptr<DataPackage> pck_received;  // 计算结束之后delete

    if(!this->input_fifo->isEmpty()){
        //std::cout<<"debug1"<<std::endl;
        pck_received = this->input_fifo->front();
        std::vector<bool> data_received = pck_received->get_data_vector();
        assert(data_received.size() == 2);

        // 2. 执行OR操作
        // std::cout<<"befor OR the result is : "<<this->result<<std::endl;
        // std::cout<<"pooling unit : "<<this->location<<"      data is : "<<data_received[0]<<" and "<<data_received[1]<<std::endl;
        this->result = this->result || data_received[0] || data_received[1];
        // std::cout<<"after OR the result is : "<<this->result<<std::endl;

        //    判断是否计算结束
        if(this->current_num == (this->operate_num-1)){
            std::shared_ptr<DataPackage> pck_result = std::make_shared<DataPackage>(this->result, pck_received->channel_num, pck_received->retrieve_num, this->location);
            this->output_fifo->push(pck_result);
            this->current_num = 0;
            this->result = 0;
        } else {
            this->current_num++;
        }
        //std::cout<<"debug2"<<std::endl;

        // delete pck_received;
        this->input_fifo->pop();
        // delete pck_received; // ???????????????????????????????????????????????????????????????????????????????????
        //std::cout<<"debug3"<<std::endl;
    }

    //std::cout<<"debug2"<<std::endl;
    // 3. 发送数据
    if(!this->output_fifo->isEmpty()){
        //std::cout<<"debug4"<<std::endl;
        // std::cout<<"----------------------------------------------------"<<std::endl;
        //std::cout<<"debug"<<std::endl;
        std::shared_ptr<DataPackage> pck_result = this->output_fifo->pop(); 
        this->output_connection->send_pooling_result(pck_result); // 这个数据包被写入内存之后delete
        // std::cout<<"pooling unit : "<<this->location<<"   send data is : "<<pck_result->data_result<<std::endl;
        // std::cout<<"----------------------------------------------------"<<std::endl;


    }
    //std::cout<<"debug5"<<std::endl;
}