
#include "PoolingMemory.h"
#include <assert.h>
#include <math.h>

PoolingMemory::PoolingMemory(id_t id, std::string name, Config stonne_cfg) : Unit(id, name) {

    this->num_unit = stonne_cfg.m_MSNetworkCfg.ms_rows / 2;
    this->sram_bus_width = stonne_cfg.m_MSNetworkCfg.ms_rows;
    this->num_row = 2; // 2*2的池化

    this->current_num_channels = 0;  //当前处理的输出通道
    this->current_num_retrieve = 0;  // 当前处理的输出通道中的第几次从sram取出的数据
    this->current_num_row = 0;  // 当前处理的当前通道的第几行（2*2的池化一共需要处理三行）
    
    this->current_state = POOLINGMODULE_CONFIGURING; // 初始状态为配置状态

    this->input_fifo = new Fifo(1);

    for(int i=0; i<this->num_unit; i++){
        Fifo* write_fi = new Fifo(1);
        this->write_fifos.push_back(write_fi);
    }

    this->layer_loaded = false;

    this->required_output = 0;
    this->current_output = 0;

    this->execution_finished = false;

}

PoolingMemory::~PoolingMemory(){
    delete this->input_fifo;
    for(int i=0; i<this->num_unit; i++){
        delete write_fifos[i];
    }
}

void PoolingMemory::setLayer(int Y_, int channels, address_t input_data, address_t output_data) {
    this->num_channels = channels;
    this->Y_ = Y_;
    this->on_chip_sram = input_data;
    this->output_regs = output_data;
    this->num_retrieve = std::ceil(this->Y_ / (float)this->sram_bus_width);
    this->layer_loaded = true;
}

void PoolingMemory::receive(){
    for(int i=0; i<write_connection.size(); i++){
        if(write_connection[i]->existPendingData()){
            std::shared_ptr<DataPackage> data_received = write_connection[i]->receive_pooling_result();
            write_fifos[i]->push(data_received);
        }
    }
}

// 将从SRAM中取出的数据发送到池化模块
void PoolingMemory::send() {
    std::vector<std::shared_ptr<DataPackage>> pck_to_send; 
    if(!this->input_fifo->isEmpty()) {
        std::shared_ptr<DataPackage> pck = input_fifo->pop();
        pck_to_send.push_back(pck);
        this->read_connection->send(pck_to_send);
    }
}

// 初始状态为配置状态，配置池化模块
// 然后进入数据分发状态，每一个cycle从SRAM中去取出一次数据发送到池化模块
// 同时每个周期从池化模块接收数据
void PoolingMemory::cycle() {

    assert(this->layer_loaded);

    if(this->current_state == POOLINGMODULE_CONFIGURING) {
        // 配置阶段
        this->pooling_module->resetSignals();
        this->required_output = this->num_channels * (this->Y_/2);
    }

    if(this->current_state == POOLING_DATA_DIST) {
        //std::cout<<"num_retrieve : "<<this->num_retrieve<<std::endl;
        // 数据分发状态
        // 根据当前取数据的情况，计算出要从SRAM中取出数据的地址，将从SRAM中取出连续的数据送入到input_fifo中
        std::vector<bool> data;
        int address = this->current_num_channels*2*this->Y_ + this->current_num_row*this->Y_ + this->current_num_retrieve*this->sram_bus_width;
        //std::cout<<"Distribute data, address is : "<<address<<std::endl;
        int data_length;
        //std::cout<<"current_num_retrieve : "<<this->current_num_retrieve<<std::endl;
        if(this->current_num_retrieve == this->num_retrieve-1){ // 一行中最后一次取数据，不一定是完整的数据，要计算出需要取出数据的个数
            data_length = this->Y_ - this->sram_bus_width*this->current_num_retrieve;
        } else {
            data_length = this->sram_bus_width;
        }
        //std::cout<<"Y_ : "<<Y_<<std::endl;
        // std::cout<<"address : "<<address<<std::endl;
        //std::cout<<"data_length : "<<data_length<<std::endl;

        // std::cout<<"data is : ";
        for(int i=0; i<data_length; i++){
            data.push_back(this->on_chip_sram[address+i]);
            //std::cout<<this->on_chip_sram[address+i]<<"  ";
        }

        // std::cout<<std::endl;

        std::shared_ptr<DataPackage> pck_to_send = std::make_shared<DataPackage>(data, this->current_num_channels, this->current_num_retrieve);
        this->input_fifo->push(pck_to_send);

        // 计数器计算
        this->current_num_row++;

        if(this->current_num_row == this->num_row){
            this->current_num_row = 0;
            this->current_num_retrieve++;
            if(this->current_num_retrieve == this->num_retrieve){
                this->current_num_retrieve = 0;
                this->current_num_channels++;
                if(this->current_num_channels == this->num_channels){
                    this->current_num_channels = 0;
                    this->current_state = POOLING_ALL_DATA_SENT;
                }
            }
        }

    }

    // 接收池化结果，将池化结果存取输出寄存器组
    this->receive();

    for(int i=0; i<this->num_unit; i++){
        if(!write_fifos[i]->isEmpty()){  // 将池化结果
            // DataPackage(bool data, int channel_num, int retrieve_num, int location);
            std::shared_ptr<DataPackage> pck_received = write_fifos[i]->pop();
            bool data = pck_received->data_result;
            int channel_num = pck_received->channel_num;
            int retrieve_num = pck_received->retrieve_num;
            int location = pck_received->location;
            int addr = channel_num*this->Y_/2 + retrieve_num*this->num_unit + location;
            this->output_regs[addr] = data;

            //std::cout<<"Write data, address is : "<<addr<<"   "<<std::endl;

            // std::cout<<std::endl;
            // std::cout<<"pooling unit : "<<i<<"  write data is : "<<data<<std::endl;
            // std::cout<<"write addr is : "<<addr<<std::endl;
            // std::cout<<std::endl;

            current_output++; 

            // std::cout<<"required_output : "<<this->required_output<<std::endl;
            // std::cout<<"current_output : "<<this->current_output<<std::endl;

            if(this->current_output == this->required_output) {
                this->execution_finished = true;
            }
        }
    }

    if(this->current_state == POOLINGMODULE_CONFIGURING){
        this->current_state = POOLING_DATA_DIST;
    }
    // 发送数据到池化模块
    this->send();

}
