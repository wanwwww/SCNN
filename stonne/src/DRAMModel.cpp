
#include "DRAMModel.h"
#include "RequestPackage.h"
#include <assert.h>

Dram::Dram(std::function<void(uint64_t)> read_callback, std::function<void(uint64_t)> write_callback){

    std::string config_file = "/home/zww/DRAMsim3/configs/DDR4_8Gb_x8_3200.ini";
    std::string output_dir = "/home/zww/DRAMsim3/output";       

    this->dram = dramsim3::GetMemorySystem(config_file,output_dir,read_callback,write_callback);
}

Dram::~Dram(){
    delete dram;
}

void Dram::run(){

    if(!this->read_request_fifo->isEmpty()){  // 如果请求队列不空，则判断dram是否可以接收读请求事务
         // 返回列表的第一个请求，但不移除该请求，因为不确定dram能否处理该请求。
        RequestPackage* read_request = static_cast<RequestPackage*>(this->read_request_fifo->front()); 
        uint64_t addr = read_request->get_addr();
        bool write = read_request->get_request_type();
        assert(!write); // 这里是读请求
        if(this->dram->WillAcceptTransaction(addr,write)){
            if(this->dram->AddTransaction(addr,write)){
                //std::cout << "Added read transaction at address: 0x" << std::hex << addr <<std::endl;
                this->read_request_fifo->pop(); // 读请求发送给dram之后，将该请求移除
                delete read_request;
            } else{
                //std::cout << "Failed to add read transaction!!!" <<std::endl;
            }
        } else {
            //std::cout << "DRAM not ready to accept new transaction." << std::endl;  // 等待下一周期
        }
    }

    if(!this->write_request_fifo->isEmpty()){
        RequestPackage* write_request = static_cast<RequestPackage*>(this->write_request_fifo->front());  // 返回列表的第一个请求，但不移除该请求，因为不确定dram能否处理该请求。
        uint64_t addr = write_request->get_addr();
        bool write = write_request->get_request_type();
        assert(write); // 这里是写请求
        if(this->dram->WillAcceptTransaction(addr,write)){
            if(this->dram->AddTransaction(addr,write)){
                //std::cout << "Added write transaction at address: 0x" << std::hex << addr <<std::endl;
                this->write_request_fifo->pop(); // 读请求发送给dram之后，将该请求移除
                delete write_request;
            } else{
                //std::cout << "Failed to add write transaction!!!" <<std::endl;
            }
        } else {
            //std::cout << "DRAM not ready to accept new transaction." << std::endl;  // 等待下一周期
        }
    }

    this->dram->ClockTick();

}