//Created 13/06/2019

#include "DataPackage.h"
#include <assert.h>
#include <string.h>

//General constructor implementation

DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source) {
    this->size_package = size_package;
    this->data = data;
    this->data_type =data_type;
    this->source = source;
    this->traffic_type = UNICAST; //Default
}

// -------------------------------------------------------- add
DataPackage::DataPackage(std::vector<bool> data, int channel_num, int retrieve_num){
    this->data_vector = data;
    this->channel_num = channel_num;
    this->retrieve_num = retrieve_num;
}
DataPackage::DataPackage(bool data, int channel_num, int retrieve_num, int location){
    this->data_result = data;
    this->channel_num = channel_num;
    this->retrieve_num = retrieve_num;
    this->location = location;
}
// -------------------------------------------------------- add

DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type) : DataPackage(size_package, data, data_type, source) {
    this->traffic_type = traffic_type;
    this->dests = NULL;
}

// Unicast package constructor. // 单播数据包
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, unsigned int unicast_dest) : 
DataPackage(size_package, data, data_type, source, traffic_type) {
    //std::cout<<"new a datapackage"<<std::endl;
    assert(traffic_type == UNICAST);
    this->unicast_dest = unicast_dest;
}
//Multicast package. dests must be dynamic memory since the array is not copied. 
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, bool* dests, unsigned int n_dests) : DataPackage(size_package, data, data_type, source, traffic_type) {
    this->dests = dests;
    this->n_dests = n_dests;
}

//psum package  部分和数据包 
DataPackage::DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source, unsigned int VN, adderoperation_t operation_mode): DataPackage(size_package, data, data_type, source) {
    this->VN = VN;
    this->operation_mode = operation_mode;
}

void DataPackage::setOutputPort(unsigned int output_port) {
    this->output_port = output_port;
}

void DataPackage::setIterationK(unsigned int iteration_k) {
    this->iteration_k = iteration_k;
}

//Copy constructor
DataPackage::DataPackage(std::shared_ptr<DataPackage> pck) {
    this->size_package = pck->get_size_package();
    this->data = pck->get_data();
    this->data_type = pck->get_data_type();
    this->source = pck->get_source();
    this->traffic_type = pck->get_traffic_type();
    this->unicast_dest = pck->get_unicast_dest();
    this->VN = pck->get_vn();
    this->operation_mode = pck->get_operation_mode();
    this->output_port = output_port;
    this->iteration_k=pck->getIterationK();
    if(this->traffic_type == MULTICAST) {
        this->n_dests = pck->get_n_dests();  
        const bool* prev_pck_dests = pck->get_dests();
        this->dests = new bool[this->n_dests]; // 分配内存 
        //for(int i=0; i<n_dests; i++) {
        //    this->dests[i]=prev_pck_dests[i];
        //}
        // 复制多播数组
        // 复制prev_pck_dests数组的内容到当前对象的dests数组中
        memcpy(this->dests, prev_pck_dests, sizeof(bool)*this->n_dests);

    }

}

// DataPackage::~DataPackage() {
  
//     if(this->traffic_type==MULTICAST) {
//         delete[] dests;
//     }
// }


