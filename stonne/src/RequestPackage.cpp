
#include "RequestPackage.h"
#include "types.h"


RequestPackage::RequestPackage(uint64_t addr,bool write) : DataPackage(size_package, data, data_type, source){
    this->addr = addr;
    this->write = write;
}

RequestPackage::~RequestPackage(){

}