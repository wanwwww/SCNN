
#ifndef __request_package_h__
#define __request_package_h__

//#include <iostream>
#include "DataPackage.h"

class RequestPackage : public DataPackage{

public:

    uint64_t addr;
    bool write;

    RequestPackage(uint64_t addr,bool write);
    ~RequestPackage();

    uint64_t get_addr() {return this->addr;}
    bool get_request_type() {return this->write;}

};

#endif