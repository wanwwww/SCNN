
#ifndef __DRAMMODEL__H
#define __DRAMMODEL__H

#include "dramsim3.h"
#include <iostream>
#include "Fifo.h"

class Dram{

public:

    dramsim3::MemorySystem* dram; // 成员变量

    Fifo* read_request_fifo;
    Fifo* write_request_fifo;

    // 构造函数
    Dram(std::function<void(uint64_t)> read_callback, std::function<void(uint64_t)> write_callback);

    ~Dram();

    void run();
    void run_1(int layer_id, int j, int completed_reads);

    void set_read_request_fifo(Fifo* read_request_fifo){this->read_request_fifo = read_request_fifo;}
    void set_write_request_fifo(Fifo* write_request_fifo){this->write_request_fifo = write_request_fifo;}

};

#endif