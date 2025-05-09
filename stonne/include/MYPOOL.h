
#ifndef __MYPOOL__H
#define __MYPOOL__H

#include "PoolingMemory.h"
#include "PoolingModule.h"
#include "Config.h"

class MYPOOL {
public:

    Config stonne_cfg;

    PoolingMemory* mem;

    PoolingModule* poolingnet;

    bool layer_loaded;
    int n_cycle;

    void connectMemoryandPooling(); // 连接内存和池化模块

    void cycle();

    MYPOOL(Config stonne_cfg);  // 构造函数
    ~MYPOOL();

    void loadPOOLLayer(int Y_, int channels, address_t input_data, address_t output_data);

    void run();

};

#endif