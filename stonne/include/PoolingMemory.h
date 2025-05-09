
#ifndef __PoolingMemory__h
#define __PoolingMemory__h

#include "Connection.h"
#include "Unit.h"
#include "Config.h"
#include "types.h"
#include "Fifo.h"
#include <map>
#include "PoolingModule.h"

class PoolingMemory : Unit {
public:

    int num_unit;  // 池化单元个数

    int num_channels; // 输出通道个数，脉动阵列中用到的列数
    int num_retrieve; // 一行需要取出数据的次数，由计算得到
    int num_row; // 一个输出通道内的数据行数，也就是池化窗口的行数
    int sram_bus_width; // 一次从SRAM中取出的数据量，单位为bit，在一个输出通道内，最后一次从SRAM取出的数据可能不全是有效数据
    int Y_;

    int current_num_channels;
    int current_num_retrieve;
    int current_num_row;

    int required_output;
    int current_output;

    bool execution_finished;

    address_t on_chip_sram; // 存储池化的输入
    address_t output_regs; // 暂存池化的输出

    PoolingMemoryControllerState current_state;

    bool layer_loaded; 

    PoolingModule* pooling_module;

    Fifo* input_fifo;
    std::vector<Fifo*> write_fifos;

    // 内存的读写连接
    Connection* read_connection;
    std::map<int, Connection*> write_connection;

    PoolingMemory(id_t id, std::string name, Config stonne_cfg);
    ~PoolingMemory();

    void setReadConnection(Connection* read_connection) {this->read_connection = read_connection; };
    void setWriteConnection(std::map<int, Connection*> write_connection) {this->write_connection = write_connection; };

    void receive(); // 接收数据，存储到寄存器组中 
    void send();

    void cycle();

    void setLayer(int Y_, int channels, address_t input_data, address_t output_data);

    void setPoolingModule(PoolingModule* pooling_module) {this->pooling_module = pooling_module;};

};

#endif

