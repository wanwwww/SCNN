
//Created by Francisco Munoz Martinez on 25/06/2019

// This class is used in the simulator in order to limit the size of the fifo.

#ifndef __Fifo_h__
#define __Fifo_h__

#include <memory>
#include <queue>
#include "DataPackage.h"
#include "types.h"
#include "Stats.h"

// 用于模拟有限容量的FIFO队列
class Fifo {
private:
    std::queue<std::shared_ptr<DataPackage>> fifo;
    
    unsigned int capacity; //Capacity in number of bits
    unsigned int capacity_words; //Capacity in number of words allowed. i.e., capacity_words = capacity / size_word
    FifoStats fifoStats; //Tracking parameters
public:
    Fifo(unsigned int capacity);

    bool isEmpty();
    bool isFull();
    void push(std::shared_ptr<DataPackage> data);
    std::shared_ptr<DataPackage> pop();
    std::shared_ptr<DataPackage> front();
    unsigned int size(); //Return the number of elements in the fifo
    void printStats(std::ofstream& out, unsigned int indent);
    void printEnergy(std::ofstream& out, unsigned int indent);
};
#endif
