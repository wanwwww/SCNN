#include "Fifo.h"
#include <assert.h>
#include <iostream>
#include "utility.h"

// 构造函数
Fifo::Fifo(unsigned int capacity) {
    this->capacity = capacity;
    this->capacity_words = capacity / sizeof(data_t); //Data size
    // data_t被定义为float类型，sizeof(data_t) 的值等同于 sizeof(float)。在大多数现代系统中，float 通常占用 4 字节（32 位）。
}

bool Fifo::isFull() {
    //this->capacity_words 表示队列的最大容量，以“words”为单位 
    return  this->fifo.size() >= this->capacity_words;  // > is forbidden
}

bool Fifo::isEmpty() {
    return this->fifo.size()==0;
}

void Fifo::push(std::shared_ptr<DataPackage> data) {
//    assert(!isFull());  //The fifo must not be full
    fifo.push(data); //Inserting at the end of the queue
    if(this->size() > this->fifoStats.max_occupancy) {
        this->fifoStats.max_occupancy = this->size();
    }
    this->fifoStats.n_pushes+=1; // To track information
    
}

std::shared_ptr<DataPackage> Fifo::pop() {
    assert(!isEmpty());
    this->fifoStats.n_pops+=1; //To track information
    std::shared_ptr<DataPackage> pck = fifo.front(); //Accessing the first element of the queue
    fifo.pop(); //Extracting the first element
    return pck; 
}

std::shared_ptr<DataPackage> Fifo::front() {
    assert(!isEmpty());
    std::shared_ptr<DataPackage> pck = fifo.front();
    this->fifoStats.n_fronts+=1; //To track information
    return pck;
}

unsigned int Fifo::size() {
    return fifo.size();
}

void Fifo::printStats(std::ofstream& out, unsigned int indent) {
    this->fifoStats.print(out, indent);
}

void Fifo::printEnergy(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "FIFO PUSH=" << fifoStats.n_pushes; //Same line
    out << ind(indent) << " POP=" << fifoStats.n_pops;  //Same line
    out << ind(indent) << " FRONT=" << fifoStats.n_fronts << std::endl; //New line 
}


