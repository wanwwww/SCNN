//Created 19/06/2019 by Francisco Munoz-Martinez

#include "Accumulator.h"
#include <iostream>
#include <assert.h>
#include <math.h>
#include "DataPackage.h"
#include "utility.h"

using namespace std;

/* This class represents an accumulator module. */
// 从inputConnection接收数据包，存入input_fifo
// 从input_fifo中提取数据，执行累加操作，若是当前累积的第一个数据包，则直接存入暂存寄存器，否则与暂存寄存器中的值进行累加，
//  若完成所有累加操作，将结果存入output_fifo
// 将output_fifo中的数据通过outputConnection发送到下游模块 

Accumulator::Accumulator(id_t id, std::string name, Config stonne_cfg, unsigned int n_accumulator)  : Unit(id, name) {

    this->n_accumulator = n_accumulator; // 累加器编号 
    this->input_ports = stonne_cfg.m_ASwitchCfg.input_ports;  // 2
    this->output_ports = stonne_cfg.m_ASwitchCfg.output_ports;  // 1

    //Collecting parameters from the configuration file
    this->buffers_capacity = stonne_cfg.m_ASwitchCfg.buffers_capacity; // 256
    this->port_width = stonne_cfg.m_ASwitchCfg.port_width;  // 16
    this->latency = stonne_cfg.m_ASwitchCfg.latency;  // 1
    //End collecting parameters from the configuration file

    this->current_capacity = 0;
    this->inputConnection = NULL;
    this->outputConnection = NULL;
    this->input_fifo = new Fifo(this->buffers_capacity);
    this->output_fifo = new Fifo(this->buffers_capacity);
    this->local_cycle=0;
    this->current_psum=0;
    this->n_psums=0;
    this->operation_mode=ADDER;
   
}


Accumulator::~Accumulator() {
    delete this->input_fifo;
    delete this->output_fifo;
}

// 设置本次累积操作需要处理的部分和数量
void Accumulator::setNPSums(unsigned int n_psums) {
	this->n_psums=n_psums;
	this->accumulatorStats.n_configurations++; //To track the stats
}


void Accumulator::resetSignals() {
    this->current_psum=0;
    this->operation_mode=ADDER;
    this->n_psums=0;
}


//Connection setters

void Accumulator::setInputConnection(Connection* inputConnection) {
    this->inputConnection = inputConnection;
}


void Accumulator::setOutputConnection(Connection* outputConnection) {
    this->outputConnection = outputConnection;
}


//Configuration settings (control signals)
// 将output_fifo中的数据发送到输出连接中 
void Accumulator::send() {
    if(!output_fifo->isEmpty()) {
        std::vector<std::shared_ptr<DataPackage>> vector_to_send_parent;
        while(!output_fifo->isEmpty()) {
                std::shared_ptr<DataPackage> pck = output_fifo->pop();
                vector_to_send_parent.push_back(pck);
        }

    //Sending if there is something
        #ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ACCUMULATOR_FUNC] Cycle " << local_cycle << ", Accumulator " << this->n_accumulator << " has sent a psum to memory (FORWARDING DATA)" << std::endl;
        #endif

        this->accumulatorStats.n_memory_send++;
        // 这个输出连接是与总线的连接 
        this->outputConnection->send(vector_to_send_parent); //Send the data to the corresponding output
    }
    
}

// 接收输入连接的数据到input_fifo
void Accumulator::receive() { 
    if(this->inputConnection->existPendingData()) { //If there is data to receive on the left
    	std::vector<std::shared_ptr<DataPackage>> data_received = this->inputConnection->receive(); //Copying the data to receive
	    this->accumulatorStats.n_receives++;    //To track the stats
        for(int i=0; i<data_received.size(); i++) {

            #ifdef DEBUG_ASWITCH_FUNC
            std::cout << "[ACCUMULATOR_FUNC] Cycle " << local_cycle << ", Accumulator " << this->n_accumulator << " has received a psum" << std::endl;
            #endif

            input_fifo->push(data_received[i]); //Inserting to the local queuqe from connection
        }
    }

    return;
}

//Perform operation based on the parameter this->operation_mode
std::shared_ptr<DataPackage> Accumulator::perform_operation_2_operands(std::shared_ptr<DataPackage> pck_left, std::shared_ptr<DataPackage> pck_right) {
    //Extracting the values
    assert(pck_left->get_vn() == pck_right->get_vn()); // vn must fit
    
    data_t result; // Result of the operation
    switch(this->operation_mode) {
        case ADDER: //SUM
            result = pck_left->get_data() +  pck_right->get_data();
	        this->accumulatorStats.n_adds++;      //To track the stats
            //this->aswitchStats.n_2_1_sums++;  //Track the information
            #ifdef DEBUG_ASWITCH_FUNC
                std::cout << "[ACCUMULATOR] Cycle " << local_cycle << ", Accumulator " << this->n_accumulator << " has performed an accumulation operation" << std::endl;
            #endif

            break;
        default:
            assert(false); // This case must not occur in this type of configuration adder
    }

    //Creating the result package with the output
    std::shared_ptr<DataPackage> result_pck = std::make_shared<DataPackage>(sizeof(data_t), result, PSUM, 0, pck_left->get_vn(), this->operation_mode);  //TODO the size of the package corresponds with the data size
    //Adding to the creation list to be deleted afterward
//    this->psums_created.push_back(result_pck);
    return result_pck;
     
}

void Accumulator::route() {
    std::shared_ptr<DataPackage> pck_received;
    if(!input_fifo->isEmpty()) {
        pck_received = input_fifo->pop();
        std::shared_ptr<DataPackage> result;
        if(current_psum == 0) {  //There is no package yet to sum in this iteration
            //Creating package 0
            this->temporal_register = pck_received;
	        this->accumulatorStats.n_register_writes++;   //To track the stats
        }
        else {
            result = perform_operation_2_operands(this->temporal_register, pck_received);  // 是用new实例化得到的
	        this->accumulatorStats.n_register_reads++; //To track the stats
            // delete this->temporal_register;  // ???????????????????????????????????????????????????????????????????
            // this->temporal_register = nullptr;
            // delete pck_received; // ????????????????????????????????????????????????????????????????????????????????????????????????????
            // pck_received = nullptr;
            this->temporal_register = result;
	        this->accumulatorStats.n_register_writes++;  //To track the stats
        }
        
        // add
        //std::cout<<"the number of accumulator is ["<<this->n_accumulator<<"] and the result is ["<<this->temporal_register->get_data()<<"]"<<std::endl;

        if(this->current_psum == (this->n_psums-1)) {  // 累加结束后，将结果放入output_fifo中
            this->output_fifo->push(this->temporal_register);
            this->current_psum = 0;
        }
        else {
            this->current_psum++;
        }

        // if(this->n_accumulator == 0){
        //     std::cout<<"**********************************"<<std::endl;
        //     std::cout<<"current_psum is : "<<this->current_psum<<std::endl;
        //     std::cout<<"current_sum is : "<<this->temporal_register->data<<std::endl;
        // }

    }
}

//TODO
// this->receive() : 接收input_connection中的数据，存放在input_fifo中
// this->route() : 如果input_fifos中有数据，则将其进行累加，若是最后一次累加，则将结果存入output_fifo
// this->send() : 将output_fifo中的数据送入output_connection（与总线的连接）
void Accumulator::cycle() {
    this->local_cycle+=1; 
    this->accumulatorStats.total_cycles++;  //To track the stats
    this->receive(); //Receive input
    this->route();
    this->send(); //Send towards the memory

}



/*
//Print the configuration of the Accumulator
void FEASwitch::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; //TODO put ID
    out << ind(indent+IND_SIZE) << "\"Configuration\" : \"" <<get_string_adder_configuration(this->config_mode) << "\"" << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Augmented_link_enabled\" : " << this->fw_enabled << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Augmented_link_direction\" : \"" << get_string_fwlink_direction(this->fl_direction) << "\"" << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Left_child_enabled\" : " << this->left_child_enabled << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Right_child_enabled\" : " << this->right_child_enabled << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"BusID\" : " << this->busID << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"InputID\" : " << this->inputID << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Send_result_to_fold_node\" : " << this->forward_to_fold_node << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"Send_result_to_memory\" : " << this->forward_to_memory  << std::endl;
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}
*/
//Print the statistics obtained during the execution

void Accumulator::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; //TODO put ID
    this->accumulatorStats.print(out, indent+IND_SIZE);
    //Printing Fifos

    out << ind(indent+IND_SIZE) << ",\"InputFifo\" : {" << std::endl;
        this->input_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}," << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent+IND_SIZE) << "\"OutputFifo\" : {" << std::endl;
        this->output_fifo->printStats(out, indent+IND_SIZE+IND_SIZE);
    out << ind(indent+IND_SIZE) << "}" << std::endl; //Take care. Do not print endl here. This is parent responsability

    out << ind(indent) << "}"; //TODO put ID

}

void Accumulator::printEnergy(std::ofstream& out, unsigned int indent){
    /* 
     This component prints:
         - Number of accumulator reads
         - Number of accumulator writes
	 - Number of sums performed by the accumulator
         - Number of reads and writes to the next fifos:
             * input_fifo: fifo to receive data
             * output_fifo: fifo to send data to memory
    */

    out << ind(indent) << "ACCUMULATOR READ=" << this->accumulatorStats.n_register_reads;
    out << ind(indent) << " WRITE=" << this->accumulatorStats.n_register_writes;
    out << ind(indent) << " ADD=" << this->accumulatorStats.n_adds << std::endl;

    //Fifos
    this->input_fifo->printEnergy(out, indent);
    this->output_fifo->printEnergy(out, indent);

}

