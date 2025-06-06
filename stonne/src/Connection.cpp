//Created 13/06/2019

#include "Connection.h"
#include <iostream>
#include <assert.h>

using namespace std;

// 构造函数，设置连接的最大带宽，即一次传输中允许的数据量 
Connection::Connection(int bw) { //Constructor
    this->bw = bw; //Maximum bw allowed in the connection
    this->current_capacity = 0; 
    this->pending_data=false;
}


bool Connection::existPendingData() {
    return this->pending_data;
}

// 向连接中发送数据 
//Send a package to the interconnection. If there is no remaining bandiwth an exception is raised
void Connection::send(vector<std::shared_ptr<DataPackage>> data_p) {
// #ifdef DEBUG
//     //Check the connection is not busy
//     assert(pending_data==false);
//     //Check there is enouth bandwidth. This case should not happen so if happens, an assert is raised. 
//     this->current_capacity=0;
//     for(int i=0; i<data_p.size(); i++) {
//         std::shared_ptr<DataPackage> current_package = data_p[i];
// 	//Check there is enouth bandwidth. This case should not happen so if happens, an assert is raised
// 	assert( (current_package->get_size_package() + this->current_capacity) <= this->bw );   
// 	this->current_capacity += current_package->get_size_package(); //Increasing the amount of data in the connection
//     }
// #endif

    this->data = data_p; //list of pointers assignment. All the vectors are replicated to save a copy and track it.
    this->pending_data = true;
    
    //Tracking parameters
    this->connectionStats.n_sends+=1;   
    return; 
}

void Connection::send_pooling_result(std::shared_ptr<DataPackage> data) {

    this->data_pooling_result = data; //list of pointers assignment. All the vectors are replicated to save a copy and track it.
    this->pending_data = true;
    
    //Tracking parameters
    this->connectionStats.n_sends+=1;   
    return; 
}

// 从连接中接收数据  
//Return the packages from the interconnection
vector<std::shared_ptr<DataPackage>> Connection::receive() { 
    
    if(this->pending_data) {
            this->pending_data = false;
	    return this->data;
    }

    //If there is no pending data
    data.clear(); //Set the list of elements to return to 0
    this->pending_data = false;
    
    //Tracking parameters
    this->connectionStats.n_receives+=1;
  
    return data; //Return empty list indicating that there is no data
}

std::shared_ptr<DataPackage> Connection::receive_pooling_result() { 
    
    if(this->pending_data) {
            this->pending_data = false;
	    return this->data_pooling_result;
    }

    //If there is no pending data
    //data_pooling_result.clear(); //Set the list of elements to return to 0
    this->pending_data = false;
    
    //Tracking parameters
    this->connectionStats.n_receives+=1;
  
    return data_pooling_result; //Return empty list indicating that there is no data
}

void Connection::printEnergy(std::ofstream &out, unsigned int indent, std::string wire_type) {
    out << wire_type << " WRITE=" << connectionStats.n_sends; //Same line
    out << " READ=" << connectionStats.n_receives << std::endl;
}




