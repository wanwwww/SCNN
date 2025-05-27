//Created 13/06/2019

#ifndef __Connection__h
#define __Connection__h

#include <memory>
#include "types.h"
#include "DataPackage.h"
#include <vector>
#include "Stats.h"

/*
This class Connection does not need ACK responses since in the accelerator the values are sent without a need of a request. Everything is controlled
by the control of the accelerator. 
*/

// 模拟硬件连接的行为，包括数据的发送和接收，带宽限制，以及跟踪统计信息

class Connection {
private:
    bool pending_data;   // Indicates if data exists
    size_t bw;           // Size in bytes of actual data. In the simulator this size is greater since we wrap the data into wrappers to track.
    std::vector<std::shared_ptr<DataPackage>> data;   // Array of packages that are send/receive in  a certain cycle. The number of packages depends on the bw of the connection
    std::shared_ptr<DataPackage> data_pooling_result;  // add
    unsigned int current_capacity; // the capacity must not exceed the bw of the connection
    ConnectionStats connectionStats; //Tracking parameters

public:
    Connection(int bw);
    void send(std::vector<std::shared_ptr<DataPackage>> data); //Package of data to be send. The sum of all the size_package of each package must not be greater than bw.
    void send_pooling_result(std::shared_ptr<DataPackage> data);
    std::vector<std::shared_ptr<DataPackage>> receive();  //Receive a  packages from the connection
    std::shared_ptr<DataPackage> receive_pooling_result();
    bool existPendingData();
    
    void printEnergy(std::ofstream &out, unsigned int indent, std::string wire_type);


};


#endif

