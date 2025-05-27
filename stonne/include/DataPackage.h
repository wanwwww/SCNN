//Created 13/06/2019

#ifndef __data_package_h__
#define __data_package_h__

#include <memory>
#include "types.h"
#include <iostream>
#include <vector>

/*

This class represents the wrapper of a certain data. It is used in both networks ART and DS but there are some fields that are used in just one particular class. For example, 
since the DS package does not need the VN, it is not used during that network. 

*/

// DataPackage类是一个封装数据类，模拟在特定网络中传输的数据包
// 包含数据包的通用属性和专门针对特定网络的字段

class DataPackage {

public:
    //General field
    // 数据包的大小、数据本身、数据类型、数据包来源 
    size_t size_package; //Actual size of the package. This just accounts for the truly data that is sent in a real implementation
    data_t data; //Data in the package

    std::vector<bool> data_vector; // add  建模二进制串数据
    bool data_result; // add 建模池化结果
    // 为池化模块增加的变量（为了计算池化结果的写地址）
    int channel_num;  
    int retrieve_num;
    int location;

    operand_t data_type; //Type of data (i.e., WEIGHT, IACTIVATION, OACTIVATION, PSUM)
    id_t source; //Source that sent the package
   
    // Fields only used for the DS
    bool* dests;  // Used in multicast traffic to indicate the receivers  一个布尔数组
    unsigned int n_dests; //Number of receivers in multicast operation
    unsigned int unicast_dest; //Indicates the destination in case of unicast package
    traffic_t traffic_type; // IF UNICAST dest is unicast_dest. If multicast, dest is indicate using dests and n_dests. 广播、多播、单播

    unsigned int VN; //Virtual network where the psum is found
    adderoperation_t operation_mode; //operation that got this psum (Comparation or SUM) ： ADDER, COMPARATOR, MULTIPLIER, NOP
    
    unsigned int output_port; //Used in the psum package to get the output port that was used in the bus to send the data 
    unsigned int iteration_g; //Indicates the g value of this package (i.e., the number of g iteration). This is used to avoid sending packages of some iteration g and k without having performing the previous ones.
    unsigned int iteration_k; //Indicates the k value of this package (i.e, the number of k iteration). This is used to avoid sending packages of some iteration k (output channel k) without having performed the previous iterations yet
    

    virtual ~DataPackage() = default;
    
    //General constructor to be reused in both types of packages
    DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source);

    // add
    DataPackage(std::vector<bool> data, int channel_num, int retrieve_num);
    DataPackage(bool data, int channel_num, int retrieve_num, int location);
    
    //DS Package constructors for creating unicasts, multicasts and broadcasts packages
    //General constructor for DS
    DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type);
    // Unicast package constructor. 
    DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, unsigned int unicast_dest);
    //Multicast package. dests must be dynamic memory since the array is not copied. 
    DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, bool* dests, unsigned int n_dests); //Constructor
    //Broadcast package
    //Needs nothing. Just indicates is the type broadcast

    //ART Package constructor (only one package for this type)
    DataPackage(size_t size_package, data_t data, operand_t data_type, id_t source, unsigned int VN, adderoperation_t operation_mode);
    // ~DataPackage();
    
    DataPackage(std::shared_ptr<DataPackage> pck); //Constructor copy used to repeat a package

    

    //Getters 访问器方法，用于访问私有成员变量
    const size_t get_size_package()            const {return this->size_package;}
    const std::vector<bool> get_data_vector()  const {return this->data_vector;}  // add 建模二进制的数据
    const data_t get_data()                    const {return this->data;}
    const operand_t get_data_type()            const {return this->data_type;}
    const id_t get_source()                    const {return this->source;}
    const traffic_t get_traffic_type()         const {return this->traffic_type;}
    bool isBroadcast()                   const {return this->traffic_type==BROADCAST;}
    bool isUnicast()                     const {return this->traffic_type==UNICAST;}
    bool isMulticast()                   const {return this->traffic_type==MULTICAST;}
    const bool* get_dests()                    const {return this->dests;}
    unsigned int get_unicast_dest()        const {return this->unicast_dest;}
    unsigned int get_n_dests()                  const {return this->n_dests;}
    unsigned int getOutputPort()           const {return this->output_port;}
    unsigned int getIterationK()           const {return this->iteration_k;}
    // 设置包的输出端口和迭代值
    void setOutputPort(unsigned int output_port);
    void setIterationK(unsigned int iteration_k); //Used to control avoid a package from the next iteration without having calculated the previous ones.

    unsigned int get_vn()                  const {return this->VN;}
    adderoperation_t get_operation_mode()   const {return this->operation_mode;}
};

#endif
