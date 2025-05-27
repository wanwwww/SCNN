// Created the 4th of november of 2019 by Francisco Munoz Martinez

#include "CollectionBusLine.h"
#include "utility.h"

// 构造函数，初始化总线线，创建input_connections、input_fifos、output_port

CollectionBusLine::CollectionBusLine(id_t id, std::string name, unsigned int busID, unsigned int input_ports_bus_line, unsigned int connection_width, unsigned int fifo_size) : Unit(id,name) {
    this->input_ports=input_ports_bus_line; 
    this->busID = busID;
    //Creating the connections for this bus line
    for(int i=0; i<this->input_ports; i++) {
        //Adding the input connection
        Connection* input_connection = new Connection(connection_width);  
        input_connections.push_back(input_connection);
   
        //Adding the input fifo
        Fifo* fifo = new Fifo(fifo_size);
        input_fifos.push_back(fifo);

        //Creating the output connection
        output_port = new Connection(connection_width);
        this->collectionbuslineStats.n_inputs_receive.push_back(0); //To track information
       
    }

    next_input_selected=0;  // 初始化为0，表示轮询从第一个端口开始 
}

CollectionBusLine::~CollectionBusLine() {
    //First removing the input_connections
    for(int i=0; i<input_connections.size(); i++) {
        delete input_connections[i];
    }

    //Deleting the input_fifos
    for(int i=0; i<input_fifos.size(); i++) {
        delete input_fifos[i];
    }

    //Deleting output connection
    delete output_port;
}


Connection* CollectionBusLine::getInputPort(unsigned int inputID) {
    return this->input_connections[inputID];
}

// 遍历每个输入连接，如果某个连接有待处理的数据，接收数据包并存入对应的FIFO队列，更新统计信息 
void CollectionBusLine::receive() {
    for(int i=0; i<this->input_connections.size(); i++) {
        if(input_connections[i]->existPendingData()) {
            std::vector<std::shared_ptr<DataPackage>> pck = input_connections[i]->receive();
            for(int j=0; j<pck.size(); j++) { //Actually this is 1
                this->collectionbuslineStats.n_inputs_receive[i]+=1; //To track information. Number of packages received by each input line for this output port
                input_fifos[i]->push(pck[j]); //Inserting the package into the fifo
            }
        }
    }
}


// 首先接收数据包，然后检查FIFO队列是否有数据，使用轮询机制从非空FIFO中取出一个数据包，将其发送到输出连接

void CollectionBusLine::cycle() {
    this->collectionbuslineStats.total_cycles++;
    this->receive(); //Receiving packages from the connections // 从输入连接中接收数据，存放在input_fifos中
    bool selected=false;
    unsigned int n_iters = 0;

    //To track Information 检查FIFO队列是否有数据，如果有多个队列有数据，记录冲突信息
    unsigned int n_inputs_trying=0;

    for(int i=0; i<input_fifos.size(); i++) {
        if(!input_fifos[i]->isEmpty()) {
            n_inputs_trying+=1;
        }
    }

    this->collectionbuslineStats.n_conflicts_average+=n_inputs_trying; //Later this will be divided by the number of total cycles to calculate the average
    if(n_inputs_trying>1) {
        this->collectionbuslineStats.n_times_conflicts+=1; //To track information
    }

    //End to track information and the actual code to perform the cycle is executed
    
    // 轮询从非空的FIFO中取出一个数据包，
    // 一次只取出一个数据包！！！
    std::vector<std::shared_ptr<DataPackage>> data_to_send;
    while(!selected && (n_iters < input_fifos.size())) { //if input not found or there is still data to look up
        if(!input_fifos[next_input_selected]->isEmpty()) { //If there is data in this input then
            selected=true;
            std::shared_ptr<DataPackage> pck = input_fifos[next_input_selected]->pop(); //Poping from the fifo
            pck->setOutputPort(this->busID); //Setting tracking information to the package 更新数据包的目标端口 
            data_to_send.push_back(pck); //Sending the package to memory 将pck追加到data_to_send向量尾部
            this->collectionbuslineStats.n_sends++; //To track information
        }
        next_input_selected = (next_input_selected + 1) % input_fifos.size(); 
        n_iters++;
    }

    //Sending the data to the output connection
    // 将数据包发送到输出连接
    if(selected) {
        this->output_port->send(data_to_send);
    }
 
}

void CollectionBusLine::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "{" << std::endl; //TODO put ID
    this->collectionbuslineStats.print(out, indent+IND_SIZE);
    out << ind(indent+IND_SIZE) << ",\"input_fifos_stats\" : [" << std::endl;
    for(int i=0; i<input_fifos.size(); i++) {
        out << ind(indent+IND_SIZE+IND_SIZE) << "{" << std::endl;
        input_fifos[i]->printStats(out, indent+IND_SIZE+IND_SIZE+IND_SIZE);
        out << ind(indent+IND_SIZE+IND_SIZE) << "}";
        if(i<(input_fifos.size()-1)) {
            out << ",";
        }

        out << std::endl;
    }
    out << ind(indent+IND_SIZE) << "]" << std::endl;
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}

void CollectionBusLine::printEnergy(std::ofstream& out, unsigned int indent) {
   /*
       This component prints: 
           - The input wires connected to this output wire
           - The input FIFOs to connect every input wire
           - The output wire 
   */

   //Printing input wires
   for(int i=0; i<input_fifos.size(); i++) {
       Connection* conn = input_connections[i];
       conn->printEnergy(out, indent, "CB_WIRE");
   }

   //Printing input fifos
   for(int i=0; i<input_fifos.size(); i++) {
       Fifo* fifo = input_fifos[i];
       fifo->printEnergy(out, indent); 
   }

   //Printing output wire
   output_port->printEnergy(out, indent, "CB_WIRE");
}
