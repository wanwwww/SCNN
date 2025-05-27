
#include "OSMeshSDMemory.h"
#include <assert.h>
#include <iostream>
#include "utility.h"

#include "NeuronStateUpdater.h"

OSMeshSDMemory::OSMeshSDMemory(id_t id, std::string name, Config stonne_cfg) : MemoryController(id, name) {
    //this->write_connection = write_connection;  // 这个连接是和查找表的连接 
    //Collecting parameters from the configuration file
    this->ms_rows = stonne_cfg.m_MSNetworkCfg.ms_rows;  //Used to send data
    this->ms_cols = stonne_cfg.m_MSNetworkCfg.ms_cols; 
    this->n_read_ports=stonne_cfg.m_SDMemoryCfg.n_read_ports;
    this->n_write_ports=stonne_cfg.m_SDMemoryCfg.n_write_ports;
    this->write_buffer_capacity=stonne_cfg.m_SDMemoryCfg.write_buffer_capacity;
    this->port_width=stonne_cfg.m_SDMemoryCfg.port_width;
    //End collecting parameters from the configuration file

    //Initializing parameters
    this->ms_size_per_input_port = (this->ms_rows + this->ms_cols) / this->n_read_ports; // 每个端口负责的乘法器数
    this->write_fifo = new Fifo(write_buffer_capacity);
    for(int i=0; i<this->n_read_ports; i++) {

        Connection* conn = new Connection(this->port_width);  // add

        Fifo* read_fi = new Fifo(this->write_buffer_capacity);
        Fifo* psum_fi = new Fifo(this->write_buffer_capacity);

        read_connections[i] = conn;  // add

        input_fifos.push_back(read_fi); // 用于在获取输入之前存储输入
        psum_fifos.push_back(psum_fi);  // 用于在获取之前存储部分和
        this->sdmemoryStats.n_SRAM_read_ports_weights_use.push_back(0);  //To track information  端口读取权重的次数 
        this->sdmemoryStats.n_SRAM_read_ports_inputs_use.push_back(0);   //To track information
        this->sdmemoryStats.n_SRAM_read_ports_psums_use.push_back(0);    //To track information
    }

    for(int i=0; i<this->n_write_ports; i++) {  //To track information
        this->sdmemoryStats.n_SRAM_write_ports_use.push_back(0);  //To track information
    }  //To track information

    this->configuration_done = false;
    this->execution_finished = false;
    this->metadata_loaded = false;
    this->layer_loaded = false;
    this->local_cycle=0;
    this->current_state = OS_CONFIGURING; // 初始状态为配置状态 
    this->current_output = 0;
    this->output_size = 0;
    this->current_output_iteration = 0;
    this->output_size_iteration = 0;
    this->current_M=0;
    this->current_N=0;
    this->current_K=0;
    this->iteration_completed=false;
    this->n_iterations_completed=0;

    // add
    this->current_progress = 0;
    this->current_Timestamp = 0;
    this->iter_Timestamp = stonne_cfg.Timestamp;
    this->n_timestamp_completed = 0;

    this->pooling_enabled = false;

    this->print_flag = false;
}

OSMeshSDMemory::~OSMeshSDMemory() {
    delete write_fifo;
    //Deleting the input ports
    for(int i=0; i<this->n_read_ports; i++) {
        delete input_fifos[i];
        delete psum_fifos[i];
    }
    

}

// 设置写端口连接 复制所有connection的指针 
void OSMeshSDMemory::setWriteConnections(std::vector<Connection*> write_port_connections) {
    this->write_port_connections=write_port_connections; //Copying all the poiners 
    //assert(this->write_port_connections.size()==this->n_write_ports); 
}

// void OSMeshSDMemory::setReadConnections(std::vector<Connection*> read_connections) {
//     assert(read_connections.size() == n_read_ports); //Checking that the number of input ports is valid.
//     this->read_connections = read_connections; //Copying all the pointers
// }

// 设置加载的层 
void OSMeshSDMemory::setLayer(DNNLayer* dnn_layer, address_t MK_address, address_t KN_address, address_t output_address, address_t neuron_state, Dataflow dataflow) {
    //std::cout<<"debug in setLayer"<<std::endl;
    this->dnn_layer = dnn_layer;
    //assert(this->dnn_layer->get_layer_type()==DenseGEMM);  // This controller only supports GEMM with one sparse and one dense
    //this->dataflow = dataflow; 
    this->output_address = output_address;
    this->neuron_state = neuron_state;
    this->layer_loaded = true;

    //Loading parameters according to the equivalence between CNN layer and GEMM. This is done
    //in this way to keep the same interface.
    this->M = this->dnn_layer->get_X_()*this->dnn_layer->get_Y_();
    //std::cout<<" M (OSMeshSDMemory::setLayer)  :  "<<this->M<<std::endl;
    //this->M = this->dnn_layer->get_X();
    //this->K = this->dnn_layer->get_S();   //Be careful. K in GEMMs (SIGMA taxonomy) is not the same as K in CNN taxonomy (number of filters)
    this->K = this->dnn_layer->get_R()*this->dnn_layer->get_S()*this->dnn_layer->get_C(); 
    //std::cout<<" K  :  "<<this->K<<std::endl;
    this->N = this->dnn_layer->get_K();  //In this case both parameters match each other.
    //std::cout<<" N  :  "<<this->N<<std::endl;

    sdmemoryStats.dataflow=dataflow; 
    
    this->MK_address = MK_address;
    this->KN_address = KN_address;

  //  this->output_size = dim_sta*dim_str;
    this->output_size = M*N;
    //std::cout<<"debug in setLayer end"<<std::endl;
}


//Dense Tiles
void OSMeshSDMemory::setTile(Tile* current_tile)
{
    this->T_M = current_tile->get_T_X_(); //According to loadGemmTile function in STONNE the T_M gemm parameter is saved in T_K DNN parameter
    this->T_N = current_tile->get_T_K();
    this->T_K = 1;                     // =================================================== 这里为什么取1
    this->iter_M = M / T_M + ((M % T_M)!=0);		
    this->iter_K = K;
    this->iter_N = N / T_N + ((N % T_N)!=0);
    
    for(int i=0;i<(T_N*T_M);i++)
    	this->vnat_table.push_back(0); // 在向量的末尾添加一个0
}


// 负责管理数据的发送和接收以及状态转换 

void OSMeshSDMemory::cycle() {
    // std::cout<<"Calling the memory controller "<<std::endl;

    //Sending input data over read_connection
    assert(this->layer_loaded);  // Layer has been loaded
    
    // data_to_send 临时存储要发送的输入和权重数据 
    std::vector<std::shared_ptr<DataPackage>> data_to_send; //Input and weight temporal storage
    //std::vector<std::shared_ptr<DataPackage>> psum_to_send; // psum temporal storage
    this->local_cycle+=1;
    this->sdmemoryStats.total_cycles++; //To track information

    // 配置状态 
    if(current_state==OS_CONFIGURING) 
    {	//Initialize these for the first time
        // std::cout<<"Calling the memory controller and the cycle is "<<this->local_cycle<<std::endl;
        this->sdmemoryStats.n_reconfigurations++;
        //std::cout<<"this->sdmemoryStats.n_reconfigurations(OSMeshSDMemory.cpp) : "<<this->sdmemoryStats.n_reconfigurations<<std::endl;
        unsigned int remaining_M = M - (current_M*T_M);
        unsigned int remaining_N = N - (current_N*T_N);
        //std::cout<<"1"<<std::endl;
        // 实际使用的行数和列数 
        this->cols_used = (remaining_N < T_N) ? remaining_N: T_N; //min(remaining_N, T_N) 
        this->rows_used = (remaining_M < T_M) ? remaining_M: T_M;
        // Tile(unsigned int T_R, unsigned int T_S, unsigned int T_C, unsigned int T_K, unsigned int T_G,  unsigned int T_N, unsigned int T_X_, unsigned int T_Y_, bool folding);
        Tile* tile1 = new Tile(1, 1, 1, cols_used, 1, 1, rows_used, 1, false); 
        this->multiplier_network->resetSignals(); // 将MN中每个MS的forward_right、forward_bottom设置为false，VN设置为0
        //std::cout<<"2"<<std::endl;
        this->reduce_network->resetSignals();  // 将AN中每个AS的current_psum、n_psums设置为0，operation_mode设置为ADDER
        //std::cout<<"3"<<std::endl;
        // 配置MN的信号，也就是配置所使用的PE当中MS的向下向右转发使能、VN编号 
        this->multiplier_network->configureSignals(tile1, this->dnn_layer, this->ms_rows, this->ms_cols);
        //std::cout<<"4"<<std::endl;
        // 配置AN，也就是配置AN中每个AS需要进行累加部分和的次数为iter_k，即配置AS的变量n_psums为iter_k
        this->reduce_network->configureSignals(tile1, this->dnn_layer, this->ms_rows*this->ms_cols, this->iter_K);
        //std::cout<<"5"<<std::endl;

        iteration_completed=false;
        //std::cout<<"config complete"<<std::endl;

        // 当前PE完成处理需要往内存中写多少次
        if(this->pooling_enabled){
            this->all_timestamp_required_output = iter_Timestamp*M*N+iter_Timestamp*M*N/4;

            if(!this->print_flag){
                std::cout<<"M ： "<<M<<std::endl;
                std::cout<<"N ： "<<N<<std::endl;
                std::cout<<"T ： "<<iter_Timestamp<<std::endl;
                std::cout<<"all_timestamp_required_output ： "<<this->all_timestamp_required_output<<std::endl;
                this->print_flag = true;
            }

            this->one_timestamp_required_output = M*N+M*N/4;
            this->one_PE_requierd_output = this->rows_used*this->cols_used+this->rows_used*this->cols_used/4;
        } else{
            this->all_timestamp_required_output = iter_Timestamp*2*M*N;
            this->one_timestamp_required_output = 2*M*N;
            this->one_PE_requierd_output = 2*this->rows_used*this->cols_used;
            //std::cout<<"all_timestamp_required_output ： "<<this->all_timestamp_required_output<<std::endl;
        }

        //std::cout<<"debug"<<std::endl;
    }
    
    //std::cout<<"debug 1"<<std::endl;

    // 分发数据状态，分发的是当前tile的数据 
    if(current_state == OS_DIST_INPUTS) {
        //std::cout<<"============================================================="<<std::endl;

        //std::cout<<"Current timestamp : "<<current_Timestamp<<std::endl;
        
        //Distribution of the stationary matrix
        unsigned int dest = 0; //MS destination
        
        //SENDING N PACKAGES  // N个数据包：权重数据 
        // 先迭代K，也就是依次取iter_K行数据 
        for(int i=0; i<this->cols_used; i++) {  // 每次循环取一行数据（个数是实际使用的列数）
            data_t data;//Accessing to memory
            int index_N=current_N*T_N;
            // 从内存中读取输入，每次取一个数据 
            data = this->KN_address[(index_N+i)*this->K + this->current_K]; //Notice that in dense operation the KN matrix is actually NK 
            //std::cout<<"the address of weight is : "<<(index_N+i)*this->K + this->current_K<<std::endl;
            //std::cout<<"The weight taken out is : "<<data<<std::endl;
            //std::cout<<std::endl;
            sdmemoryStats.n_SRAM_weight_reads++;  
            // (size_t size_package, data_t data, operand_t data_type, id_t source,traffic_t traffic_type, unsigned int unicast_dest) : 
            std::shared_ptr<DataPackage> pck_to_send = std::make_shared<DataPackage>(sizeof(data_t), data, WEIGHT, 0, UNICAST, i);
            //std::cout<<"old data : "<<pck_to_send->getIterationK()<<std::endl;
            this->sendPackageToInputFifos(pck_to_send);  // 将取出的数据发送到内存相应读端口的fifo中
        }
        //std::cout<<std::endl;
        //SENDING M PACKAGES. Note that not delay is necessary as the fifos will carry out these delays 
        // M个数据包：激活数据。从内存中读取的延迟放在fifo中考虑
        for(int i=0; i<rows_used; i++) {  // 每次循环取出一列数据（个数是实际使用的行数），依次取出iter_K列数据 
            data_t data;//Accessing to memory
            int index_M=current_M*T_M;  // 注意input数据读取的地址
            data = this->MK_address[current_Timestamp*M*K+(index_M+i)*this->K + this->current_K]; 
            //std::cout<<"the address of input is : "<<current_Timestamp*M*K+(index_M+i)*this->K + this->current_K<<std::endl;
            //std::cout<<"The input taken out is : "<<data<<std::endl;
            //std::cout<<std::endl;
            sdmemoryStats.n_SRAM_input_reads++;  
            std::shared_ptr<DataPackage> pck_to_send = std::make_shared<DataPackage>(sizeof(data_t), data, IACTIVATION, 0, UNICAST, i+this->ms_cols);
            this->sendPackageToInputFifos(pck_to_send);
        }
        //std::cout<<std::endl;
        this->current_K+=1;

        // K -> N -> M
        // 如果this->current_K == this->iter_K说明current_K维度的计算已经完成，重置current_K为0
        // this->current_N+=1;表示进入下一列块 
        // this->current_M+=1;表示进入下一行块 

        if(this->current_K == this->iter_K) {
            this->current_K = 0;
            this->current_N += 1;
            this->iteration_completed=true;  // 表示完成一次T_M行与T_N列的计算 
            if(this->current_N == this->iter_N) { 
                this->current_N = 0;
                this->current_M+=1;   
                this->iteration_completed=true;  // 表示完成一次T_M行与T_N列的计算 
                if(this->current_M == this->iter_M) {
                    this->current_M = 0;
                    this->current_Timestamp+=1;
                    //std::cout<<"timestamp++ : "<<this->current_Timestamp<<std::endl;
                    this->iteration_completed = true;  // 表示完成一个时间步的计算 
                }
            } 
        } 
       
    } //END STATE
   
    //std::cout<<"debug 2"<<std::endl;

    //Receiving output data from write_connection 
    this->receive(); // 接收write_connection和write_port_connections的数据，将其存放在write_fifo中

    //std::cout<<"debug 3"<<std::endl;

    if(!write_fifo->isEmpty()) {
        //std::cout<<"debug 4"<<std::endl;
        //std::cout<<"the length of write_fifo(OSMeshSDMemory.cpp) : "<<write_fifo->size()<<std::endl;
        //Index the data by using the VN Address Table and the VN id of the packages
        std::size_t write_fifo_size = write_fifo->size();
        for(int i=0; i<write_fifo_size; i++) {
            //std::cout<<" i : "<<i<<std::endl;
            std::shared_ptr<DataPackage> pck_received = write_fifo->pop();
            unsigned int vn = pck_received->get_vn();
            data_t data = pck_received->get_data();
            this->sdmemoryStats.n_SRAM_psum_writes++; //To track information 
            // n_iterations_completed / this->iter_N 得到当前迭代属于的M行块索引。
            // 乘以 T_M，计算当前 M 行块的起始位置（行号）
            // n_iterations_completed % this->iter_N 得到当前迭代属于的N列块索引
            // 乘以 T_N，计算当前 N 列块的起始位置（列号）
            //unsigned int current_tile_M_pointer = (n_iterations_completed /this->iter_N)*this->T_M;  //Change this to change the dataflow
            unsigned int current_tile_M_pointer = (n_iterations_completed /this->iter_N)*this->ms_rows;  //Change this to change the dataflow
            //unsigned int current_tile_N_pointer = (n_iterations_completed % this->iter_N)*T_N;
            unsigned int current_tile_N_pointer = 0;
            // 当前虚拟节点在当前数据块内的行和列位置。
            unsigned int vn_M_pointer = vn / this->cols_used;
            unsigned int vn_N_pointer = vn % this->cols_used;
            unsigned int timestamp_offset = n_timestamp_completed*M*N; // 已经计算完毕的偏移量 12
            unsigned int timestamp_offset_pooling = n_timestamp_completed*M*N/4 ;  // 带有池化模块的计算完毕的偏移量 
            unsigned int addr_offset = (current_tile_M_pointer+vn_M_pointer)*this->N + current_tile_N_pointer + vn_N_pointer; 
            //unsigned int addr_offset = (current_tile_M_pointer+vn_M_pointer)*this->N + current_tile_N_pointer + vn_N_pointer; 
            unsigned int addr_pooling = ((current_tile_M_pointer+vn_M_pointer)/2)*this->N/2 + (current_tile_N_pointer + vn_N_pointer)/2;
            vnat_table[vn]++; // 记录每个虚拟节点的访问次数 

            // modify
            // 根据数据类型判断将数据写入输出内存还是膜电位内存
            switch (pck_received->get_data_type()){
                case VTH:
                    this->neuron_state[addr_offset] = data;
                    // if(addr_offset == 96){
                    //     std::cout<<"  neuron state is : "<<data<<"======================="<<std::endl;
                    // }
                    
                    //std::cout<<"the data type is neuron state"<<std::endl;
                    break;
                case SPIKE:
                    if(pooling_enabled){
                        this->output_address[addr_pooling+timestamp_offset_pooling] = data;
                        //std::cout<<"-------------------------------------------------- the address of spike with pooling "<<addr_pooling<<std::endl;
                    } else {
                        this->output_address[timestamp_offset+addr_offset] = data;
                        // std::cout<<"output write addr is : "<<addr_offset<<std::endl;
                        // if(addr_offset == 96){
                        //     std::cout<<"write data is : "<<data<<std::endl;
                        // }
                    }
                    //std::cout<<"the data type is spike"<<std::endl;
                    break;
                default:
                    assert(false);
            }

            current_progress++;  // 总的输出进度 
            current_output++;  // 当前时间步的输出进度 
            current_output_iteration++;  // 当前PE阵列计算的进度

            //std::cout<<""<<std::endl;

            // if(current_output == 2*M*N) {
            //     current_Timestamp++;
            //     current_output = 0;
            //     if(current_Timestamp == iter_Timestamp){
            //         execution_finished=true;
            //     }
            // }

            if((current_progress % 2000) == 0) {
                std::cout << "Output completed " << current_progress << "/" << this->all_timestamp_required_output << ")" << std::endl;
            }

            // if(this->flag){
            //     std::cout<<"The total number of outputs required : "<<this->all_timestamp_required_output<<std::endl;
            //     this->flag = false;
            // }
            // //std::cout<<"The total number of outputs required : "<<this->all_timestamp_required_output<<std::endl;
            // if(current_progress % 2000 ==0){
            //     std::cout<<"Current number of outputs : "<<current_progress<<std::endl;    
            // }

            if(current_progress == (this->all_timestamp_required_output)){
                //std::cout<<"end : "<<current_progress<<std::endl;
                execution_finished = true;
            }

            if(current_output_iteration==this->one_PE_requierd_output) {  // PE阵列完成一次计算 
                current_output_iteration = 0;
                n_iterations_completed++;  // 一个时间步中使用PE阵列进行计算完成的次数 
                if(current_state == OS_WAITING_FOR_NEXT_ITER) {
                    iteration_completed=true;
                }
            }

            // if(current_output == this->one_timestamp_required_output){  // 完成一个时间步的计算 
            //     n_iterations_completed = 0;
            //     current_output = 0;
            //     n_timestamp_completed++;
            // }

            // delete pck_received; //Deleting the current package ?????????????????????????????????????????????????????????
            // pck_received = nullptr;
            //std::cout<<"the length of write_fifo (after) : "<<write_fifo->size()<<std::endl;
        }

    }

    //std::cout<<"debug 5"<<std::endl;

    //Transitions
    if(current_state==OS_CONFIGURING) {
        current_state=OS_DIST_INPUTS;  // 配置后进入分发状态 
    }
    else if(current_state==OS_DIST_INPUTS) {
        if(iteration_completed) {
            if(current_Timestamp >= this->iter_Timestamp) { //Change the to change the dataflow
                current_state=OS_ALL_DATA_SENT;  // 所有数据已经发送
            }
	        else {
                current_state=OS_WAITING_FOR_NEXT_ITER;  // 等待下次迭代，该状态表示等待硬件完成当前tile的计算
                this->iteration_completed = false;
            }
	    }
    }
    else if(current_state == OS_WAITING_FOR_NEXT_ITER) {
        if(iteration_completed) {
            current_state=OS_CONFIGURING;  // 准备重新配置 
	    } 
    }

    // 发送数据 
    this->send();

}

bool OSMeshSDMemory::isExecutionFinished() {
    return this->execution_finished;
}

OSMeshControllerState OSMeshSDMemory::getCurrentState(){
    return this->current_state;
}

void OSMeshSDMemory::reset(){
    this->execution_finished = false;
    this->current_progress = 0;
    this->current_state = OS_CONFIGURING;
    this->current_Timestamp = 0;
}


/* The traffic generation algorithm generates a package that contains a destination for all the ms. We have to divide it into smaller groups of ms since they are divided into several ports */
// 根据发送方式（单播、广播、多播）发送数据包到相应端口的fifo
void OSMeshSDMemory::sendPackageToInputFifos(std::shared_ptr<DataPackage> pck) {
    // BROADCAST PACKAGE 广播
    if(pck->isBroadcast()) {
        //Send to all the ports with the flag broadcast enabled
        for(int i=0; i<this->n_read_ports; i++) {
            //Creating a replica of the package to be sent to each port
            std::shared_ptr<DataPackage> pck_new = std::make_shared<DataPackage>(pck->get_size_package(), pck->get_data(), pck->get_data_type(), i, BROADCAST); //Size, data, data_type, source (port in this case), BROADCAST
            //Sending the replica to the suitable fifo that correspond with the port  将副本发送到端口对应的fifo
            if(pck->get_data_type() == PSUM) { //Actually a PSUM cannot be broadcast. But we put this for compatibility
                psum_fifos[i]->push(pck_new);
            }          
            else {  //INPUT OR WEIGHT
                //Seting iteration of the package
                pck_new->setIterationK(pck->getIterationK()); //Used to avoid sending packages from a certain iteration without performing the previous.
                input_fifos[i]->push(pck_new);
            }     
           
        }
    }

    // UNICAST PACKAGE  单播 
    else if(pck->isUnicast()) {
        //We only have to send the weight to one port and change the destination to adapt it to the subgroup
        unsigned int dest = pck->get_unicast_dest(); //This is according to ALL the mswitches. 
        unsigned int input_port = dest / this->ms_size_per_input_port;  // 属于哪个端口
        unsigned int local_dest = dest % this->ms_size_per_input_port;  // 端口对应的第几个乘法器 
        //Creating the package  // 创建数据包的副本 
        std::shared_ptr<DataPackage> pck_new = std::make_shared<DataPackage>(pck->get_size_package(), pck->get_data(), pck->get_data_type(), input_port, UNICAST, local_dest); //size, data, type, source (port), UNICAST, dest_local
        //Sending to the fifo corresponding with port input_port
        if(pck->get_data_type() == PSUM) { 
            psum_fifos[input_port]->push(pck_new);
        }          
        else {  //INPUT OR WEIGHT
            input_fifos[input_port]->push(pck_new);
            pck_new->setIterationK(pck->getIterationK());
            //std::cout<<"new data : "<<pck_new->getIterationK()<<std::endl;
        }

    }

    //MULTICAST PACKAGE 多播 
    else { //The package is multicast and then we have to send the package to several ports
        const bool* dest = pck->get_dests();  //One position for mswitch in all the msarray
        bool thereis_receiver;
        for(int i=0; i<this->n_read_ports; i++) { //Checking each port with size this->ms_size_per_input_port each. Total=ms_size
            unsigned int port_index = i*this->ms_size_per_input_port;
            thereis_receiver = false; // To know at the end if the group
            bool* local_dest = new bool[this->ms_size_per_input_port]; //Local destination array for the subtree corresponding with the port i
            for(int j=0; j<this->ms_size_per_input_port; j++) {  //For each ms in the group of the port i
                local_dest[j] = dest[port_index + j]; //Copying the subarray
                if(local_dest[j] == true) {
                    thereis_receiver=true; // To avoid iterating again to know whether the data have to be sent to the port or not.
                }
            }

            if(thereis_receiver) { //If this port have at least one ms to true then we send the data to this port i
                std::shared_ptr<DataPackage> pck_new = std::make_shared<DataPackage>(pck->get_size_package(), pck->get_data(), pck->get_data_type(), i, MULTICAST, local_dest, this->ms_size_per_input_port); 
                if(pck->get_data_type() == PSUM) {
                    psum_fifos[i]->push(pck_new);
                }
 
                else {
                    pck_new->setIterationK(pck->getIterationK());
                    input_fifos[i]->push(pck_new);
                    
                }
            }
            else {
                delete[] local_dest; //If this vector is not sent we remove it.
            }
        }
    }

    // 已经根据所需创建了数据包的副本，所以可以删除掉
    // delete pck; // We have created replicas of the package for the ports needed so we can delete this
} 

void OSMeshSDMemory::send() {
    //Iterating over each port and if there is data in its fifo we send it. We give priority to the psums
    // 遍历每个端口，如果其fifo中有数据，就发送。优先考虑部分和

    for(int i=0; i<this->n_read_ports; i++) { 
        std::vector<std::shared_ptr<DataPackage>> pck_to_send; 
        
        if(!this->psum_fifos[i]->isEmpty()) { //If there is something we may send data though the connection
            std::shared_ptr<DataPackage> pck = psum_fifos[i]->pop();
            #ifdef DEBUG_MEM_INPUT
                std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending a psum through input port " << i  << std::endl;
            #endif
            pck_to_send.push_back(pck);
            this->sdmemoryStats.n_SRAM_read_ports_psums_use[i]++; //To track information
            //Sending to the connection
            this->read_connections[i]->send(pck_to_send);  // 发送数据到read_connections[i]中
        }
        //If psums fifo is empty then input fifo is checked. If psum is not empty then else do not compute. Important this ELSE to give priority to the psums and do not send more than 1 pck
        else if(!this->input_fifos[i]->isEmpty()) {
            //If the package belongs to a certain k iteration but the previous k-1 iteration has not finished the package is not sent
            std::shared_ptr<DataPackage> pck = input_fifos[i]->front(); //Front because we are not sure if we have to send it. // 不从fifo中删除 
           
            if(pck->get_data_type()==WEIGHT) {
                this->sdmemoryStats.n_SRAM_read_ports_weights_use[i]++; //To track information
                #ifdef DEBUG_MEM_INPUT
                    std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending a WEIGHT through input port " << i << std::endl;
                #endif
            }  
            else {
                this->sdmemoryStats.n_SRAM_read_ports_inputs_use[i]++; //To track information
                #ifdef DEBUG_MEM_INPUT
                    std::cout << "[MEM_INPUT] Cycle " << local_cycle << ", Sending an INPUT ACTIVATION through input port " << i << std::endl;
                #endif
            }

            pck_to_send.push_back(pck); //storing into the vector data type structure used in class Connection 
            this->read_connections[i]->send(pck_to_send); //Sending the input or weight through the connection
            input_fifos[i]->pop(); //pulling from fifo
        }
            
    }

}

//TODO Remove this connection
void OSMeshSDMemory::receive() { //TODO control if there is no space in queue
    //std::cout<<"debug 3"<<std::endl;
    // this->write_connection 是查找表和内存的接口 
    // if(this->write_connection->existPendingData()) {
    //     std::cout<<"debug 4"<<std::endl;
    //     std::vector<std::shared_ptr<DataPackage>> data_received = write_connection->receive();
    //     for(int i=0; i<data_received.size(); i++) {
    //         write_fifo->push(data_received[i]);
    //     }
    // } 
    // write_port_connections 是总线和内存的接口
    //std::cout<<"the size of write_port_connections : "<<write_port_connections.size()<<std::endl;
    for(int i=0; i<write_port_connections.size(); i++) { //For every write port
        //std::cout<<"debug 5"<<std::endl;
        if(write_port_connections[i]->existPendingData()) {
            //std::cout<<"debug 6"<<std::endl;
            std::vector<std::shared_ptr<DataPackage>> data_received = write_port_connections[i]->receive();
            for(int i=0; i<data_received.size(); i++) {
            //std::cout<<"the length of dataPackage(OSMeshSDMemory.cpp) : "<<data_received.size()<<std::endl;
            write_fifo->push(data_received[i]);
            }
        }   
    }
}

void OSMeshSDMemory::printStats(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"SDMemoryStats\" : {" << std::endl; //TODO put ID
    this->sdmemoryStats.print(out, indent+IND_SIZE);
    out << ind(indent) << "}"; //Take care. Do not print endl here. This is parent responsability
}

void OSMeshSDMemory::printEnergy(std::ofstream& out, unsigned int indent) {
    /*
        This component prints:
            - The number of SRAM reads
            - The number of SRAM writes

        Note that the number of times that each port is used is not shown. This is so because the use of those wires are
        taken into account in the CollectionBus and in the DSNetworkTop
   */

   counter_t reads = this->sdmemoryStats.n_SRAM_weight_reads + this->sdmemoryStats.n_SRAM_input_reads + this->sdmemoryStats.n_SRAM_psum_reads;
   counter_t writes = this->sdmemoryStats.n_SRAM_psum_writes;
   out << ind(indent) << "GLOBALBUFFER READ=" << reads; //Same line
   out << ind(indent) << " WRITE=" << writes << std::endl;
        
}