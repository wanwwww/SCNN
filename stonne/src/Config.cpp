//Created on 22/10/2019 by Francisco Munoz Martinez

#include "Config.h"
#include <iostream>
#include "types.h"
#include "utility.h"
#include "cpptoml.h"

Config::Config() {
    this->reset();
}

void Config::loadFile(std::string config_file) {

    auto config = cpptoml::parse_file(config_file);

    //General parameters
    auto print_stats_enabled_conf = config->get_as<bool>("print_stats_enabled");  //print_stats_enabled
    if(print_stats_enabled_conf) {
        this->print_stats_enabled = *print_stats_enabled_conf;
    }

    // SNN parameters
    auto V_th_conf = config->get_qualified_as<unsigned int>("SNN.V_th");
    if(V_th_conf){
        this->V_th = *V_th_conf;
    }

    auto Timestamp_conf = config->get_qualified_as<unsigned int>("SNN.Timestamp");
    if(Timestamp_conf){
        this->Timestamp = *Timestamp_conf;
    }

    auto pooling_type_conf = config->get_qualified_as<std::string>("SNN.pooling_type");
    if(pooling_type_conf){
        if(*pooling_type_conf == "max"){
            this->pooling_type = MAXPOOLING;
        } else if(*pooling_type_conf =="average"){
            this->pooling_type = AVERAGEPOOLING;
        } else{
            std::cerr << "Invalid pooling type in config file." << std::endl;
        }
    }

    // DRAM 
    auto input_dram_size_conf = config->get_qualified_as<unsigned int>("DRAM.input_dram_size");
    if(input_dram_size_conf){
        this->m_DRAMCfg.input_dram_size = *input_dram_size_conf;
    }

    auto weight_dram_size_conf = config->get_qualified_as<unsigned int>("DRAM.weight_dram_size");
    if(weight_dram_size_conf){
        this->m_DRAMCfg.weight_dram_size = *weight_dram_size_conf;
    }

    auto output_dram_size_conf = config->get_qualified_as<unsigned int>("DRAM.output_dram_size");
    if(output_dram_size_conf){
        this->m_DRAMCfg.output_dram_size = *output_dram_size_conf;
    }

    // On-chip Buffer
    auto input_buffer_size_conf = config->get_qualified_as<unsigned int>("On_Chip_Buffer.input_buffer_size");
    if(input_buffer_size_conf){
        this->m_BufferCfg.input_buffer_size = *input_buffer_size_conf;
    }

    auto weight_buffer_size_conf = config->get_qualified_as<unsigned int>("On_Chip_Buffer.weight_buffer_size");
    if(weight_buffer_size_conf){
        this->m_BufferCfg.weight_buffer_size = *weight_buffer_size_conf;
    }

    auto neuron_state_buffer_size_conf = config->get_qualified_as<unsigned int>("On_Chip_Buffer.neuron_state_buffer_size");
    if(neuron_state_buffer_size_conf){
        this->m_BufferCfg.neuron_state_buffer_size = *neuron_state_buffer_size_conf;
    }

    auto output_buffer_size_conf = config->get_qualified_as<unsigned int>("On_Chip_Buffer.output_buffer_size");
    if(output_buffer_size_conf){
        this->m_BufferCfg.output_buffer_size = *output_buffer_size_conf;
    }


    //MSNetwork Configuration Parameters
    auto ms_rows_conf = config->get_qualified_as<unsigned int>("MSNetwork.ms_rows");
    if(ms_rows_conf) {
        this->m_MSNetworkCfg.ms_rows=*ms_rows_conf;
    }

    auto ms_cols_conf = config->get_qualified_as<unsigned int>("MSNetwork.ms_cols");
    if(ms_cols_conf) {
        this->m_MSNetworkCfg.ms_cols=*ms_cols_conf;
    }

    //MSwitch Configuration parameters
    auto mswitch_latency_conf = config->get_qualified_as<unsigned int>("MSwitch.latency"); //latency
    if(mswitch_latency_conf) {
        this->m_MSwitchCfg.latency = *mswitch_latency_conf; 
    }

    auto mswitch_input_ports_conf = config->get_qualified_as<unsigned int>("MSwitch.input_ports"); //input_ports
    if(mswitch_input_ports_conf) {
        this->m_MSwitchCfg.input_ports = *mswitch_input_ports_conf;
    }

    auto mswitch_output_ports_conf = config->get_qualified_as<unsigned int>("MSwitch.output_ports"); //output_ports
    if(mswitch_output_ports_conf) {
        this->m_MSwitchCfg.output_ports = *mswitch_output_ports_conf;
    }

    auto mswitch_forwarding_ports_conf = config->get_qualified_as<unsigned int>("MSwitch.forwarding_ports"); //forwarding_ports
    if(mswitch_forwarding_ports_conf) {
        this->m_MSwitchCfg.forwarding_ports = *mswitch_forwarding_ports_conf;
    }

    auto mswitch_port_width_conf = config->get_qualified_as<unsigned int>("MSwitch.port_width"); //port_width
    if(mswitch_port_width_conf) {
        this->m_MSwitchCfg.port_width = *mswitch_port_width_conf;
    }

    auto mswitch_buffers_capacity_conf = config->get_qualified_as<unsigned int>("MSwitch.buffers_capacity"); //buffers_capacity
    if(mswitch_buffers_capacity_conf) {
        this->m_MSwitchCfg.buffers_capacity = *mswitch_buffers_capacity_conf;
    }

    //ReduceNetwork Configuration Parameters
    auto accumulation_buffer_enabled_conf = config->get_qualified_as<unsigned int>("ReduceNetwork.accumulation_buffer_enabled");
    if(accumulation_buffer_enabled_conf) {
        this->m_ASNetworkCfg.accumulation_buffer_enabled = *accumulation_buffer_enabled_conf;
    }
    

    //ASwitch Configuration Parameters
    auto aswitch_buffers_capacity_conf = config->get_qualified_as<unsigned int>("ASwitch.buffers_capacity");  //Buffers_capacity
    if(aswitch_buffers_capacity_conf) {
        this->m_ASwitchCfg.buffers_capacity = *aswitch_buffers_capacity_conf;
    } 

    auto aswitch_input_ports_conf = config->get_qualified_as<unsigned int>("ASwitch.input_ports");    //input ports
    if(aswitch_input_ports_conf) {
        this->m_ASwitchCfg.input_ports = *aswitch_input_ports_conf;
    }

    auto aswitch_output_ports_conf = config->get_qualified_as<unsigned int>("ASwitch.output_ports");  //output ports
    if(aswitch_output_ports_conf) {
        this->m_ASwitchCfg.output_ports = *aswitch_output_ports_conf;
    }

    auto aswitch_forwarding_ports_conf = config->get_qualified_as<unsigned int>("ASwitch.forwarding_ports");  //forwarding ports
    if(aswitch_forwarding_ports_conf) {
        this->m_ASwitchCfg.forwarding_ports = *aswitch_forwarding_ports_conf;
    }

    auto aswitch_port_width_conf = config->get_qualified_as<unsigned int>("ASwitch.port_width");  //port width
    if(aswitch_port_width_conf) {
        this->m_ASwitchCfg.port_width = *aswitch_port_width_conf;
    }

    auto aswitch_latency_conf = config->get_qualified_as<unsigned int>("ASwitch.latency");  //latency
    if(aswitch_latency_conf) {
        this->m_ASwitchCfg.latency = *aswitch_latency_conf;
    }

    //UpdateSwitch Configuration Parameters
    auto update_port_conf = config->get_qualified_as<unsigned int>("UpdateSwitch.port_width");
    if(update_port_conf){
        this->m_UpdateSwitchCfg.port_width = *update_port_conf;
    }

    auto update_buffers_conf = config->get_qualified_as<unsigned int>("UpdateSwitch.buffers_capacity");
    if(update_buffers_conf){
        this->m_UpdateSwitchCfg.buffers_capacity = *update_buffers_conf;
    }

    auto update_latency_conf = config->get_qualified_as<unsigned int>("UpdateSwitch.latency");
    if(update_latency_conf){
        this->m_UpdateSwitchCfg.latency = *update_latency_conf;
    }
    
    //SDMemory Configuration Parameters
    auto sdmemory_dn_bw_conf = config->get_qualified_as<unsigned int>("SDMemory.dn_bw");  //DN_BW
    if(sdmemory_dn_bw_conf) {
        this->m_SDMemoryCfg.n_read_ports = *sdmemory_dn_bw_conf;   
    }

    auto sdmemory_rn_bw_conf = config->get_qualified_as<unsigned int>("SDMemory.rn_bw");  //RN_BW
    if(sdmemory_rn_bw_conf) {
        this->m_SDMemoryCfg.n_write_ports = *sdmemory_rn_bw_conf; 
    }

    auto sdmemory_port_width_conf = config->get_qualified_as<unsigned int>("SDMemory.port_width");
    if(sdmemory_port_width_conf) {
        this->m_SDMemoryCfg.port_width = *sdmemory_port_width_conf;
    }

}

void Config::reset() {
// SNN parameter
    V_th = 0;
    Timestamp = 1;
    layer_type = CONV;
    pooling_type = MAXPOOLING; 

//General parameters
    print_stats_enabled=true;

// ---------------------------------------------------------
// DSNetwork Configuration Parameters
// ---------------------------------------------------------
//    m_DSNetworkCfg.n_switches_traversed_by_cycle=23; //From paper. This is not implemented yet anyway.

// ---------------------------------------------------------
// DSwitch Configuration Parameters
// ---------------------------------------------------------

    // //There is nothing yet
    // m_DSwitchCfg.latency = 1; //Actually is less than 1. We do not implement this either
    // m_DSwitchCfg.input_ports = 1;
    // m_DSwitchCfg.output_ports=2;
    // m_DSwitchCfg.port_width=16;  //Size in bits

// ---------------------------------------------------------
// DRAM configuration parameters
// ---------------------------------------------------------
    m_DRAMCfg.input_dram_size = 2;  // size in MB
    m_DRAMCfg.weight_dram_size = 3;
    m_DRAMCfg.output_dram_size = 2;

// ---------------------------------------------------------
// On-chip buffer configuration parameters
// ---------------------------------------------------------
    m_BufferCfg.input_buffer_size = 32;  // size in KB
    m_BufferCfg.weight_buffer_size = 128;
    m_BufferCfg.neuron_state_buffer_size = 8;
    m_BufferCfg.output_buffer_size = 1;

// ---------------------------------------------------------
// MSNetwork Configuration Parameters
// ---------------------------------------------------------
    //m_MSNetworkCfg.multiplier_network_type=LINEAR;
    //m_MSNetworkCfg.ms_size=64;
    m_MSNetworkCfg.ms_rows=4; //Not initialized
    m_MSNetworkCfg.ms_cols=4; //Not initalized

// ---------------------------------------------------------
// MSwitch Configuration Parameters
// ---------------------------------------------------------
    m_MSwitchCfg.latency=1; //Latency in ns not implemented
    m_MSwitchCfg.input_ports=2;
    m_MSwitchCfg.output_ports=1;
    m_MSwitchCfg.forwarding_ports=1;
    m_MSwitchCfg.port_width=16;
    m_MSwitchCfg.buffers_capacity=2048;


// ---------------------------------------------------------
// ASNetwork Configuration Parameters
// ---------------------------------------------------------
    //m_ASNetworkCfg.reduce_network_type=ASNETWORK;
    m_ASNetworkCfg.accumulation_buffer_enabled=1;

// ---------------------------------------------------------
// ASwitch Configuration Parameters
// ---------------------------------------------------------
    m_ASwitchCfg.buffers_capacity=256;
    m_ASwitchCfg.input_ports=2;
    m_ASwitchCfg.output_ports=1;
    m_ASwitchCfg.forwarding_ports=1;
    m_ASwitchCfg.port_width=16;
    m_ASwitchCfg.latency=1;

// ---------------------------------------------------------
// LookUpTable Configuration Parameters
// ---------------------------------------------------------
    // m_LookUpTableCfg.latency=1; //Latency in ns not implemented yet
    // m_LookUpTableCfg.port_width=1;

// ---------------------------------------------------------
// UpdateSwitch Configuration Parameters
// ---------------------------------------------------------
    m_UpdateSwitchCfg.buffers_capacity = 256;
    m_UpdateSwitchCfg.latency = 1;
    m_UpdateSwitchCfg.port_width = 16;

// ---------------------------------------------------------
// SDMemory Controller Configuration Parameters
// ---------------------------------------------------------
    //m_SDMemoryCfg.mem_controller_type=MAERI_DENSE_WORKLOAD;
    m_SDMemoryCfg.write_buffer_capacity=256;
    m_SDMemoryCfg.n_read_ports=4; 
    m_SDMemoryCfg.n_write_ports=4; 
    m_SDMemoryCfg.port_width=16;

}

// bool Config::sparsitySupportEnabled() {
//     return m_SDMemoryCfg.mem_controller_type==SIGMA_SPARSE_GEMM; //If the controller is sparse, then sparsity is allowed
// }


// bool Config::convOperationSupported() {
//     return m_SDMemoryCfg.mem_controller_type==MAERI_DENSE_WORKLOAD;
// }


/** PRINTING FUNCTIONS **/

// std::string ind(unsigned int indent)   Generates a string consisting of a specified number of spaces, used to indent code or text.

// -----------------------------------------------------------------------------------------------
// Config printing function
// -----------------------------------------------------------------------------------------------
void Config::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"hardwareConfiguration\" : {" << std::endl;
        //Printing general parameters
        out << ind(indent+IND_SIZE) << "\"print_stats_enabled\" : " << this->print_stats_enabled << "," << std::endl;  
        //Printing specific parameters of each unit
        //this->m_DSNetworkCfg.printConfiguration(out, indent+IND_SIZE); // IND_SIZE = 4
        //out << "," << std::endl;
        //this->m_DSwitchCfg.printConfiguration(out, indent+IND_SIZE);
        //out << "," << std::endl;

        this->m_DRAMCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;

        this->m_BufferCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;

        this->m_MSNetworkCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_MSwitchCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_ASNetworkCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_ASwitchCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        //this->m_LookUpTableCfg.printConfiguration(out, indent+IND_SIZE);
        //out << "," << std::endl;
        this->m_UpdateNetworkCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_UpdateSwitchCfg.printConfiguration(out, indent+IND_SIZE);
        out << "," << std::endl;
        this->m_SDMemoryCfg.printConfiguration(out, indent+IND_SIZE);
        out  << std::endl; //Take care of the comma since this is the last one
    out << ind(indent) << "}";
}


// -----------------------------------------------------------------------------------------------
// DSNetworkConfig printing function
// -----------------------------------------------------------------------------------------------
// void DSNetworkConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
//     out << ind(indent) << "\"DSNetwork\" : {" << std::endl;
//         //Printing DSNetwork configuration
//         out << ind(indent+IND_SIZE) << "\"n_switches_traversed_by_cycle\" : " << this->n_switches_traversed_by_cycle << std::endl;
//     out << ind(indent) << "}";
// }


// -----------------------------------------------------------------------------------------------
// DSwitchConfig printing function
// -----------------------------------------------------------------------------------------------
// void DSwitchConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
//     out << ind(indent) << "\"DSwitch\" : {" << std::endl;
//         out << ind(indent+IND_SIZE) << "\"latency\" : " << this->latency << "," << std::endl; 
//         out << ind(indent+IND_SIZE) << "\"input_ports\" : " << this->input_ports << "," << std::endl;
//         out << ind(indent+IND_SIZE) << "\"output_ports\" : " << this->output_ports << "," << std::endl;
//         out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width  << std::endl;
//     out << ind(indent) << "}";
// }


// -----------------------------------------------------------------------------------------------
// DRAMConfig printing function
// -----------------------------------------------------------------------------------------------
void DRAMConfig::printConfiguration(std::ofstream& out, unsigned int indent){
    out << ind(indent) << "\"DRAM Buffer\" : {"<<std::endl; // ind是一个自定义的函数，用于生成一个缩进
    out << ind(indent+IND_SIZE) << "\"input_dram_size\" : " << this->input_dram_size << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"weight_dram_size\" : " << this->weight_dram_size << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"output_dram_size\" : " << this->output_dram_size << "," << std::endl;
    out << ind(indent) << "}";
}

// -----------------------------------------------------------------------------------------------
// BufferConfig printing function
// -----------------------------------------------------------------------------------------------
void BufferConfig::printConfiguration(std::ofstream& out, unsigned int indent){
    out << ind(indent) << "\"On-chip Buffer\" : {"<<std::endl; // ind是一个自定义的函数，用于生成一个缩进
    out << ind(indent+IND_SIZE) << "\"input_buffer_size\" : " << this->input_buffer_size << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"weight_buffer_size\" : " << this->weight_buffer_size << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"neuron_state_buffer_size\" : " << this->neuron_state_buffer_size << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"output_buffer_size\" : " << this->output_buffer_size << "," << std::endl;
    out << ind(indent) << "}";
}


// -----------------------------------------------------------------------------------------------
// MSNetworkConfig printing function
// -----------------------------------------------------------------------------------------------
void MSNetworkConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"MSNetwork\" : {" << std::endl;
    //out << ind(indent+IND_SIZE) << "\"multiplier_network_type\" : " << "\"" << get_string_multiplier_network_type(this->multiplier_network_type) << "\"" << ","  << std::endl;
    out << ind(indent+IND_SIZE) << "\"ms_rows\" : " << this->ms_rows << "," << std::endl;
    out << ind(indent+IND_SIZE) << "\"ms_columns\" : " << this->ms_cols << "," << std::endl;
    //out << ind(indent+IND_SIZE) << "\"ms_size\" : " << this->ms_size  << std::endl;
    out << ind(indent) << "}";

}


// -----------------------------------------------------------------------------------------------
// MSwitchConfig printing function
// -----------------------------------------------------------------------------------------------
void MSwitchConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"MSwitch\" : {" << std::endl;
        out << ind(indent+IND_SIZE) << "\"latency\" : " << this->latency << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"input_ports\" : " << this->input_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"output_ports\" : " << this->output_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"forwarding_ports\" : " << this->forwarding_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"buffers_capacity\" : " << this->buffers_capacity  << std::endl;
    out << ind(indent) << "}";

}


// -----------------------------------------------------------------------------------------------
// ASNetworkConfig printing function
// -----------------------------------------------------------------------------------------------
void ASNetworkConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"ReduceNetwork\" : {" << std::endl;
        //out << ind(indent+IND_SIZE) << "\"reduce_network_type\" : " << "\"" << get_string_reduce_network_type(this->reduce_network_type) << "\"" << ","  << std::endl;
        out << ind(indent+IND_SIZE) << "\"accumulation_buffer_enabled\" : " << this->accumulation_buffer_enabled  << std::endl;
    out << ind(indent) << "}";

}


// -----------------------------------------------------------------------------------------------
// ASwitchConfig printing function
// -----------------------------------------------------------------------------------------------
void ASwitchConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"ASwitch\" : {" << std::endl;
        out << ind(indent+IND_SIZE) << "\"latency\" : " << this->latency << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"input_ports\" : " << this->input_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"output_ports\" : " << this->output_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"forwarding_ports\" : " << this->forwarding_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"buffers_capacity\" : " << this->buffers_capacity  << std::endl;
    out << ind(indent) << "}";

}


// -----------------------------------------------------------------------------------------------
// LookUpTaleConfig printing function
// -----------------------------------------------------------------------------------------------
// void LookUpTableConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
//     out << ind(indent) << "\"LookUpTable\" : {" << std::endl;
//         out << ind(indent+IND_SIZE) << "\"latency\" : " << this->latency << "," << std::endl;
//         out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width  << std::endl;
//     out << ind(indent) << "}";

// }


// -----------------------------------------------------------------------------------------------
// UpdateNetworkConfig printing function
// -----------------------------------------------------------------------------------------------
void UpdateNetworkConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"UpdateNetwork\" : {" << std::endl;
        //out << ind(indent+IND_SIZE) << "\"mem_controller_type\" : " << "\"" << get_string_memory_controller_type(this->mem_controller_type) << "\""  << "," << std::endl;
        out << ind(indent+IND_SIZE) << "null" << std::endl; 
        //out << "null"<< std::endl;
    out << ind(indent) << "}";
}

// -----------------------------------------------------------------------------------------------
// UpdateSwitchConfig printing function
// -----------------------------------------------------------------------------------------------
void UpdateSwitchConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"UpdateSwitch\" : {" << std::endl;
        //out << ind(indent+IND_SIZE) << "\"mem_controller_type\" : " << "\"" << get_string_memory_controller_type(this->mem_controller_type) << "\""  << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"latency\" : " << this->latency << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"buffers_capacity\" : " << this->buffers_capacity << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width << std::endl;
    out << ind(indent) << "}";
}

// -----------------------------------------------------------------------------------------------
// SDMemoryConfig printing function
// -----------------------------------------------------------------------------------------------
void SDMemoryConfig::printConfiguration(std::ofstream& out, unsigned int indent) {
    out << ind(indent) << "\"SDMemory\" : {" << std::endl;
        //out << ind(indent+IND_SIZE) << "\"mem_controller_type\" : " << "\"" << get_string_memory_controller_type(this->mem_controller_type) << "\""  << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"write_buffers_capacity\" : " << this->write_buffer_capacity << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"dn_bw\" : " << this->n_read_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"rn_bw\" : " << this->n_write_ports << "," << std::endl;
        out << ind(indent+IND_SIZE) << "\"port_width\" : " << this->port_width << std::endl;
    out << ind(indent) << "}";

}
