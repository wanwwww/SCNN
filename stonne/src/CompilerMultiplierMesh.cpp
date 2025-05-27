#include "CompilerMultiplierMesh.h"
#include "Tile.h"
#include "utility.h"
#include <math.h>
#include "types.h"
#include <assert.h>
#include "cpptoml.h"

void CompilerMultiplierMesh::configureSignals(Tile* current_tile, DNNLayer* dnn_layer, unsigned int ms_rows, unsigned int ms_cols) {
    assert(current_tile->get_T_K() <= ms_cols); //Number of filters
    assert(current_tile->get_T_X_()*current_tile->get_T_Y_() <= ms_rows); //Number of conv windows

    this->current_tile = current_tile;
    this->dnn_layer = dnn_layer;
    this->ms_rows = ms_rows;
    this->ms_cols = ms_cols;
    this->signals_configured = true;
    //Configuring Multiplier switches
    //std::cout<<"debug in compiler"<<std::endl;
    this->generate_ms_signals(ms_rows, ms_cols);
    //std::cout<<"debug in compiler end"<<std::endl;
}

void CompilerMultiplierMesh::configureSparseSignals(std::vector<SparseVN> sparseVNs, DNNLayer* dnn_layer, unsigned int num_ms) {
    assert(false); //TPU implementation does not allow sprsity due to its rigit nature
}


void CompilerMultiplierMesh::generate_ms_signals(unsigned int ms_rows, unsigned int ms_cols) {
    forwarding_bottom_enabled.clear();
    forwarding_right_enabled.clear();
    ms_vn_configuration.clear();
    // PE阵列中使用到的行和列 
    unsigned int rows_used = this->current_tile->get_T_X_()*this->current_tile->get_T_Y_();
    unsigned int cols_used = this->current_tile->get_T_K();
    // std::cout<<"rows_used : "<<rows_used<<std::endl;
    // std::cout<<"cols_used : "<<cols_used<<std::endl;
    // std::cout<<"ms_rows : "<<ms_rows<<std::endl;
    // std::cout<<"ms_cols : "<<ms_cols<<std::endl;
    //Bottom and right signals 
    for(int i=0; i<ms_rows; i++) {
        for(int j=0; j<ms_cols; j++) {
            // std::cout<<"begin =============="<<std::endl;
            // std::cout<<"i : "<<i<<std::endl;
            // std::cout<<"j : "<<j<<std::endl;

            std::pair<int,int> ms_index(i,j);

            if((i < rows_used) && (j < cols_used)) { // VN的配置，为每个用到的乘法器设置一个编号 
                //std::cout<<"debug"<<std::endl;
                unsigned int VN = i*cols_used+j;
                ms_vn_configuration[ms_index]=VN;
                //std::cout<<"====================== "<<VN<<" ==================================="<<std::endl;
            }
            //std::cout<<"1"<<std::endl;

            if((i < (rows_used-1)) && (j < cols_used)) {
                forwarding_bottom_enabled[ms_index]=true;
            }
            else {
                forwarding_bottom_enabled[ms_index]=false;
            }
            //std::cout<<"2"<<std::endl;

            if((j < (cols_used-1)) && (i < rows_used)) {
                //std::cout<<"debug1"<<std::endl;
                forwarding_right_enabled[ms_index]=true;
                //std::cout<<"debug2"<<std::endl;
            }
            else {
                forwarding_right_enabled[ms_index]=false;
            }
            //std::cout<<"3"<<std::endl;
	    }
        //std::cout<<"over ==============="<<std::endl;
    }
              
}
