#include <iostream>
#include "DNNLayer.h"
#include "STONNEModel.h"
#include <assert.h>
#include "types.h"
#include <cstring>
#include <algorithm> // 用于 std::max


void sequential_layer(unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G,  unsigned int N, unsigned int X, unsigned int Y, unsigned int strides, 
int* input, int* filters, int* outputs, int* neuron_state, int Vth, int Timestamp, bool pooling_enabled) {

    unsigned int OX=(X - R + strides) / strides;
    unsigned int OY=(Y - S + strides) / strides;
    //std::cout<<"output rows : "<<OX<<std::endl;
    //std::cout<<"output cols : "<<OY<<std::endl;
    K=K/G;
    C=C/G;
    unsigned int output_size_n = G*K*OX*OY;  // OX*OY行 K列 
    unsigned int input_size_n = G*C*X*Y;
    unsigned int filter_size=R*S*C;
    unsigned int size_oy=OY*K*G;  // 输出特征图中一行的数据个数 
    unsigned int size_y=Y*G*C;  // 输入特征图中一行的数据个数

    int* outputs_without_pooling = new int[Timestamp*output_size_n]; // 没有池化的输出，多个时间步 
    for(int i=0;i<Timestamp*output_size_n;i++){
        outputs_without_pooling[i] = 0;
    }

    for(int t=0;t<Timestamp;t++){

        // std::cout<<" current timestamp is : "<<t<<std::endl;
        // std::cout<<std::endl;

        for(int n=0; n<N; n++) {
            //std::cout<<"n : "<<n<<std::endl;
            for(int g=0; g<G; g++) {
                //std::cout<<"g : "<<g<<std::endl;
                for(int k=0; k<K; k++) {  // 遍历每个输出通道 
                    //std::cout<<"k : "<<k<<std::endl;
                    for(int ox=0; ox<OX; ox++) { // 遍历输出的每一行
                        for(int oy=0; oy<OY; oy++) {  // 遍历每个输出的每一列
                            //neuron_state[n*output_size_n + ox*size_oy + oy*K*G + g*K + k]=0;
                            for(int c=0; c<C;c++) {
                                for(int r=0;r<R;r++) {
                                    for(int s=0;s<S;s++) {
                                        // std::cout<<"s : "<<s<<std::endl;
                                        // std::cout<<"debug in testbench"<<std::endl;
                                        // std::cout<<ox*size_oy + oy*K*G + k<<std::endl;
                                        // std::cout<<t*X*Y*C+ox*strides*size_y + oy*strides*C*G + r*size_y + s*C*G + c<<std::endl;
                                        // std::cout<<g*K*filter_size + k*filter_size + r*S*C + s*C + c<<std::endl;
                                        // std::cout<<"input[0] : "<<input[0]<<std::endl;
                                        // std::cout<<"filters[0] : "<<filters[0]<<std::endl;
                                        // std::cout<<"neuron_state[0] : "<<neuron_state[0]<<std::endl;
                                        //neuron_state[n*output_size_n + ox*size_oy + oy*K*G + g*K + k] += input[t*filter_size*OX*OY+n*input_size_n+ ox*strides*size_y + oy*strides*C*G + r*size_y + s*C*G + g*C + c]*filters[g*K*filter_size + k*filter_size + r*S*C + s*C + c];
                                        neuron_state[ox*size_oy + oy*K*G + k] += input[t*X*Y*C+ox*strides*size_y + oy*strides*C*G + r*size_y + s*C*G + c]*filters[g*K*filter_size + k*filter_size + r*S*C + s*C + c];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        

        // 一个时间步的计算结束后，得到该时间步产生的脉冲，以及更新后的膜电位 
        for(int i=0;i<output_size_n;i++){
            outputs_without_pooling[t*output_size_n+i] = 0;
            if(neuron_state[i] >= Vth){
                neuron_state[i] = 0;
                outputs_without_pooling[t*output_size_n+i] = 1;
            }
        }

        // std::cout<<"the output spike without pooling : "<<std::endl;
        // for(int i=0;i<OX*OY;i++){
        //     for(int j=0;j<K;j++){
        //         std::cout<<outputs_without_pooling[t*output_size_n+i*K+j]<<" ";
        //     }
        //     std::cout<<std::endl;
        // }

        // std::cout<<"the output Vth : "<<std::endl;
        // for(int i=0;i<OX*OY;i++){
        //     for(int j=0;j<K;j++){
        //         std::cout<<neuron_state[i*K+j]<<" ";
        //     }
        //     std::cout<<std::endl;
        // }

        // 根据是否启用池化模块，对输出进行计算 

        if(pooling_enabled){

            for(int i=1;i<OX*OY;i=i+2){
                for(int j=1;j<K;j=j+2){
                    int spike_sum = outputs_without_pooling[t*output_size_n+i*K+j] + outputs_without_pooling[t*output_size_n+i*K+j-1] + outputs_without_pooling[t*output_size_n+i*K+j-K] + outputs_without_pooling[t*output_size_n+i*K+j-K-1];
                    if(spike_sum>0){
                        outputs[t*output_size_n/4 + i/2*K/2 + j/2] = 1;
                    } else {
                        outputs[t*output_size_n/4 + i/2*K/2 + j/2] = 0;
                    }
                }
            }

            // std::cout<<"the output spike with pooling : "<<std::endl;
            // for(int i=0;i<OX*OY/2;i++){
            //     for(int j=0;j<K/2;j++){
            //         std::cout<<outputs[t*output_size_n/4+i*K+j]<<" ";
            //     }
            //     std::cout<<std::endl;
            // }


        } else{

            for(int i=0;i<output_size_n;i++){
                outputs[t*output_size_n+i] = outputs_without_pooling[t*output_size_n+i];
            }

            // std::cout<<"the output spike without pooling : "<<std::endl;
            // for(int i=0;i<OX*OY;i++){
            //     for(int j=0;j<K;j++){
            //         std::cout<<outputs[t*output_size_n+i*K+j]<<" ";
            //     }
            //     std::cout<<std::endl;
            // }

        }
    }

    delete[] outputs_without_pooling;
}

void matrixMultiply(int M, int K, int N, int* input, int* weight, int* output, int* neuron_state, int V_th, int num_tile){

    // 输入尺寸：M*K
    // 权重尺寸：K*N
    // std::cout<<std::endl;
    // std::cout<<"================================================== cpu ============================================="<<std::endl;
    // std::cout<<std::endl;
    // std::cout<<"input : "<<std::endl;
    // for(int x=0; x<M; x++){
    //     for(int y=0; y<K; y++){
    //         std::cout<<input[x*K + y]<<"   ";
    //     }
    //     std::cout<<std::endl;
    // }
    // std::cout<<std::endl;
    // std::cout<<"weight : "<<std::endl;
    // for(int y=0; y<N; y++){
    //     for(int x=0; x<K; x++){
    //         std::cout<<weight[y*K+x]<<"   ";
    //     }
    //     std::cout<<std::endl;
    // }

    // std::cout<<"================================================== cpu ============================================="<<std::endl;
    // std::cout<<std::endl;
    
    int add_offset = num_tile*M*N;
    //std::cout<<"add_offset : "<<add_offset<<std::endl;

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            int sum = 0;
            for(int k=0; k<K; k++){
                sum += input[i*K+k] * weight[k+j*K];
            }
            sum += neuron_state[add_offset + i*N+j];
            if(sum>=V_th){
                output[add_offset + i*N+j] = 1;
                neuron_state[add_offset + i*N+j] = 0;
            }else {
                output[add_offset + i*N+j] = 0;
                neuron_state[add_offset + i*N+j] = sum;
            }
        }
    }

}

// X * Y * C 是输入维度
void pooling_compute(int X, int Y, int C, int R, int S, int stride, int* input, int* output){

    int outX = (X - R)/stride + 1;
    int outY = (Y - S)/stride + 1;

    for (int i = 0; i < outX; i++) {
        for (int j = 0; j < outY; j++) {
            for (int k = 0; k < C; k++) {
                // 由于输入数据为0或1，初始化最大值为0
                int max_val = 0;
                // 在池化窗口内做最大池化
                for (int r = 0; r < R; r++) {
                    for (int s = 0; s < S; s++) {
                        int in_x = i * stride + r;
                        int in_y = j * stride + s;
                        // 计算输入索引：输入按照 (X,Y,C) 顺序存储
                        int idx = (in_x * Y + in_y) * C + k;
                        int val = input[idx];
                        if (val > max_val)
                            max_val = val;
                    }
                }
                // 计算输出索引：输出按照 (outX,outY,C) 顺序存储
                int out_idx = (i * outY + j) * C + k;
                output[out_idx] = max_val;
            }
        }
    }

}

void conv_and_pooling_compute(int R, int S, int C, int K, int P, int strides, int X, int Y, int* input, int* filters, int* output, int* neuron_state, int V_th){

    int padded_X = X + 2 * P;
    int padded_Y = Y + 2 * P;

    int outX = (X + 2*P - R) / strides + 1;
    int outY = (Y + 2*P - S) / strides + 1;

    int* padded_input = new int[padded_X * padded_Y * C];
    int* conv_output = new int[outX * outY * K];

    // 加padding
    for(int i=0; i<padded_X; i++){
        for(int j=0; j<padded_Y; j++){
            for(int c=0; c<C; c++){
                if (i >= P && i < P + X && j >= P && j < P + Y) {
                    int input_index = ((i - P) * Y + (j - P)) * C + c;
                    int padded_input_index = (i * padded_Y + j) * C + c;
                    padded_input[padded_input_index] = input[input_index];
                } else {
                    // If outside the valid range, set to 0 (padding)
                    int padded_input_index = (i * padded_Y + j) * C + c;
                    padded_input[padded_input_index] = 0;
                }
            }
        }
    }

    for (int i = 0; i < outX; ++i) {
        for (int j = 0; j < outY; ++j) {
            for (int k = 0; k < K; ++k) {
                int sum = 0;
                // 对于当前卷积窗口，遍历卷积核所有位置及所有输入通道
                for (int r = 0; r < R; ++r) {
                    for (int s = 0; s < S; ++s) {
                        for (int c = 0; c < C; ++c) {
                            // 输入中对应的坐标
                            int in_row = i * strides + r;
                            int in_col = j * strides + s;
                            int padded_input_index = (in_row * padded_Y + in_col) * C + c;
                            
                            // 卷积核在存储时每个核占用连续的R*S*C个元素，
                            // 其中核的内部顺序为 (r, s, c)
                            int filter_index = ((r * S + s) * C + c) + k * (R * S * C);
                            
                            sum += padded_input[padded_input_index] * filters[filter_index];
                        }
                    }
                }
                // 输出的下标计算：输出数据存储顺序为 (outX, outY, K)
                int output_index = (i * outY + j) * K + k;
                sum += neuron_state[output_index]; // 累积上一个时间步的膜电位
                if(sum>=V_th){
                    conv_output[output_index] = 1;
                    neuron_state[output_index] = 0;
                } else {
                    conv_output[output_index] = 0;
                    neuron_state[output_index] = sum;
                }
            }
        }
    }

    // std::cout<<"Below is the convolution result for each channel : "<<std::endl;
    // for(int i=0; i<K; i++){
    //     std::cout<<"channel : "<<i<<std::endl;
    //     for(int m=0; m<outX; m++){
    //         for(int n=0; n<outY; n++){
    //             int index = (m*outY+n)*K + i;
    //             std::cout<<conv_output[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    // }

    int pooledX = outX / 2;
    int pooledY = outY / 2;
    for(int i = 0; i < pooledX; i++){
        for(int j = 0; j < pooledY; j++){
            for(int k = 0; k < K; k++){
                int i0 = i*2, j0 = j*2;
                int idx0 = ( (i0  )*outY + (j0  ) ) * K + k;
                int idx1 = ( (i0+1)*outY + (j0  ) ) * K + k;
                int idx2 = ( (i0  )*outY + (j0+1) ) * K + k;
                int idx3 = ( (i0+1)*outY + (j0+1) ) * K + k;

                int m = conv_output[idx0] | conv_output[idx1] | conv_output[idx2] | conv_output[idx3];
                int dst = (i * pooledY + j) * K + k;
                output[dst]       = m;
            }
        }
    }
    // std::cout<<"Below is the pooling result for each channel : "<<std::endl;
    // for(int i=0; i<K; i++){
    //     std::cout<<"channel : "<<i<<std::endl;
    //     for(int m=0; m<pooledX; m++){
    //         for(int n=0; n<pooledY; n++){
    //             int index = (m*pooledY+n)*K + i;
    //             std::cout<<output[index]<<"  ";
    //         }
    //         std::cout<<std::endl;
    //     }
    // }

    delete[] padded_input;
    delete[] conv_output;
}

void conv_compute(int R, int S, int C, int K, int P, int strides, int X, int Y, int* input, int* filters, int* output, int* neuron_state, int V_th){

    int padded_X = X + 2 * P;
    int padded_Y = Y + 2 * P;

    int* padded_input = new int[padded_X * padded_Y * C];

    // 加padding
    for(int i=0; i<padded_X; i++){
        for(int j=0; j<padded_Y; j++){
            for(int c=0; c<C; c++){
                if (i >= P && i < P + X && j >= P && j < P + Y) {
                    int input_index = ((i - P) * Y + (j - P)) * C + c;
                    int padded_input_index = (i * padded_Y + j) * C + c;
                    padded_input[padded_input_index] = input[input_index];
                } else {
                    // If outside the valid range, set to 0 (padding)
                    int padded_input_index = (i * padded_Y + j) * C + c;
                    padded_input[padded_input_index] = 0;
                }
            }
        }
    }

    int outX = (X + 2*P - R) / strides + 1;
    int outY = (Y + 2*P - S) / strides + 1;

    for (int i = 0; i < outX; ++i) {
        for (int j = 0; j < outY; ++j) {
            for (int k = 0; k < K; ++k) {
                int sum = 0;
                // 对于当前卷积窗口，遍历卷积核所有位置及所有输入通道
                for (int r = 0; r < R; ++r) {
                    for (int s = 0; s < S; ++s) {
                        for (int c = 0; c < C; ++c) {
                            // 输入中对应的坐标
                            int in_row = i * strides + r;
                            int in_col = j * strides + s;
                            int padded_input_index = (in_row * padded_Y + in_col) * C + c;
                            
                            // 卷积核在存储时每个核占用连续的R*S*C个元素，
                            // 其中核的内部顺序为 (r, s, c)
                            int filter_index = ((r * S + s) * C + c) + k * (R * S * C);
                            
                            sum += padded_input[padded_input_index] * filters[filter_index];
                        }
                    }
                }
                // 输出的下标计算：输出数据存储顺序为 (outX, outY, K)
                int output_index = (i * outY + j) * K + k;
                sum += neuron_state[output_index]; // 累积上一个时间步的膜电位
                if(sum>=V_th){
                    output[output_index] = 1;
                    neuron_state[output_index] = 0;
                } else {
                    output[output_index] = 0;
                    neuron_state[output_index] = sum;
                }
            }
        }
    }



    delete[] padded_input;
}

void pool2x2(int* on_chip_sram, int* output_regs, int Y_, int channels) {

    for(int i=0; i<channels; i++){  // 遍历每个通道
        // std::cout<<"channels id : "<<i<<std::endl;
        for(int j=0; j<Y_/2; j++){
            int addr1 = i*Y_*2 + j*2;
            int addr2 = i*Y_/2 + j;
            int sum = on_chip_sram[addr1] + on_chip_sram[addr1+1] + on_chip_sram[addr1+Y_] + on_chip_sram[addr1+Y_+1];
            if(sum>=1){
                output_regs[addr2] = 1;
            } else {
                output_regs[addr2] = 0;
            }
            // std::cout<<"addr1 : "<<addr1<<std::endl;
            // std::cout<<"addr2 : "<<addr2<<std::endl;
        }
    }
    
}

void conv_compute_dataflow(int R, int S, int C, int K, int P, int strides,
                  int X, int Y,
                  int* input, int* filters, int* output,
                  int* neuron_state, int V_th) {
    int padded_X = X + 2 * P;
    int padded_Y = Y + 2 * P;
    int outX = (padded_X - R) / strides + 1;
    int outY = (padded_Y - S) / strides + 1;

    // allocate padded input in (C, Xp, Yp) order
    int total_pad = C * padded_X * padded_Y;
    int* padded_input = new int[total_pad];
    // zero padding
    std::memset(padded_input, 0, sizeof(int) * total_pad);

    // copy input into padded buffer
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < padded_X; ++i) {
            for (int j = 0; j < padded_Y; ++j) {
                int pad_idx = c * (padded_X * padded_Y) + i * padded_Y + j;
                if (i >= P && i < P + X && j >= P && j < P + Y) {
                    int in_i = i - P;
                    int in_j = j - P;
                    int in_idx = c * (X * Y) + in_i * Y + in_j;
                    padded_input[pad_idx] = input[in_idx];
                } else {
                    padded_input[pad_idx] = 0;
                }
            }
        }
    }

    // perform convolution + integrate-and-fire
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < outX; ++i) {
            for (int j = 0; j < outY; ++j) {
                int sum = 0;
                for (int c = 0; c < C; ++c) {
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            int in_row = i * strides + r;
                            int in_col = j * strides + s;
                            int pad_idx = c * (padded_X * padded_Y) + in_row * padded_Y + in_col;
                            int filt_idx = k * (C * R * S) + c * (R * S) + r * S + s;
                            sum += padded_input[pad_idx] * filters[filt_idx];
                        }
                    }
                }
                int out_idx = k * (outX * outY) + i * outY + j;
                sum += neuron_state[out_idx];
                if (sum >= V_th) {
                    output[out_idx] = 1;
                    neuron_state[out_idx] = 0;
                } else {
                    output[out_idx] = 0;
                    neuron_state[out_idx] = sum;
                }
            }
        }
    }

    delete[] padded_input;
}

void conv_and_pooling_compute_dataflow(int R, int S, int C, int K, int P, int strides,
                  int X, int Y,
                  int* input, int* filters, int* output,
                  int* neuron_state, int V_th) {
    int padded_X = X + 2 * P;
    int padded_Y = Y + 2 * P;
    int convX = (padded_X - R) / strides + 1;
    int convY = (padded_Y - S) / strides + 1;

    // allocate padded input in (C, Xp, Yp) order
    int total_pad = C * padded_X * padded_Y;
    int* padded_input = new int[total_pad];
    std::memset(padded_input, 0, sizeof(int) * total_pad);

    // copy input into padded buffer
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < padded_X; ++i) {
            for (int j = 0; j < padded_Y; ++j) {
                int pad_idx = c * (padded_X * padded_Y) + i * padded_Y + j;
                if (i >= P && i < P + X && j >= P && j < P + Y) {
                    int in_i = i - P;
                    int in_j = j - P;
                    int in_idx = c * (X * Y) + in_i * Y + in_j;
                    padded_input[pad_idx] = input[in_idx];
                }
            }
        }
    }

    // temporary conv output (spikes) in (K, convX, convY)
    int conv_map_size = K * convX * convY;
    int* conv_out = new int[conv_map_size];

    // convolution + integrate-and-fire
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < convX; ++i) {
            for (int j = 0; j < convY; ++j) {
                int sum = 0;
                for (int c = 0; c < C; ++c) {
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            int in_row = i * strides + r;
                            int in_col = j * strides + s;
                            int pad_idx = c * (padded_X * padded_Y) + in_row * padded_Y + in_col;
                            int filt_idx = k * (C * R * S) + c * (R * S) + r * S + s;
                            sum += padded_input[pad_idx] * filters[filt_idx];
                        }
                    }
                }
                int idx = k * (convX * convY) + i * convY + j;
                sum += neuron_state[idx];
                if (sum >= V_th) {
                    conv_out[idx] = 1;
                    neuron_state[idx] = 0;
                } else {
                    conv_out[idx] = 0;
                    neuron_state[idx] = sum;
                }
            }
        }
    }

    // 2x2 max pooling on conv_out -> output
    int pooledX = convX / 2;
    int pooledY = convY / 2;
    for (int k = 0; k < K; ++k) {
        for (int i = 0; i < pooledX; ++i) {
            for (int j = 0; j < pooledY; ++j) {
                int base = k * (convX * convY);
                int maxv = 0;
                // pool window 2x2
                for (int dx = 0; dx < 2; ++dx) {
                    for (int dy = 0; dy < 2; ++dy) {
                        int xi = i * 2 + dx;
                        int yj = j * 2 + dy;
                        int v = conv_out[base + xi * convY + yj];
                        if (v > maxv) maxv = v;
                    }
                }
                int out_idx = k * (pooledX * pooledY) + i * pooledY + j;
                output[out_idx] = maxv;
            }
        }
    }

    delete[] conv_out;
    delete[] padded_input;
}

