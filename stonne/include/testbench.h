#ifndef _TESTBENCH_H
#define _TESTBENCH_H


void sequential_layer(unsigned int R, unsigned int S, unsigned int C, unsigned int K, unsigned int G,  unsigned int N, unsigned int X, unsigned int Y, unsigned int strides,
int* input, int* filters, int* outputs, int* neuron_state, int Vth, int Timestamp, bool pooling_enabled);

void matrixMultiply(int M, int K, int N, int* input, int* weight, int* output, int* neuron_state, int V_th, int num_tile);

// 连续计算多个tile的输出结果存储到一个输出中
void matrixMultiply_new(int M, int K, int N, int* input, int* weight, int* output, int* neuron_state, int V_th, int num_tile, int rows);

void pooling_compute(int X, int Y, int C, int R, int S, int stride, int* input, int* output);

void conv_compute(int R, int S, int C, int K, int P, int strides, int X, int Y, int* input, int* filters, int* output, int* neuron_state, int V_th);

void conv_and_pooling_compute(int R, int S, int C, int K, int P, int strides, int X, int Y, int* input, int* filters, int* output, int* neuron_state, int V_th);

void pool2x2(int* on_chip_sram, int* output_regs, int Y_, int channels);

void conv_compute_dataflow(int R, int S, int C, int K, int P, int strides, int X, int Y, int* input, int* filters, int* output, int* neuron_state, int V_th);

void conv_and_pooling_compute_dataflow(int R, int S, int C, int K, int P, int strides, int X, int Y, int* input, int* filters, int* output, int* neuron_state, int V_th);

// void cpu_gemm(float* MK_dense_matrix, float* KN_dense_matrix, float* output, unsigned int M, unsigned int N, unsigned int K);
// void run_simple_tests();
// void run_stonne_architecture_tests(layerTest layer, unsigned int num_ms);
// void hand_tests();

#endif
