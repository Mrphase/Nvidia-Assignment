
/*
* Nvidia_assignment01
* Author: л���
*/

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

//zehui
#include <device_launch_parameters.h>

/*
* helper function
* �������ݣ���ӡ�������
*/
template <typename T>
T* random_list(T size);

template <typename T>
void print_list(T* list, int size);

template <typename T>
void check(T* input, T* output, int dimx, int dimy);
/*
* helper function end
*/

//ԭʼKernel
__global__ void kernel_A(float* g_data, int dimx, int dimy)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * dimx + ix;

	float value = g_data[idx];

	if (ix % 2)
	{
		value += sqrtf(logf(value) + 1.f);
	}
	else
	{
		value += sqrtf(cosf(value) + 1.f);
	}

	g_data[idx] = value;
}

/*
Opt1 kernel__Divergence1 ʹ����Ŀ��������ı�blocksize
ͨ�� Nsight compute �����ɼ� SM �� Memory ռ�ö���60%���£�
�����ж���������ΪOccopuancy Bottleneck�� ����ix = blockIdx.x; 
�� int idx = iy * dimx + ix; ������blocksize(dim.x, dim.y)=(1,512) ʱ��
�����߳�����Ϊiy * 1* blockIdx.x �� ���Ը��� 
blocksize(dim.x, dim.y)=(512,1)��
ix = blockIdx.x* blockDim.x + threadIdx.x; �����ռ���ʡ�
  
*/
__global__ void kernel__Divergence1(float* g_data, int dimx, int dimy)
{
	int ix = blockIdx.x* blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * dimx + ix;

	float value = g_data[idx];

	value += (ix % 2? sqrtf(logf(value) + 1.f) : sqrtf(cosf(value) + 1.f));

	g_data[idx] = value;
}
/*
*Opt2 kernel__Divergence2: 
ʹ������kernel�ֱ���� sqrtf(logf(value) + 1.f) 
�� sqrtf(cosf(value) + 1.f);
�����˺����У�ÿ���ƶ�����Ϊ2��
�޸Ĳ��Ժ���, ���У�Grid size ��СΪԭ����1/2�� �Ⱥ�������kernel��
*/
__global__ void kernel__Divergence2_log(float* g_data, int dimx, int dimy)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * dimx + (2 * ix+1);

	float value = g_data[idx];
	
	value +=  sqrtf(logf(value) + 1.f);

	g_data[idx] = value;
}
__global__ void kernel__Divergence2_cos(float* g_data, int dimx, int dimy)
{
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * dimx + (2 * ix );

	float value = g_data[idx];
	
	value += sqrtf(cosf(value) + 1.f);

	g_data[idx] = value;
}

/*
*Opt3 kernel__Table: 
* �����Դ��Leetcode�Ĵ����
��������/�����������ɿ�ʱ����Ԥ�ȼ������������ڰ���������ѯ�����ټ�������
�۲��������ݵ����ɣ� �����������ݷ�ΧΪ 10-265 ������ �����ֻ�� 2*255 �֣�
���ǵ�����������Ϊ2 x 1024 x 2 x 1024�� ������ʹ��������
Lookup Table is my favorite method in Leetcode contest!
*/

// ���
// ���� table �ǳ���Ϊ 510 ��һά����, ��Ϊ�����֣�
//sqrtf(cosf(value) + 1.f); �ļ����������� [0-254], 
//sqrtf(logf(value) + 1.f); �ļ����������� [255-590]��
__global__ void kernel__GeneTable(float* table, int max_lut_val, int niterations) {
	int ix = threadIdx.x;

	if (blockIdx.x == 0)
	{
		float value = ix + 10;
			value += sqrtf(logf(value) + 1.f);
		table[ix + max_lut_val] = value;
	}
	else
	{
		float value = ix + 10;
			value += sqrtf(cosf(value) + 1.f);
		table[ix] = value;
	}
}
//���
// ÿ���̼߳����в��ҽ���� ����Timing_experiment�����������޸Ľ��10�Σ�
// �������ɵı��Ǿ�̬�ģ��ڵ���timing_experiment����֮�����ַ�������������в��졣

__global__ void kernel__Table(float* g_data, float* table, int max_lut_val, int dimx, int dimy) {
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * dimx + ix;
	if ((ix < dimx) && (iy < dimy)) {
		int tableID = (int)roundf(g_data[idx] - 10);
		if (tableID < max_lut_val) {
			tableID = (ix % 2) ? (tableID + max_lut_val) : tableID;
			g_data[idx] = table[tableID];
		}
	}
}

__global__ void kernel__Table_log(float* g_data, float* table, int max_lut_val, int dimx, int dimy)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * dimx + (2 * ix + 1);

	int tableID = (int)roundf(g_data[idx] - 10)+ max_lut_val;

	g_data[idx] = table[tableID];
}
__global__ void kernel__Table_cos(float* g_data, float* table, int max_lut_val, int dimx, int dimy)
{
	int ix = blockIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int idx = iy * dimx + (2 * ix );

	int tableID = (int)roundf(g_data[idx] - 10) ;

	g_data[idx] = table[tableID];
}

/*
*Opt4 kernel__device_copy_vector2: �������ڴ���ʣ�Vectorized Memory Access��
*�ο���
https://stackoverflow.com/questions/26676806/efficiency-of-cuda-vector-types-float2-float3-float4
* https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
ʹ���������غʹ洢���԰�����ߴ��������ʣ�ͬʱ������ִ��ָ���������
���´�����ÿ���̴߳��� KERNEL_X_DIM2 * KERNEL_Y_DIM2 ��4�����ݣ� Grid, block size ��Ӧ��С1/2
*/

#define KERNEL_X_DIM2 2
#define KERNEL_Y_DIM2 2
__global__ void kernel__device_copy_vector2(float* g_data, const int dimx, const int dimy) {
	const int ix = blockIdx.x * blockDim.x + threadIdx.x;
	const int iy = blockIdx.y * blockDim.y + threadIdx.y;
	const int idx = KERNEL_Y_DIM2 * iy * dimx + ix;
	float2* f2_data = reinterpret_cast<float2*>(g_data);
	float2 values0 = f2_data[idx + 0 * dimx];
	float2 values1 = f2_data[idx + 1 * dimx];
	//compute
	values0.x += sqrtf(cosf(values0.x) + 1.f);
	values0.y += sqrtf(logf(values0.y) + 1.f);
	values1.x += sqrtf(cosf(values1.x) + 1.f);
	values1.y += sqrtf(logf(values1.y) + 1.f);
	//restore anwser
	f2_data[idx + 0 * dimx] = values0;
	f2_data[idx + 1 * dimx] = values1;
}

/*
* Timing_experiment����kernel10�Σ�����ζ��d_data �Ľ�����޸��� 10�Ρ�
*/
float timing_experiment(void (*kernel)(float*, int, int), float* d_data, float* h_data,int nbytes, // input h_data for error checking
	int dimx, int dimy, int nreps, int blockx, int blocky)
{
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block(blockx, blocky);
	dim3 grid(dimx / block.x, dimy / block.y);

	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++)	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel << <grid, block >> > (d_data, dimx, dimy);
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms /= nreps;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/*
	* add by zehui, �鿴��� 
	*/
	float* output = (float*)malloc(nbytes);
	// Copy result from device to host
	checkCudaErrors(cudaMemcpy(output, d_data, nbytes, cudaMemcpyDeviceToHost));
	print_list(output,4*1024);

	free(output);
	//check(h_data, output, dimx, dimy);


	/*
	* add by zehui end
	*/

	return elapsed_time_ms;
}

// timing_experiment for 2 kernel testing
float timing_experiment_Divergence2(float* d_data, float* h_data, int nbytes, 
	int dimx, int dimy, int nreps, int blockx, int blocky);

// timing_experiment �����ֱ��壬�����ڷ�������kernel�Լ����
//�������룺 float* table�� int max_lut_val
float timing_experiment_Table(float* d_data, float* table, int max_lut_val, int nbytes,
	int dimx, int dimy, int nreps, int blockx, int blocky);
float timing_experiment_Table_Divergence2(float* d_data, float* table, int max_lut_val, int nbytes, // input h_data for error checking
	int dimx, int dimy, int nreps, int blockx, int blocky);



int main()
{

	int dimx = 2 * 1024;
	int dimy = 2 * 1024;

	int nreps = 10;

	int nbytes = dimx * dimy * sizeof(float);

	//zehui
	float* outpot_0 = 0, * outpot_1 = 0;
	outpot_0 = (float*)malloc(nbytes);
	outpot_1 = (float*)malloc(nbytes);

	float* d_data = 0, * h_data = 0;
	cudaMalloc((void**)&d_data, nbytes);
	if (0 == d_data)
	{
		printf("couldn't allocate GPU memory\n");
		return -1;
	}
	printf("allocated %.2f MB on GPU\n", nbytes / (1024.f * 1024.f));
	h_data = (float*)malloc(nbytes);
	if (0 == h_data)
	{
		printf("couldn't allocate CPU memory\n");
		return -2;
	}
	printf("allocated %.2f MB on CPU\n", nbytes / (1024.f * 1024.f));
	for (int i = 0; i < dimx * dimy; i++)
		h_data[i] = 10.f + rand() % 256;

	std::cout << "\ninput: ";
	print_list(h_data, 4 * 1024);//zehui

	std::cout << "\n===========================kernel_A (origional)============================\n";
	checkCudaErrors(cudaMemcpy(d_data, h_data, nbytes, cudaMemcpyHostToDevice));

	float elapsed_time_ms = 0.0f;

	elapsed_time_ms = timing_experiment(kernel_A, d_data, h_data , nbytes,//zehui
		dimx, dimy, nreps, 1, 512);
	
	printf("A:  %8.2f ms\n", elapsed_time_ms);

	printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

	//zehui
	checkCudaErrors(cudaMemcpy(outpot_0, d_data, nbytes, cudaMemcpyDeviceToHost));


	std::cout << "\n===========================kernel__Divergence1_Occopuancy============================\n";
	checkCudaErrors(cudaMemcpy(d_data, h_data, nbytes, cudaMemcpyHostToDevice));

	elapsed_time_ms = timing_experiment(kernel__Divergence1, d_data, h_data, nbytes,//zehui
		dimx, dimy, nreps, 512, 1);
	printf("kernel__Divergence1_Occopuancy:  %8.2f ms\n", elapsed_time_ms);

	printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

	checkCudaErrors(cudaMemcpy(outpot_1, d_data, nbytes, cudaMemcpyDeviceToHost));


	check(outpot_0, outpot_1, dimx, dimy);

	//run kernel__Divergence2
	std::cout << "\n===========================kernel__Divergence2============================\n";
	checkCudaErrors(cudaMemcpy(d_data, h_data, nbytes, cudaMemcpyHostToDevice));

	elapsed_time_ms = timing_experiment_Divergence2( d_data, h_data, nbytes,//zehui
		dimx, dimy, nreps, 512, 1);
	printf("kernel__Divergence2:  %8.2f ms\n", elapsed_time_ms);

	printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

	checkCudaErrors(cudaMemcpy(outpot_1, d_data, nbytes, cudaMemcpyDeviceToHost));


	check(outpot_0, outpot_1, dimx, dimy);


	//run kernel__Table
	std::cout << "\n===========================kernel__Table============================\n";
	std::cout << "===h_data[i] = 10.f + rand() % 256;, range from 10-265 unique values===========\n";
	std::cout << "===Since timing_experiment() function modify restult 10 times,\nBut our table is static, this mehtod may not output exact same value after call timing_experiment()===\n";

	checkCudaErrors(cudaMemcpy(d_data, h_data, nbytes, cudaMemcpyHostToDevice));
	

	//gene table
	float* d_table = 0, * h_table =0;
	const int max_lut_val = 255;
	h_table = (float*)malloc(2 * max_lut_val * sizeof(float));
	checkCudaErrors(cudaMalloc((void**)&d_table, 2 * max_lut_val * sizeof(float)));

	kernel__GeneTable << <2, max_lut_val >> > (d_table, max_lut_val,  nreps);
	checkCudaErrors(cudaMemcpy(h_table, d_table, 2 * max_lut_val * sizeof(float), cudaMemcpyDeviceToHost));
	//genetable finish

	//// check table
	//std::cout << "---------\n";
	//for (int i = 0; i < 2 * max_lut_val; i++)
	//	std::cout << ((i<255)? i:(i-255)) << "--" << h_table[i] << "    ";

	checkCudaErrors(cudaMemcpy(d_data, h_data, 265 * sizeof(float), cudaMemcpyHostToDevice));
	timing_experiment_Table(d_data, d_table, 255, nbytes, dimx, dimy, nreps, 512, 1);
	//timing_experiment_Table_Divergence2(d_data, d_table, 255, nbytes, dimx, dimy, nreps, 1, 512);
	printf("kernel__Table:  %8.2f ms\n", elapsed_time_ms);

	printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
	checkCudaErrors(cudaMemcpy(outpot_1, d_data, nbytes, cudaMemcpyDeviceToHost));
	
	//Since timing_experiment() function modify restult 10 times,
	//But our table is static, this mehtod may not output same value after call timing_experiment()

	std::cout << "\n===========================kernel_kernel__device_copy_vector2 Vectorized Memory Access============================\n";
	//run optimized kernel learned from here:https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/
	checkCudaErrors(cudaMemcpy(d_data, h_data, nbytes, cudaMemcpyHostToDevice));

	elapsed_time_ms = timing_experiment(kernel__device_copy_vector2, d_data,h_data,nbytes ,dimx / KERNEL_X_DIM2, dimy / KERNEL_Y_DIM2, nreps, 256, 1);
	printf("kernel__device_copy_vector2:  %8.2f ms\n", elapsed_time_ms);

	printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));
	checkCudaErrors(cudaMemcpy(outpot_1, d_data, nbytes, cudaMemcpyDeviceToHost));
	check(outpot_0, outpot_1, dimx, dimy);






	if (d_data)
		cudaFree(d_data);
	if (h_data)
		free(h_data);

	cudaThreadExit();

	return 0;
}



template <typename T>
T* random_list(T size) {
	T* list = (T*)malloc(size * sizeof(T));
#pragma omp parallel for
	for (int i = 0; i < size; i++)
	{
		list[i] = rand() / (T)(RAND_MAX / 10);;
	}
	return list;
}
template <typename T>
void print_list(T* list, int size) {
	int size2 = (size > 10) ? 10 : size;
	std::cout <<"print_list:\n ";
	for (int i = 0; i < size2; i++)
	{
		std::cout << list[i] << " ";
	}
	std::cout << "...\n ";
}

/*
* Just use KernelA to check, i'm dumb

*/
template <typename T>
void check(T* input, T* output, int dimx, int dimy){
	//T* temp = 0;
	//int nbytes = dimx * dimy * sizeof(T);
	//temp = (float*)malloc(nbytes);
#pragma omp parallel for
	for (int i = 0; i < dimx * dimy; i++) {
		if (input[i] - output[i] > 0.01 ) {
			printf("%d %d\n", input[i], output[i]  );
			printf("%d incosist answer!\n",i);
			return;
		}
	}printf("res correct\n");
	
}



float timing_experiment_Divergence2( float* d_data, float* h_data, int nbytes, // input h_data for error checking
	int dimx, int dimy, int nreps, int blockx, int blocky)
{
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block(blockx, blocky);
	dim3 grid(dimx / block.x/2, dimy / block.y);// ZEHUI: use half of grid

	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++) {	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel__Divergence2_log << <grid, block >> > (d_data, dimx, dimy);
		kernel__Divergence2_cos << <grid, block >> > (d_data, dimx, dimy);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms /= nreps;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/*
	* add by zehui, check result
	*/
	float* output = (float*)malloc(nbytes);
	// Copy result from device to host
	checkCudaErrors(cudaMemcpy(output, d_data, nbytes, cudaMemcpyDeviceToHost));
	print_list(output, 4 * 1024);

	free(output);

	/*
	* add by zehui end
	*/

	return elapsed_time_ms;
}

// add  input�� float* table�� int max_lut_val
float timing_experiment_Table(float* d_data, float* table, int max_lut_val,int nbytes, // input h_data for error checking
	int dimx, int dimy, int nreps, int blockx, int blocky)
{
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block(blockx, blocky);
	dim3 grid(dimx / block.x , dimy / block.y);

	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++) {	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel__Table << <grid, block >> > (d_data, table, max_lut_val, dimx, dimy);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms /= nreps;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/*
	* add by zehui, check result
	*/
	float* output = (float*)malloc(nbytes);
	// Copy result from device to host
	checkCudaErrors(cudaMemcpy(output, d_data, nbytes, cudaMemcpyDeviceToHost));
	print_list(output, 4 * 1024);

	free(output);
	//check(h_data, output, dimx, dimy);


	/*
	* add by zehui end
	*/

	return elapsed_time_ms;
}

float timing_experiment_Table_Divergence2(float* d_data, float* table, int max_lut_val, int nbytes, // input h_data for error checking
	int dimx, int dimy, int nreps, int blockx, int blocky)
{
	float elapsed_time_ms = 0.0f;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 block(blockx, blocky);
	dim3 grid(dimx / block.x / 2, dimy / block.y);// ZEHUI: use half of grid

	cudaEventRecord(start, 0);
	for (int i = 0; i < nreps; i++) {	// do not change this loop, it's not part of the algorithm - it's just to average time over several kernel launches
		kernel__Table_log << <grid, block >> > (d_data, table, max_lut_val, dimx, dimy);
		kernel__Table_cos << <grid, block >> > (d_data, table, max_lut_val, dimx, dimy);
	}
	cudaEventRecord(stop, 0);
	cudaThreadSynchronize();
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms /= nreps;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/*
	* add by zehui, check result
	*/
	float* output = (float*)malloc(nbytes);
	// Copy result from device to host
	checkCudaErrors(cudaMemcpy(output, d_data, nbytes, cudaMemcpyDeviceToHost));
	print_list(output, 4 * 1024);

	free(output);
	//check(h_data, output, dimx, dimy);


	/*
	* add by zehui end
	*/

	return elapsed_time_ms;
}