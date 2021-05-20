//#include <unistd.h>



#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include"cuda.h"
#include "device_launch_parameters.h"
#include  "device_functions.h"
#include <vector>
using std::cout;
using std::generate;
using std::vector;
#include <chrono>
using std::cout;
using std::generate;
using std::vector;
using namespace std;
using namespace std::chrono;
#define Tile_size 16
#define TILE_SIZE 16

//Function To handle any errors occurred in the function calls



	int numARows; 		// number of rows in the matrix A
	int numAColumns; 	// number of columns in the matrix A
	int numBRows; 		// number of rows in the matrix B
	int numBColumns; 	// number of columns in the matrix B
	int numCRows;		// number of rows in the matrix C 
	int numCColumns;	// number of columns in the matrix C 

	__global__ void matrixMultiply(float* A, float* B, float* C,
		int numARows, int numAColumns,
		int numBRows, int numBColumns,
		int numCRows, int numCColumns) 
	{

		int Row = blockIdx.y * blockDim.y + threadIdx.y;
		int Col = blockIdx.x * blockDim.x + threadIdx.x;
		if (numAColumns != numBRows) return;
		if ((Row < numARows) && (Col < numBColumns)) {
			float Cvalue = 0;
			for (int k = 0; k < numAColumns; ++k)
				Cvalue += A[Row * numAColumns + k] * B[k * numBColumns + Col];
			C[Row * numCColumns + Col] = Cvalue;
		}

	}

	// Compute C = A * B
	//*************************************************************
	//Kernel for shared memory/ Tiled execution
	__global__ void matrixMultiplyShared(float* A, float* B, float* C,
		int numARows, int numAColumns,
		int numBRows, int numBColumns,
		int numCRows, int numCColumns)
	{
		__shared__ float sA[Tile_size][Tile_size];   // Tile size to store elements in shared memory
		__shared__ float sB[Tile_size][Tile_size];

		int Row = blockDim.y * blockIdx.y + threadIdx.y; //To generate ids of threads.
		int Col = blockDim.x * blockIdx.x + threadIdx.x;
		float Cvalue = 0.0;
		sA[threadIdx.y][threadIdx.x] = 0.0;
		sB[threadIdx.y][threadIdx.x] = 0.0;

		for (int k = 0; k < (((numAColumns - 1) / Tile_size) + 1); k++)
		{
			if ((Row < numARows) && (threadIdx.x + (k * Tile_size)) < numAColumns)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
			{
				sA[threadIdx.y][threadIdx.x] = A[(Row * numAColumns) + threadIdx.x + (k * Tile_size)];
			}
			else
			{
				sA[threadIdx.y][threadIdx.x] = 0.0; //printf("  SA ! %d, %d  ", Row, threadIdx.x + (k * Tile_size) );
			}
			if (Col < numBColumns && (threadIdx.y + k * Tile_size) < numBRows)//Copy Data to Tile from Matrix (Global Memory to Shared Memory)
			{
				sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + k * Tile_size) * numBColumns + Col];
			}
			else
			{
				sB[threadIdx.y][threadIdx.x] = 0.0; //printf("  SB ! %d, %d  ", Col, (threadIdx.y + k * Tile_size));
			}
			__syncthreads();

			for (int j = 0; j < Tile_size; ++j)//Multiplying Elements present in tile
			{
				Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
			}
		}
		if (Row < numCRows && Col < numCColumns)//Saving Final result into Matrix C
		{
			C[Row * numCColumns + Col] = Cvalue;
		}
	}
	//*************************************************************


	//*************************************************************

	void Print_Mat(int Row, int Col, float* Mat)//Function To print the Matrix
	{
		for (int i = 0; i < Row * Col; i++)
		{
			float temp = * (Mat + i);
			int temp2 = (int)temp;
			printf("%d  ", temp2);

			if (((i+1) % Col) == 0 && i>2)
			{
				printf("\n");
			}
		}
	}//Function close
	//*************************************************************
	//Normal CPU Matrix Multiplication
	void matMultiplyOnHost(float* A, float* B, float* C, int numARows,
		int numAColumns, int numBRows, int numBColumns,
		int numCRows, int numCColumns)
	{
		for (int i = 0; i < numARows; i++)
		{
			for (int j = 0; j < numBColumns; j++)
			{
				C[i * numCColumns + j] = 0.0;
				for (int k = 0; k < numBRows; k++)
				{
					C[i * numCColumns + j] += A[i * numAColumns + k] * B[k * numBColumns + j];
				}
			}
		}
		return;
	}
	void test();

	__global__ void gpu_matrix_mult(float* a, float* b, float* c, int m, int n, int k);
	__global__ void shared_matrix_mult(float* A, float* B, float* C, int m, int n, int k);

	//*************************************************************
	int main(int argc, char** argv) {
		cout << "\n===========================test============================\n";
		 test();

		cout << "\n===========================matrixMul============================\n";
		float* hostA; // The A matrix
		float* hostB; // The B matrix
		float* hostC; // The output C matrix
		float* hostComputedC;
		float* deviceA;
		float* deviceB;
		float* deviceC;


		// count the execution time
		float shared_gpu_time_ms, gpu_elapsed_time_ms, cpu_elapsed_time_ms;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);


		int rowDimA, colDimA, colDimB;
		//testGetOpt(argc, argv, rowDimA, colDimA, colDimB);

		rowDimA = 512; colDimA = 1024; colDimB = 10;
		numARows = 512; numAColumns = 1024; numBRows = 1024; numBColumns = 640;
		printf("Zehui Xie\n rowDimA: %d  colDimA: %d  colDimB: %d\n", numARows, numAColumns, numBColumns);



		// MxN = MxK * KxN
		int M = numARows; int K = numAColumns; int N = numBColumns;
		numCRows = M; numCColumns = N;

		hostA = (float*)malloc(sizeof(float) * numARows * numAColumns);
		hostB = (float*)malloc(sizeof(float) * numBRows * numBColumns);

		for (int i = 0; i < numARows * numAColumns; i++)//Matrix Initialization
		{
			hostA[i] = 1.0;
		}
		for (int i = 0; i < numBRows * numBColumns; i++)
		{
			hostB[i] = 1.0;
		}

		//printf("\nMatrix A Values:\n");
		//Print_Mat(numARows, numAColumns, hostA);//Function Call

		//printf("\n\nMatrix B Values:\n");
		//Print_Mat(numBRows, numBColumns, hostB);//Function Call



		// Setting numCRows and numCColumns
		numCRows = numARows;
		numCColumns = numBColumns;

		hostC = (float*)malloc(sizeof(float) * numCRows * numCColumns);
		hostComputedC = (float*)malloc(sizeof(float) * numCRows * numCColumns);

		// Allocating GPU memory
		(cudaMalloc((void**)&deviceA, sizeof(float) * numARows * numAColumns));
		(cudaMalloc((void**)&deviceB, sizeof(float) * numBRows * numBColumns));
		(cudaMalloc((void**)&deviceC, sizeof(float) * numCRows * numCColumns));

		// Copy memory to the GPU
		(cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice));
		(cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice));

		// Initialize the grid and block dimensions

		dim3 dimGrid((numCColumns / Tile_size) + 1, (numCRows / Tile_size) + 1, 1);//Number of Blocks required
		dim3 dimBlock(Tile_size, Tile_size, 1);//Number of threads in each block



		// start to count execution time of GPU without using Shared Memory version
		cudaEventRecord(start, 0);
		//@@ Launch the GPU Kernel here
		for (int i = 0; i < 10; i++)
		{
			matrixMultiply << <dimGrid, dimBlock >> > (deviceA, deviceB, deviceC,
				numARows, numAColumns, 
				numBRows, numBColumns, 
				numCRows, numCColumns);
		}
		cudaDeviceSynchronize();//To synchronize the device

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		// compute time elapse on GPU computing
		cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
		gpu_elapsed_time_ms = gpu_elapsed_time_ms / 10;
		printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU without shared memory: %f ms.\n\n", numARows, numAColumns, numBRows, numBColumns, gpu_elapsed_time_ms);

		cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call



		// Copy the results in GPU memory back to the CPU
		(cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost));

		//printf("\nMatrix C From Device\n");
		//Print_Mat(numCRows, numCColumns, hostC);//Function Call

		matMultiplyOnHost(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

		//printf("\nMatrix C From Host\n");
		//Print_Mat(numCRows, numCColumns, hostComputedC);//Function Call

		for (int i = 0; i < numCColumns * numCRows; i++)//Compare both the result matrices 1. MatrixMultiplyonHost 2. MatrixMultiplyonDevice
		{
			if (hostComputedC[i] != hostC[i])
			{
				printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
				return 0;
			}
			
		}
		printf("res correct!");

		double flopsPerMatrixMul = 2.0 * static_cast<double>(numARows) *
			static_cast<double>(numAColumns) *
			static_cast<double>(numBColumns);
		double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
			(gpu_elapsed_time_ms  / 1000.0f);
		printf(
			"\nPerformance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f \n" ,
			gigaFlops,
			gpu_elapsed_time_ms ,
			flopsPerMatrixMul);

		cout << "\n===========================matrixMul_SharedMemory============================\n";


		// start to count execution time of GPU without using Shared Memory version
		cudaEventRecord(start, 0);
		for (int i = 0; i < 10; i++)
		{
			matrixMultiplyShared << <dimGrid, dimBlock >> > (deviceA, deviceB, deviceC,
				numARows, numAColumns,
				numBRows, numBColumns,
				numCRows, numCColumns);
		}
		cudaDeviceSynchronize();//To synchronize the device

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		// compute time elapse on GPU computing
		cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
		gpu_elapsed_time_ms = gpu_elapsed_time_ms / 10;
		printf("Time elapsed on matrix multiplication of %dx%d . %dx%d on GPU with shared memory: %f ms.\n\n", numARows, numAColumns, numBRows, numBColumns, gpu_elapsed_time_ms);

		 err1 = cudaPeekAtLastError();//To capture last error in function call

		// Copy the results in GPU memory back to the CPU
		(cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost));

		matMultiplyOnHost(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

		for (int i = 0; i < numCColumns * numCRows; i++)//Compare both the result matrices 1. MatrixMultiplyonHost 2. MatrixMultiplyonDevice
		{
			if (hostComputedC[i] != hostC[i])
			{
				printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
				return 0;
			}

		}
		printf("res correct!");

		gigaFlops = (flopsPerMatrixMul * 1.0e-9f) /
			(gpu_elapsed_time_ms / 1000.0f);
		printf(
			"\nPerformance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f \n",
			gigaFlops,
			gpu_elapsed_time_ms,
			flopsPerMatrixMul);




		///////////end

		//exit(0);
		// Free the GPU memory
		(cudaFree(deviceA));
		(cudaFree(deviceB));
		(cudaFree(deviceC));
		//Free the Pointer Memory
		free(hostA);
		free(hostB);
		free(hostC);
		//free(hostComputedC);
		//exit(0);
		return 0;
	}



	void test() {
	
		float* hostA; // The A matrix
		float* hostB; // The B matrix
		float* hostC; // The output C matrix
		float* hostComputedC;
		float* deviceA;
		float* deviceB;
		float* deviceC;


		auto start = high_resolution_clock::now();
		auto stop = high_resolution_clock::now();
		auto duration = duration_cast<nanoseconds>(stop - start);


		int rowDimA, colDimA, colDimB;
		//testGetOpt(argc, argv, rowDimA, colDimA, colDimB);

		rowDimA = 10; colDimA = 20; colDimB = 15;
		numARows = 5; numAColumns = 10; numBRows = 10; numBColumns = 8;
		printf("Zehui Xie\n rowDimA: %d  colDimA: %d  colDimB: %d\n", numARows, numAColumns, numBColumns);



		// MxN = MxK * KxN
		int M = numARows; int K = numAColumns; int N = numBRows;
		numCRows = M; numCColumns = N;

		hostA = (float*)malloc(sizeof(float) * numARows * numAColumns);
		hostB = (float*)malloc(sizeof(float) * numBRows * numBColumns);

		for (int i = 0; i < numARows * numAColumns; i++)//Matrix Initialization
		{
			hostA[i] = 1.0;
		}
		for (int i = 0; i < numBRows * numBColumns; i++)
		{
			hostB[i] = 1.0;
		}

		printf("\nMatrix A Values:\n");
		Print_Mat(numARows, numAColumns, hostA);//Function Call

		printf("\n\nMatrix B Values:\n");
		Print_Mat(numBRows, numBColumns, hostB);//Function Call



		// Setting numCRows and numCColumns
		numCRows = numARows;
		numCColumns = numBColumns;

		hostC = (float*)malloc(sizeof(float) * numCRows * numCColumns);
		hostComputedC = (float*)malloc(sizeof(float) * numCRows * numCColumns);

		// Allocating GPU memory
		(cudaMalloc((void**)&deviceA, sizeof(float) * numARows * numAColumns));
		(cudaMalloc((void**)&deviceB, sizeof(float) * numBRows * numBColumns));
		(cudaMalloc((void**)&deviceC, sizeof(float) * numCRows * numCColumns));

		// Copy memory to the GPU
		(cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns, cudaMemcpyHostToDevice));
		(cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns, cudaMemcpyHostToDevice));

		// Initialize the grid and block dimensions

		dim3 dimGrid((numCColumns / Tile_size) + 1, (numCRows / Tile_size) + 1, 1);//Number of Blocks required
		dim3 dimBlock(Tile_size, Tile_size, 1);//Number of threads in each block

		//@@ Launch the GPU Kernel here
		matrixMultiplyShared << <dimGrid, dimBlock >> > (deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

		cudaError_t err1 = cudaPeekAtLastError();//To capture last error in function call

		cudaDeviceSynchronize();//To synchronize the device

		// Copy the results in GPU memory back to the CPU
		(cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns, cudaMemcpyDeviceToHost));

		printf("\nMatrix C From Device\n");
		Print_Mat(numCRows, numCColumns, hostC);//Function Call

		matMultiplyOnHost(hostA, hostB, hostComputedC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);

		printf("\nMatrix C From Host\n");
		Print_Mat(numCRows, numCColumns, hostComputedC);//Function Call

		for (int i = 0; i < numCColumns * numCRows; i++)//Compare both the result matrices 1. MatrixMultiplyonHost 2. MatrixMultiplyonDevice
		{
			if (hostComputedC[i] != hostC[i])
			{
				printf("Mismatch at Row = %d Col = %d hostComputed[] = %f --device[] %f\n", i / numCColumns, i % numCColumns, hostComputedC[i], hostC[i]);
				break;
			}
		}


		// Free the GPU memory
		(cudaFree(deviceA));
		(cudaFree(deviceB));
		(cudaFree(deviceC));
		//Free the Pointer Memory
		free(hostA);
		free(hostB);
		free(hostC);
		free(hostComputedC);

	}

