#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>

#define REDUCTION_THREADS 32
__global__ void paralelMin(const float* input, float* output, int len)
{
	int threadID = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ float sdata[REDUCTION_THREADS];
	if (threadID > len)
	{
		sdata[threadIdx.x] = 10000.0f;
	}
	else
	{
		sdata[threadIdx.x] = input[threadID];
	}

	__syncthreads();

	for (unsigned int desp = blockDim.x / 2; desp > 0; desp /= 2)
	{
		if (threadIdx.x < desp) {
			sdata[threadIdx.x] = min(sdata[threadIdx.x], sdata[threadIdx.x + desp]);
		}
		__syncthreads();
	}

	output[blockIdx.x] = sdata[0];
}

__global__ void paralelMax(const float* input, float* output, int len)
{
	int threadID = threadIdx.x + blockDim.x * blockIdx.x;
	__shared__ float sdata[REDUCTION_THREADS];
	if ( threadID > len)
	{
		sdata[threadIdx.x] = -100000.0f;
	}
	else
	{
		sdata[threadIdx.x] = input[threadID];
	}

	__syncthreads();

	for (unsigned int desp = blockDim.x / 2 ; desp > 0; desp /= 2)
	{
		if (threadIdx.x < desp) {
			sdata[threadIdx.x] = max(sdata[threadIdx.x], sdata[threadIdx.x + desp]);
		}
		__syncthreads();
	}

	output[blockIdx.x] = sdata[0];
}

void calculate_cdf(const float* const d_logLuminance,
	unsigned int* const d_cdf,
	float &min_logLum,
	float &max_logLum,
	const size_t numRows,
	const size_t numCols,
	const size_t numBins)
{
	/* TODO
	  1) Encontrar el valor máximo y mínimo de luminancia en min_logLum and max_logLum a partir del canal logLuminance
	  2) Obtener el rango a representar
	  3) Generar un histograma de todos los valores del canal logLuminance usando la formula
	  bin = (Lum [i] - lumMin) / lumRange * numBins
	  4) Realizar un exclusive scan en el histograma para obtener la distribución acumulada (cdf)
	  de los valores de luminancia. Se debe almacenar en el puntero c_cdf
	*/


	//max
	float* d_temp;
	int numBlocks = ((numRows * numCols) + 1) / REDUCTION_THREADS + 1;
	cudaMalloc(&d_temp, numBlocks * sizeof(float));
	paralelMax << <numBlocks, REDUCTION_THREADS >> > (d_logLuminance, d_temp, numRows * numCols);
	cudaDeviceSynchronize();

	float* d_temp2;
	cudaMalloc(&d_temp2, ((numBlocks + 1) / REDUCTION_THREADS + 1 ) * sizeof(float));
	while (numBlocks > 1) {
		int numBlocksTemp = numBlocks;
		numBlocks = (numBlocks + 1) / REDUCTION_THREADS + 1;
		paralelMax << <numBlocks, REDUCTION_THREADS >> > (d_temp, d_temp2, numBlocksTemp);
		cudaDeviceSynchronize();
		std::swap(d_temp, d_temp2);
	}
	cudaMemcpy(&max_logLum, d_temp,sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << max_logLum << std::endl;

	//min
	numBlocks = ((numRows * numCols) + 1) / REDUCTION_THREADS + 1;
	paralelMin << <numBlocks, REDUCTION_THREADS >> > (d_logLuminance, d_temp, numRows * numCols);
	cudaDeviceSynchronize();

	while (numBlocks >1)
	{
		int numBlocksTemp = numBlocks;
		numBlocks = (numBlocks + 1) / REDUCTION_THREADS + 1;
		paralelMin << <numBlocks, REDUCTION_THREADS >> > (d_temp, d_temp2, numBlocksTemp);
		cudaDeviceSynchronize();
		std::swap(d_temp, d_temp2);
	}

	cudaMemcpy(&min_logLum, d_temp, sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << min_logLum << std::endl;
}
