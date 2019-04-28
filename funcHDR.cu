#include <device_functions.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <iomanip>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

#define REDUCTION_THREADS 64
#define DEBUG 1

__global__ void parallelMinMax(float* min, float* max, int len, int threads)
{
  int threadID = threadIdx.x + blockDim.x * blockIdx.x;

  float tempMin = min[threadID + threads];
  min[threadID] = tempMin <= min[threadID] ? tempMin : min[threadID];

  float tempMax = max[threadID + threads];
  max[threadID] = tempMax >= max[threadID] ? tempMax : max[threadID];

}

__global__ void parallelMinMaxInit(const float* input, float* min, float* max, int len, int threads)
{
  int threadID = threadIdx.x + blockDim.x * blockIdx.x;

  float tempMin = (threadID + threads) < len ? input[threadID + threads] : input[threadID];
  min[threadID] = tempMin <= input[threadID] ? tempMin : input[threadID];

  float tempMax = tempMin;
  max[threadID] = tempMax >= input[threadID] ? tempMax : input[threadID];
}

__global__ void initSharedMinMax(const float* input, float* vmin, float* vmax,int len)
{
	__shared__ float sharedMax[REDUCTION_THREADS * 2];
	__shared__ float sharedMin[REDUCTION_THREADS * 2];

	int ThreadID = threadIdx.x + blockDim.x * 2 * blockIdx.x;
	// Rellamos los valores de los threads
	if (ThreadID > len)
	{
		sharedMax[threadIdx.x] = input[0] ;
		sharedMin[threadIdx.x] = input[0] ;

	} else
	{
		sharedMax[threadIdx.x] = input[ThreadID];
		sharedMin[threadIdx.x] = sharedMax[threadIdx.x];

		// Rellenamos los valores desplazados.
		if (ThreadID + blockDim.x > len)
		{
			sharedMax[threadIdx.x + blockDim.x] = input[0];
			sharedMin[threadIdx.x + blockDim.x] = input[0];
		}
		else
		{
			sharedMax[threadIdx.x + blockDim.x] = input[ThreadID + blockDim.x];
			sharedMin[threadIdx.x + blockDim.x] = input[ThreadID + blockDim.x];
		}
	}

	__syncthreads;

	for (unsigned int desp = blockDim.x; desp > 0; desp /= 2)
	{
		if (threadIdx.x < desp) {
			sharedMax[threadIdx.x] = max(sharedMax[threadIdx.x], sharedMax[threadIdx.x + desp]);
			sharedMin[threadIdx.x] = min(sharedMin[threadIdx.x], sharedMin[threadIdx.x + desp]);
		}
		__syncthreads();
	}

	vmin[blockIdx.x] = sharedMin[0];
	vmax[blockIdx.x] = sharedMax[0];
}

__global__ void sharedMinMax(float* vmin, float* vmax, int len)
{
	__shared__ float sharedMax[REDUCTION_THREADS * 2];
	__shared__ float sharedMin[REDUCTION_THREADS * 2];

	int ThreadID = threadIdx.x + blockDim.x * 2 * blockIdx.x;
	// Rellamos los valores de los threads
	if (ThreadID > len)
	{
		sharedMax[threadIdx.x] = vmax[0];
		sharedMin[threadIdx.x] = vmin[0];

	}
	else
	{
		sharedMax[threadIdx.x] = vmax[ThreadID];
		sharedMin[threadIdx.x] = vmin[ThreadID];

		// Rellenamos los valores desplazados.
		if (ThreadID + blockDim.x > len)
		{
			sharedMax[threadIdx.x + blockDim.x] = vmax[0];
			sharedMin[threadIdx.x + blockDim.x] = vmin[0];
		}
		else
		{
			sharedMax[threadIdx.x + blockDim.x] = vmax[ThreadID + blockDim.x];
			sharedMin[threadIdx.x + blockDim.x] = vmin[ThreadID + blockDim.x];
		}
	}

	__syncthreads;

	for (unsigned int desp = blockDim.x; desp > 0; desp /= 2)
	{
		if (threadIdx.x < desp) {
			sharedMax[threadIdx.x] = max(sharedMax[threadIdx.x], sharedMax[threadIdx.x + desp]);
			sharedMin[threadIdx.x] = min(sharedMin[threadIdx.x], sharedMin[threadIdx.x + desp]);
		}
		__syncthreads();
	}
	vmin[blockIdx.x] = sharedMin[0];
	vmax[blockIdx.x] = sharedMax[0];
}

__global__ void histogram(const float* input, const size_t numCols, const size_t numRows, const float min, const float range, const size_t numBins, unsigned int* histogram)
{
  int threadID = threadIdx.x + blockDim.x * blockIdx.x;
  if (threadID > numRows)
    return;
  unsigned int maxID = threadID * numCols + numCols;
  for(unsigned int i = threadID * numCols; i < maxID; i++)
  {
    int bin = (input[i] - min) / range * numBins;
    //Último bin no inclusivo al quedar 1*numBins, reajustamos con un min(bin, numBins-1)
    bin = bin >= numBins ? numBins - 1 : bin;
    atomicAdd(&(histogram[bin]), 1);
  }
}

__global__ void scanReduce(unsigned int* scan, int start, int offset, int threads)
{
  int threadID = threadIdx.x + blockDim.x * blockIdx.x;
  if (threadID >= threads)
    return;
  int index = start + threadID * offset * 2;

  scan[index + offset] += scan[index];
}

__global__ void scanReverse(unsigned int* scan, int start, int offset, int threads)
{
  int threadID = threadIdx.x + blockDim.x * blockIdx.x;
  if (threadID >= threads)
    return;
  int index = start + threadID * offset * 2;

  scan[index + offset] += scan[index];

  if(threadID == 0)
  {
    scan[index] = 0;
  }
}

void getMaxMin(const float* input, int len, float&min, float&max, float& range)
{
	float* d_tempMin;
	float* d_tempMax;
	int numBlocks = (len + 1) / (REDUCTION_THREADS * 2) + 1;
	cudaMalloc(&d_tempMin, numBlocks * sizeof(float));
	cudaMalloc(&d_tempMax, numBlocks * sizeof(float));
	initSharedMinMax<< <numBlocks, REDUCTION_THREADS >> > (input, d_tempMin,d_tempMax, len);
	cudaDeviceSynchronize();

	while (numBlocks > 1) {
		int numBlocksTemp = numBlocks;
		numBlocks = (numBlocks + 1) / (REDUCTION_THREADS * 2) + 1;
		sharedMinMax<< <numBlocks, REDUCTION_THREADS >> > (d_tempMin, d_tempMax, numBlocksTemp);
		cudaDeviceSynchronize();
	}

	cudaMemcpy(&min, d_tempMin, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&max, d_tempMax, sizeof(float), cudaMemcpyDeviceToHost);
	range = max - min;
	cudaFree(d_tempMin);
	cudaFree(d_tempMax);

	std::cout << min << " " << max << std::endl;
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

  unsigned int numThreads = (numRows * numCols) / 2;
  if (numThreads % 2 != 0) numThreads++;
  unsigned int numBlocks;// = (numThreads - 1) / REDUCTION_THREADS + 1;
  float* d_minArray, *d_maxArray;
  checkCudaErrors(cudaMalloc(&d_minArray, numThreads * sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_maxArray, numThreads * sizeof(float)));
  numBlocks = (numThreads - 1) / REDUCTION_THREADS + 1;
  parallelMinMaxInit << < numBlocks, REDUCTION_THREADS >> > (d_logLuminance, d_minArray, d_maxArray, numRows * numCols, numThreads);
  while (numThreads > 1) {
    numThreads = std::ceil((float)numThreads / 2.0f);
    numBlocks = (numThreads - 1) / REDUCTION_THREADS + 1;
    parallelMinMax << < numBlocks, REDUCTION_THREADS >> > (d_minArray, d_maxArray, numRows * numCols, numThreads);
    cudaDeviceSynchronize();
  }
  checkCudaErrors(cudaMemcpy(&min_logLum, d_minArray, sizeof(float), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&max_logLum, d_maxArray, sizeof(float), cudaMemcpyDeviceToHost));
  float range = max_logLum - min_logLum;

  float min, max,range2;
  getMaxMin(d_logLuminance, numRows * numCols, min, max, range2);
#if DEBUG == 1
  std::cout << "Min: " << min_logLum << " | Max: " << max_logLum << std::endl;
  std::cout << "Rango a representar: " << range << std::endl;
#endif
  checkCudaErrors(cudaFree(d_minArray));
  checkCudaErrors(cudaFree(d_maxArray));

  numThreads = numRows;
  numBlocks = (numThreads - 1) / REDUCTION_THREADS + 1;
  unsigned int* d_histogram;
  checkCudaErrors(cudaMalloc(&d_histogram, numBins * sizeof(unsigned int)));
  histogram <<< dim3{ numBlocks, 1, 1 }, dim3{ REDUCTION_THREADS, 1, 1 } >>> (d_logLuminance, numCols, numRows, min_logLum, range, numBins, d_histogram);
  cudaDeviceSynchronize();
  unsigned int* h_histogram = (unsigned int*) malloc(numBins * sizeof(unsigned int));
  checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, numBins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

#if DEBUG == 1
  std::cout << "Histograma: ";

  unsigned int total = 0;
  for(int i = 0 ; i < numBins; i++)
  {
    if (i % 25 == 0)
      std::cout << std::endl;
    std::cout << h_histogram[i] << " ";
    total += h_histogram[i];
  }
  std::cout << std::endl;
  std::cout << "Total: " << total << std::endl;
  std::cout << "Total num*col: " << numCols*numRows << std::endl;
  std::cout << "Numbins: " << numBins << std::endl;
#endif 
  unsigned int len = numBins;
  numThreads = len / 2;
  unsigned int d = 0;
  unsigned int start, offset = 1;
  //Error después de ejecutar este bloque
  do {
    start = offset - 1;
    numBlocks = (numThreads - 1) / REDUCTION_THREADS + 1;
    scanReduce << < numBlocks, REDUCTION_THREADS >> > (d_histogram, start, offset, numThreads);
    d++;
    numThreads >>= 1;
	cudaDeviceSynchronize();
  } while ((offset = (1 << d)) < len); 
  checkCudaErrors(cudaMemset(&d_histogram[len-1], 0.0f, sizeof(float)));
  numThreads = 1;
  while (d > 0)
  {
    d--;
    offset = (1 << d);
    start = offset - 1;
    numBlocks = (numThreads - 1) / REDUCTION_THREADS + 1;
    scanReverse << < numBlocks, REDUCTION_THREADS >> > (d_histogram, start, offset, numThreads);
    numThreads >>= 1;
	cudaDeviceSynchronize();
  } 
  //Resultado en scan

  checkCudaErrors(cudaFree(d_histogram));
  free(h_histogram);
}
