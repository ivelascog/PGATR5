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

#define REDUCTION_THREADS 32
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

__global__ void scanReduce(float* scan, int start, int offset, int threads)
{
  int threadID = threadIdx.x + blockDim.x * blockIdx.x;
  if (threadID >= threads)
    return;
  int index = start + threadID * offset * 2;

  scan[index + offset] += scan[index];
}

__global__ void scanReverse(float* scan, int start, int offset, int threads)
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
  checkCudaErrors(cudaFree(d_histogram));
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

  float* d_scan;
  checkCudaErrors(cudaMalloc(&d_scan, numCols*numRows * sizeof(float)));
  checkCudaErrors(cudaMemcpy(d_scan, d_logLuminance, numCols*numRows * sizeof(float), cudaMemcpyDeviceToDevice));
  unsigned int len = numRows * numCols;
  numThreads = len / 2;
  unsigned int d = 0;
  unsigned int start, offset = 1;
  //Error después de ejecutar este bloque
  do {
    start = offset - 1;
    numBlocks = (numThreads - 1) / REDUCTION_THREADS + 1;
    scanReduce << < numBlocks, REDUCTION_THREADS >> > (d_scan, start, offset, numThreads);
    d++;
    numThreads >>= 1;
  } while ((offset = (1 << d)) < len);
  checkCudaErrors(cudaMemset(&d_scan[len-1], 0.0f, 1));
  numThreads = 1;
  while (d > 0)
  {
    d--;
    offset = (1 << d);
    start = offset - 1;
    numBlocks = (numThreads - 1) / REDUCTION_THREADS + 1;
    scanReverse << < numBlocks, REDUCTION_THREADS >> > (d_scan, start, offset, numThreads);
    numThreads >>= 1;
  } 
  //Resultado en scan

  checkCudaErrors(cudaFree(d_scan));
  free(h_histogram);
}
