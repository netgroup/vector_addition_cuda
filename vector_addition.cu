#include <stdio.h>
#include <cuda_runtime.h>

// Size of array
#define N (128*1048576)
#define THREADS_PER_BLOCK 1024
#define SLOWDOWN_FACTOR 10000

/* when a cudaMalloc(size) is called, the total amount of GPU memory is: size *
 * CUDA_MALLOC_MEM_MUL.
 */
#define CUDA_MALLOC_MEM_MUL 4

#define gpuErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void inline gpuAssert(cudaError_t code, const char *file, int line)
{
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s (code=%d), file=%s, line=%d\n",
			cudaGetErrorString(code), code, file, line);
		exit(1);
	}
}

// Kernel
__global__ void add_vectors(double *a, double *b, double *c)
{
	int j;
	int id = blockDim.x * blockIdx.x + threadIdx.x;

	if(id < N) {
		// just for slowing down the whole operation
		for(j=0; j < SLOWDOWN_FACTOR; ++j)
			c[id] = a[id] + b[id];
	}
}

// Main program
int main()
{
	// Number of bytes to allocate for N doubles
	size_t bytes = N*sizeof(double);

	// Allocate memory for arrays A, B, and C on host
	double *A = (double*)malloc(bytes);
	double *B = (double*)malloc(bytes);
	double *C = (double*)malloc(bytes);

	// Allocate memory for arrays d_A, d_B, and d_C on device
	double *d_A, *d_B, *d_C;
	gpuErrorCheck(cudaMalloc(&d_A, CUDA_MALLOC_MEM_MUL * bytes));
	gpuErrorCheck(cudaMalloc(&d_B, CUDA_MALLOC_MEM_MUL * bytes));
	gpuErrorCheck(cudaMalloc(&d_C, CUDA_MALLOC_MEM_MUL * bytes));

	// Fill host arrays A and B
	for(int i=0; i<N; i++)
	{
		A[i] = 1.0;
		B[i] = 2.0;
	}

	// Copy data from host arrays A and B to device arrays d_A and d_B
	gpuErrorCheck(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
	gpuErrorCheck(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

	// Set execution configuration parameters
	//		thr_per_blk: number of CUDA threads per grid block
	//		blk_in_grid: number of blocks in grid
	int thr_per_blk = THREADS_PER_BLOCK;
	int blk_in_grid = ceil( float(N) / thr_per_blk );

	// Launch kernel
	add_vectors<<< blk_in_grid, thr_per_blk >>>(d_A, d_B, d_C);

	// Copy data from device array d_C to host array C
	gpuErrorCheck(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

	// Verify results
	double tolerance = 1.0e-14;
	for(int i=0; i<N; i++)
	{
		if( fabs(C[i] - 3.0) > tolerance)
		{ 
			printf("\nError: value of C[%d] = %f instead of 3.0\n\n", i, C[i]);
			exit(1);
		}
	}	

	// Free CPU memory
	free(A);
	free(B);
	free(C);

	// Free GPU memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Threads Per Block = %d\n", thr_per_blk);
	printf("Blocks In Grid    = %d\n", blk_in_grid);
	printf("---------------------------\n\n");

	return 0;
}
