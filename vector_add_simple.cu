#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

using namespace std::chrono;

#define RUNS       5
#define CPU_REPEAT 5
#define THREADS    256
#define BLOCKS     512

__global__ void vector_add(float *A, float *B, float *C, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        C[i] = A[i] + B[i];
}

void cpuVectorAdd(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        float t = A[i] + B[i];
        t = t * 1.0001f / 1.0001f;
        C[i] = t;
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cores = prop.multiProcessorCount * 128;
    printf("GPU: %s | Cores (est): %d\n\n", prop.name, cores);

    cudaFree(0);

    int sizes[] = {1000,2000, 3000, 4000, 5000};
    int tests   = sizeof(sizes) / sizeof(int);

    FILE *f = fopen("vector_result.csv", "w");
    fprintf(f, "N,SEQ_MS,PAR_MS,SPEEDUP,EFFICIENCY,COST\n");

    for (int t = 0; t < tests; t++) {
        int N    = sizes[t];
        size_t sz = N * sizeof(float);

        float *h_A, *h_B, *h_C;
        cudaMallocHost(&h_A, sz); cudaMallocHost(&h_B, sz); cudaMallocHost(&h_C, sz);
        for (int i = 0; i < N; i++) { h_A[i] = rand()%100; h_B[i] = rand()%100; }

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sz); cudaMalloc(&d_B, sz); cudaMalloc(&d_C, sz);

        float total_gpu = 0, total_cpu = 0;

        for (int r = 0; r < RUNS; r++) {
            // GPU
            cudaEvent_t start, stop;
            cudaEventCreate(&start); cudaEventCreate(&stop);
            cudaEventRecord(start);
            cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
            cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);
            vector_add<<<BLOCKS, THREADS>>>(d_A, d_B, d_C, N);
            cudaMemcpy(h_C, d_C, sz, cudaMemcpyDeviceToHost);
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            total_gpu += ms;
            cudaEventDestroy(start); cudaEventDestroy(stop);

            // CPU
            float cpu_time = 0;
            for (int i = 0; i < CPU_REPEAT; i++) {
                auto t0 = high_resolution_clock::now();
                cpuVectorAdd(h_A, h_B, h_C, N);
                cpu_time += duration<float, std::milli>(high_resolution_clock::now() - t0).count();
            }
            total_cpu += cpu_time / CPU_REPEAT;
        }

        float seq     = total_cpu / RUNS;
        float par     = total_gpu / RUNS;
        float speedup = seq / par;
        float eff     = speedup / cores;
        float cost    = par * cores;

        printf("N=%-9d | SEQ=%.3f ms | PAR=%.3f ms | Speedup=%.3f | Eff=%.6f | Cost=%.3f\n",
               N, seq, par, speedup, eff, cost);
        fprintf(f, "%d,%.3f,%.3f,%.4f,%.6f,%.3f\n", N, seq, par, speedup, eff, cost);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
    }

    fclose(f);
    printf("\nResults saved to vector_result.csv\n");
}
