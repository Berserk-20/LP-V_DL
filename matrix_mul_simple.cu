#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <chrono>

using namespace std::chrono;

#define RUNS       5
#define CPU_REPEAT 5

__global__ void matmul(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) sum += A[row*N+k] * B[k*N+col];
        C[row*N+col] = sum;
    }
}

void cpuMatmul(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) sum += A[i*N+k] * B[k*N+j];
            C[i*N+j] = sum;
        }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cores = prop.multiProcessorCount * 128;
    printf("GPU: %s | Cores (est): %d\n\n", prop.name, cores);

    cudaFree(0); // warm-up

    int sizes[] = {32, 64, 128, 256, 512};
    int tests   = sizeof(sizes) / sizeof(int);

    FILE *f = fopen("matrix_result.csv", "w");
    fprintf(f, "N,SEQ_MS,PAR_MS,SPEEDUP,EFFICIENCY,COST\n");

    for (int t = 0; t < tests; t++) {
        int N    = sizes[t];
        int sz   = N * N * sizeof(float);

        float *h_A = (float*)malloc(sz), *h_B = (float*)malloc(sz), *h_C = (float*)malloc(sz);
        for (int i = 0; i < N*N; i++) { h_A[i] = rand()%10; h_B[i] = rand()%10; }

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, sz); cudaMalloc(&d_B, sz); cudaMalloc(&d_C, sz);
        cudaMemcpy(d_A, h_A, sz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, sz, cudaMemcpyHostToDevice);

        dim3 threads(16, 16), blocks((N+15)/16, (N+15)/16);
        float total_gpu = 0, total_cpu = 0;

        for (int r = 0; r < RUNS; r++) {
            // GPU
            cudaEvent_t start, stop;
            cudaEventCreate(&start); cudaEventCreate(&stop);
            cudaEventRecord(start);
            matmul<<<blocks, threads>>>(d_A, d_B, d_C, N);
            cudaDeviceSynchronize();
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float ms; cudaEventElapsedTime(&ms, start, stop);
            total_gpu += ms;
            cudaEventDestroy(start); cudaEventDestroy(stop);

            // CPU
            float cpu_time = 0;
            for (int i = 0; i < CPU_REPEAT; i++) {
                auto t0 = high_resolution_clock::now();
                cpuMatmul(h_A, h_B, h_C, N);
                cpu_time += duration<float, std::milli>(high_resolution_clock::now() - t0).count();
            }
            total_cpu += cpu_time / CPU_REPEAT;
        }

        float seq      = total_cpu / RUNS;
        float par      = total_gpu / RUNS;
        float speedup  = seq / par;
        float eff      = speedup / cores;
        float cost     = par * cores;

        printf("N=%d | SEQ=%.3f ms | PAR=%.3f ms | Speedup=%.2f | Eff=%.6f | Cost=%.3f\n",
               N, seq, par, speedup, eff, cost);
        fprintf(f, "%d,%.3f,%.3f,%.4f,%.6f,%.3f\n", N, seq, par, speedup, eff, cost);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
    }

    fclose(f);
    printf("\nResults saved to matrix_result.csv\n");
}
