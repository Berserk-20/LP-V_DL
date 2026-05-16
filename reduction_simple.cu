#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cuda.h>

using namespace std::chrono;

#define RUNS       5
#define CPU_REPEAT 5
#define THREADS    1024

__global__ void reduction(float *A, float *gmin, float *gmax, float *gsum, int N) {
    __shared__ float smin[THREADS], smax[THREADS], ssum[THREADS];

    int tid = threadIdx.x;
    int i   = blockIdx.x * blockDim.x + tid;

    smin[tid] = (i < N) ? A[i] :  1e9f;
    smax[tid] = (i < N) ? A[i] : -1e9f;
    ssum[tid] = (i < N) ? A[i] :  0.0f;
    __syncthreads();

    for (int s = THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smin[tid] = min(smin[tid], smin[tid + s]);
            smax[tid] = max(smax[tid], smax[tid + s]);
            ssum[tid] += ssum[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        gmin[blockIdx.x] = smin[0];
        gmax[blockIdx.x] = smax[0];
        gsum[blockIdx.x] = ssum[0];
    }
}

void cpuReduce(float *A, int N, float &mn, float &mx, float &sm) {
    mn = mx = A[0]; sm = 0;
    for (int i = 0; i < N; i++) {
        if (A[i] < mn) mn = A[i];
        if (A[i] > mx) mx = A[i];
        sm += A[i];
    }
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int cores = prop.multiProcessorCount * 128;
    printf("GPU: %s | Cores (est): %d\n\n", prop.name, cores);

    cudaFree(0); // warm-up

    int N_values[] = {20000, 40000, 60000, 80000, 100000,120000,140000,160000,180000,200000};
    int tests = sizeof(N_values) / sizeof(int);

    FILE *f = fopen("reduction_result.csv", "w");
    fprintf(f, "N,RUN,SEQ_MS,PAR_MS,SPEEDUP,EFFICIENCY,COST\n");

    for (int t = 0; t < tests; t++) {
        int N     = N_values[t];
        int blocks = (N + THREADS - 1) / THREADS;
        printf("Running N = %d\n", N);

        float *h_A = (float *)malloc(N * sizeof(float));

        float *d_A, *d_min, *d_max, *d_sum;
        cudaMalloc(&d_A,   N      * sizeof(float));
        cudaMalloc(&d_min, blocks * sizeof(float));
        cudaMalloc(&d_max, blocks * sizeof(float));
        cudaMalloc(&d_sum, blocks * sizeof(float));

        for (int r = 0; r < RUNS; r++) {
            for (int i = 0; i < N; i++) h_A[i] = rand() % 100;
            cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

            cudaEvent_t start, stop;
            cudaEventCreate(&start); cudaEventCreate(&stop);
            cudaEventRecord(start);
            for (int k = 0; k < 5; k++)
                reduction<<<blocks, THREADS>>>(d_A, d_min, d_max, d_sum, N);
            cudaDeviceSynchronize();
            cudaEventRecord(stop); cudaEventSynchronize(stop);
            float par_time;
            cudaEventElapsedTime(&par_time, start, stop);
            cudaEventDestroy(start); cudaEventDestroy(stop);

            float cpu_total = 0;
            for (int i = 0; i < CPU_REPEAT; i++) {
                float mn, mx, sm;
                auto t0 = high_resolution_clock::now();
                cpuReduce(h_A, N, mn, mx, sm);
                cpu_total += duration<float, std::milli>(high_resolution_clock::now() - t0).count();
            }
            float seq_time = cpu_total / CPU_REPEAT;

            float speedup = seq_time / par_time;
            float eff     = speedup / cores;
            float cost    = par_time * cores;

            printf("  Run %d | SEQ=%.4f ms  PAR=%.4f ms  Speedup=%.4f  Eff=%.6f  Cost=%.4f\n",
                   r, seq_time, par_time, speedup, eff, cost);
            fprintf(f, "%d,%d,%f,%f,%f,%f,%f\n", N, r, seq_time, par_time, speedup, eff, cost);
        }

        cudaFree(d_A); cudaFree(d_min); cudaFree(d_max); cudaFree(d_sum);
        free(h_A);
    }

    fclose(f);
    printf("\nResults saved to reduction_result.csv\n");
}
