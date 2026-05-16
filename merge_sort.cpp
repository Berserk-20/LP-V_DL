#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include <cstdlib>

using namespace std;

int N_values[] = {20000, 40000, 60000, 80000, 100000};
const int TESTS = 5;

void fillRandom(vector<int>& a) {
    for (auto& x : a) x = rand() % 10000;
}

void mergeArr(vector<int>& a, vector<int>& tmp, int l, int m, int r) {
    int i = l, j = m + 1, k = l;
    while (i <= m && j <= r) tmp[k++] = (a[i] <= a[j]) ? a[i++] : a[j++];
    while (i <= m) tmp[k++] = a[i++];
    while (j <= r) tmp[k++] = a[j++];
    for (int x = l; x <= r; x++) a[x] = tmp[x];
}

void seqMerge(vector<int>& a, vector<int>& tmp, int l, int r) {
    if (l >= r) return;
    int m = (l + r) / 2;
    seqMerge(a, tmp, l, m);
    seqMerge(a, tmp, m + 1, r);
    mergeArr(a, tmp, l, m, r);
}

void parMergeRec(vector<int>& a, vector<int>& tmp, int l, int r) {
    if (r - l < 1000) { seqMerge(a, tmp, l, r); return; }
    int m = (l + r) / 2;
    #pragma omp task shared(a, tmp)
    parMergeRec(a, tmp, l, m);
    #pragma omp task shared(a, tmp)
    parMergeRec(a, tmp, m + 1, r);
    #pragma omp taskwait
    mergeArr(a, tmp, l, m, r);
}

void parMerge(vector<int>& a, vector<int>& tmp, int l, int r) {
    #pragma omp parallel
    #pragma omp single
    parMergeRec(a, tmp, l, r);
}

int main() {
    srand(42);
    int cores = omp_get_max_threads();
    cout << "Merge Sort Benchmark | Cores: " << cores << "\n";

    ofstream file("merge_result.csv");
    file << "N,SEQ,PAR,SPEEDUP,EFFICIENCY,COST\n";

    for (int t = 0; t < TESTS; t++) {
        int N = N_values[t];
        vector<int> orig(N), arr(N), tmp(N);
        fillRandom(orig);

        arr = orig;
        double t1 = omp_get_wtime(); seqMerge(arr, tmp, 0, N - 1); double seq = omp_get_wtime() - t1;
        if (seq < 1e-9) seq = 1e-9;

        arr = orig;
        t1 = omp_get_wtime(); parMerge(arr, tmp, 0, N - 1); double par = omp_get_wtime() - t1;

        double speedup = seq / par, eff = speedup / cores, cost = par * cores;

        cout << "\n==== N = " << N << " ====\n"
             << "Sequential : " << seq     << " s\n"
             << "Parallel   : " << par     << " s\n"
             << "Speedup    : " << speedup << "\n"
             << "Efficiency : " << eff     << "\n"
             << "Cost       : " << cost    << "\n";

        file << N << "," << seq << "," << par << "," << speedup << "," << eff << "," << cost << "\n";
    }

    file.close();
    cout << "\nResults saved to merge_result.csv\n";
}
