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

void seqBubble(vector<int>& a) {
    int N = a.size();
    for (int i = 0; i < N - 1; i++)
        for (int j = 0; j < N - i - 1; j++)
            if (a[j] > a[j + 1]) swap(a[j], a[j + 1]);
}

void parBubble(vector<int>& a) {
    int N = a.size();
    #pragma omp parallel
    for (int phase = 0; phase < N; phase++) {
        int start = phase % 2;
        #pragma omp for
        for (int i = start; i < N - 1; i += 2)
            if (a[i] > a[i + 1]) swap(a[i], a[i + 1]);
    }
}

int main() {
    srand(42);
    int cores = omp_get_max_threads();
    cout << "Bubble Sort Benchmark | Cores: " << cores << "\n";

    ofstream file("bubble_result.csv");
    file << "N,SEQ,PAR,SPEEDUP,EFFICIENCY,COST\n";

    for (int t = 0; t < TESTS; t++) {
        int N = N_values[t];
        vector<int> orig(N), arr(N);
        fillRandom(orig);

        arr = orig;
        double t1 = omp_get_wtime(); seqBubble(arr); double seq = omp_get_wtime() - t1;
        if (seq < 1e-9) seq = 1e-9;

        arr = orig;
        t1 = omp_get_wtime(); parBubble(arr); double par = omp_get_wtime() - t1;

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
    cout << "\nResults saved to bubble_result.csv\n";
}
