#include <iostream>
#include <vector>
#include <queue>
#include <fstream>
#include <omp.h>

using namespace std;

struct Graph {
    int V;
    vector<vector<int>> adj;

    Graph(int V) : V(V), adj(V) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void seqBFS(int s) {
        vector<bool> vis(V, false);
        queue<int> q;
        vis[s] = true;
        q.push(s);
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int n : adj[u])
                if (!vis[n]) { vis[n] = true; q.push(n); }
        }
    }

    void parBFS(int s) {
        vector<bool> vis(V, false);
        vector<int> frontier = {s};
        vis[s] = true;
        while (!frontier.empty()) {
            vector<int> next;
            #pragma omp parallel
            {
                vector<int> local;
                #pragma omp for schedule(dynamic)
                for (int i = 0; i < (int)frontier.size(); i++) {
                    for (int n : adj[frontier[i]]) {
                        if (!vis[n]) {
                            #pragma omp critical
                            if (!vis[n]) { vis[n] = true; local.push_back(n); }
                        }
                    }
                }
                #pragma omp critical
                next.insert(next.end(), local.begin(), local.end());
            }
            frontier.swap(next);
        }
    }
};

void printMetrics(double seq, double par, int threads) {
    double speedup    = seq / par;
    double efficiency = speedup / threads;
    double cost       = par * threads;
    cout << "Sequential : " << seq       << " s\n"
         << "Parallel   : " << par       << " s\n"
         << "Speedup    : " << speedup   << "\n"
         << "Efficiency : " << efficiency << "\n"
         << "Cost       : " << cost      << "\n";
}

int main() {
    const int THREADS = 4;
    omp_set_num_threads(THREADS);

    ofstream file("bfs_result.csv");
    file << "N,SEQ,PAR,SPEEDUP,EFFICIENCY,COST\n";

    cout << "BFS Benchmark | " << THREADS << " threads\n";

    for (int N = 100; N <= 9000; N += 500) {
        Graph g(N);
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < min(N, i + N / 20); j++)
                g.addEdge(i, j);

        double t1 = omp_get_wtime(); g.seqBFS(0); double seq = omp_get_wtime() - t1;
        t1 = omp_get_wtime();        g.parBFS(0); double par = omp_get_wtime() - t1;

        double speedup    = seq / par;
        double efficiency = speedup / THREADS;
        double cost       = par * THREADS;

        cout << "\n==== N = " << N << " ====\n";
        printMetrics(seq, par, THREADS);

        file << N << "," << seq << "," << par << ","
             << speedup << "," << efficiency << "," << cost << "\n";
    }

    file.close();
    cout << "\nResults saved to bfs_result.csv\n";
}
