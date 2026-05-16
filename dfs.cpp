#include <iostream>
#include <vector>
#include <stack>
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

    void seqDFS(int s) {
        vector<bool> vis(V, false);
        stack<int> st;
        st.push(s);
        while (!st.empty()) {
            int u = st.top(); st.pop();
            if (vis[u]) continue;
            vis[u] = true;
            for (int n : adj[u])
                if (!vis[n]) st.push(n);
        }
    }

    void parDFS(int s) {
        vector<bool> vis(V, false);
        vector<int> curr = {s};
        while (!curr.empty()) {
            vector<int> next;
            #pragma omp parallel
            {
                vector<int> local;
                #pragma omp for schedule(dynamic)
                for (int i = 0; i < (int)curr.size(); i++) {
                    int u = curr[i];
                    bool process = false;
                    if (!vis[u]) {
                        #pragma omp critical
                        if (!vis[u]) { vis[u] = true; process = true; }
                    }
                    if (process)
                        for (int n : adj[u])
                            if (!vis[n]) local.push_back(n);
                }
                #pragma omp critical
                next.insert(next.end(), local.begin(), local.end());
            }
            curr.swap(next);
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

    ofstream file("dfs_result.csv");
    file << "N,SEQ,PAR,SPEEDUP,EFFICIENCY,COST\n";

    cout << "DFS Benchmark | " << THREADS << " threads\n";

    for (int N = 100; N <= 9000; N += 500) {
        Graph g(N);
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < min(N, i + N / 20); j++)
                g.addEdge(i, j);

        double t1 = omp_get_wtime(); g.seqDFS(0); double seq = omp_get_wtime() - t1;
        t1 = omp_get_wtime();        g.parDFS(0); double par = omp_get_wtime() - t1;

        double speedup    = seq / par;
        double efficiency = speedup / THREADS;
        double cost       = par * THREADS;

        cout << "\n==== N = " << N << " ====\n";
        printMetrics(seq, par, THREADS);

        file << N << "," << seq << "," << par << ","
             << speedup << "," << efficiency << "," << cost << "\n";
    }

    file.close();
    cout << "\nResults saved to dfs_result.csv\n";
}
