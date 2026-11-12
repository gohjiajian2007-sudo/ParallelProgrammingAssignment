#include "serialGE.cpp"
#include <iostream>
#include<random>
#include <chrono>
using namespace std;
using namespace chrono;


int main()
{
    while (true) {
    size_t N;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(1, 10);
    cout << "Enter the matrix size (N): ";
    cin >> N;
    // Generate random N x N matrix
    vector<double> A(N * N);

    for (size_t i = 0; i < N * N; i++) {
        A[i] = dist(gen);
    }


    // Start measuring computational time
    auto start = high_resolution_clock::now();


    // Perform Gaussian Elimination
    serial_ge(A, N);

    // Stop measuring computational time
    auto stop = high_resolution_clock::now();

    // Calculate execution time
    auto duration = duration_cast<milliseconds>(stop - start);

    cout << "\nMatrix Size: " << N << " x " << N << endl;
    cout << "Computational Time: "
        << duration.count() << " ms" << endl;
}
    return 0;
}

