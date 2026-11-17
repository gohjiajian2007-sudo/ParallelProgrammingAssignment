#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <random>
#include <vector>
#include <exception>

using namespace std;

// MPI Gaussian Elimination
bool mpi_ge(vector<double>& A, int N) {
    int rank, processCount;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    vector<int> rowCounts(processCount);
    vector<int> firstRow(processCount);
    vector<int> counts(processCount);
    vector<int> displacements(processCount);

    // Compute how many rows each process owns.
    int nextRow = 0;

    for (int r = 0; r < processCount; ++r) {
        rowCounts[r] = N / processCount;
        if (r < N % processCount)
            rowCounts[r] += 1;

        // First global row owned by process r.
        firstRow[r] = nextRow;

        // Number of matrix elements owned by process r.
        counts[r] = rowCounts[r] * N;

        // Offset of the first element owned by process r.
        displacements[r] = nextRow * N;

        nextRow += rowCounts[r];
    }

    // Local matrix block for this process.
    vector<double> localA(max(1, counts[rank]));

    // Buffer used to broadcast the normalized pivot row.
    vector<double> pivotRow(N);

    // Distribute the full matrix from rank 0 to all processes.
    MPI_Scatterv(
        A.data(),
        counts.data(),
        displacements.data(),
        MPI_DOUBLE,
        localA.data(),
        counts[rank],
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    // The process that owns the current pivot row.
    int owner = 0;

    for (int k = 0; k < N; ++k) {
        // Find which process owns the pivot row for column k.
        while (k >= firstRow[owner] + rowCounts[owner]) ++owner;

        int validPivot = 1;

        if (rank == owner) {
            // Local row index of the pivot row.
            int localPivot = k - firstRow[rank];

            // Pointer to the pivot row in localA.
            double* row = localA.data() + static_cast<size_t>(localPivot) * N;

            // Pivot value.
            double scale = row[k];

            if (scale == 0.0 || !isfinite(scale)) {
                validPivot = 0;
            }
            else {
                // Normalize the pivot row so that the pivot becomes 1.0.
                for (int col = k; col < N; ++col) {
                    row[col] /= scale;
                    pivotRow[col] = row[col];

                    if (!isfinite(row[col])) {
                        validPivot = 0;
                    }
                }
            }
        }

        // All processes decide together whether the pivot is valid.
        MPI_Bcast(
            &validPivot,
            1,
            MPI_INT,
            owner,
            MPI_COMM_WORLD
        );

        if (!validPivot) {
            return false;
        }

        // Broadcast the normalized pivot row from the owner to all processes.
        MPI_Bcast(
            pivotRow.data() + k,
            N - k,
            MPI_DOUBLE,
            owner,
            MPI_COMM_WORLD
        );

        // Each process eliminates the pivot column from its own rows.
        for (int localRow = 0; localRow < rowCounts[rank]; ++localRow) {
            int globalRow = firstRow[rank] + localRow;

            // Skip rows that are already processed.
            if (globalRow <= k) continue;

            double* row = localA.data() + static_cast<size_t>(localRow) * N;

            // Multiplier for this row.
            double elimScale = row[k];

            // Set the pivot column entry to zero.
            row[k] = 0.0;

            // Perform row = row - elimScale * pivotRow.
            for (int col = k + 1; col < N; ++col) {
                row[col] -= pivotRow[col] * elimScale;
            }
        }
    }

    // Check whether the local result contains NaN or infinity.
    int localOK = 1;
    int allOK = 0;

    for (double value : localA) {
        if (!isfinite(value)) {
            localOK = 0;
            break;
        }
    }

    // All processes must agree that the result is finite.
    MPI_Allreduce(
        &localOK,
        &allOK,
        1,
        MPI_INT,
        MPI_MIN,
        MPI_COMM_WORLD
    );

    if (!allOK) return false;

    // Gather all local blocks back to rank 0.
    MPI_Gatherv(
        localA.data(),
        counts[rank],
        MPI_DOUBLE,
        A.data(),
        counts.data(),
        displacements.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    return true;
}

// Main program
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int processCount = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    try {
        int N = 0;

        // Only rank 0 reads the input matrix size.
        if (rank == 0) {
            long long inputN = 0;
            printf("Enter the matrix size (N): ");
            fflush(stdout);

            if (scanf_s("%lld", &inputN) == 1 && inputN >= 1 && inputN <= 5000) {
                N = static_cast<int>(inputN);
            }
            else {
                fprintf(stderr, "Invalid size.\n");
            }
        }

        // Broadcast the matrix size to all processes.
        MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (N == 0) {
            MPI_Finalize();
            return 1;
        }

        vector<double> A;

        // Only rank 0 generates the full matrix.
        if (rank == 0) {
            A.resize(static_cast<size_t>(N) * N);

            // Fixed random seed for reproducibility.
            mt19937 gen(42);
            uniform_int_distribution<int> dist(1, 10);

            for (double& value : A) {
                value = dist(gen);
            }
        }

        // Wait until all processes are ready, then start timing.
        MPI_Barrier(MPI_COMM_WORLD);

        double start = MPI_Wtime();
        bool success = mpi_ge(A, N);
        double localTime = MPI_Wtime() - start;
        double parallelTime = 0.0;

        // Take the running time of the slowest process.
        MPI_Reduce(
            &localTime,
            &parallelTime,
            1,
            MPI_DOUBLE,
            MPI_MAX,
            0,
            MPI_COMM_WORLD
        );

        // Rank 0 prints the result.
        if (rank == 0 && success) {
            printf("\nMatrix Size: %d x %d\n", N, N);
            printf("MPI Processes: %d\n", processCount);
            printf("Computational Time: %.3f ms\n", parallelTime * 1000.0);
        }
        else if (rank == 0 && !success) {
            printf("\nGaussian Elimination failed: zero/non-finite pivot or non-finite arithmetic.\n");
        }

        MPI_Finalize();
        return success ? 0 : 2;

    }
    catch (const exception& e) {
        fprintf(stderr, "Rank %d error: %s\n", rank, e.what());

        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}