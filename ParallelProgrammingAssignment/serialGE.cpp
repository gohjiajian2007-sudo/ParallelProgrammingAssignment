// Implementation of serial GE function
#include <iostream>
#include <vector>
#include <iomanip>
using namespace std;
// Serial implementation
static void serial_ge(vector<double>& A, size_t N) {
    // For each row in the matrix
    for (size_t pivot_row = 0; pivot_row < N; pivot_row++) {
        // Save the pivot value
        double scale = A[N * pivot_row + pivot_row];

        // Divide the remaining elements in the vector
        for (size_t col = pivot_row; col < N; col++) {
            A[N * pivot_row + col] /= scale;
        }

        // Eliminate this column from the rest of the matrix
        // For each remaining row in the matrix
        for (size_t sub_row = pivot_row + 1; sub_row < N; sub_row++) {
            // Scale factor for eliminating for elemenating this column
            double elim_scale = A[sub_row * N + pivot_row];

            // Finish the remainder of the row
            for (size_t col = pivot_row; col < N; col++) {
                A[N * sub_row + col] -= A[N * pivot_row + col] * elim_scale;
            }

        }
    }
}
