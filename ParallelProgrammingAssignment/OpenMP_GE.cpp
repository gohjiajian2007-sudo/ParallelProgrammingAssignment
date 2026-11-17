#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

//OpenMP implementation
static void openMP_ge(vector<double>& A, size_t N) {

	double scale = 0.0; //Declare outside the loop, so can be shared among all threads

    #pragma omp parallel shared(A, N, scale)
    {
        //For each row in the matrix
        for (size_t pivot_row = 0; pivot_row < N; pivot_row++) {

            #pragma omp single
            {
                //Save the pivot value
                scale = A[N * pivot_row + pivot_row];

                //Normalize the pivot row
                for (size_t col = pivot_row; col < N; col++) {
                    A[N * pivot_row + col] /= scale;
                }
            }

            //execute the following loop in parallel using multiple threads
            #pragma omp for schedule(static)  
            for (int sub_row = pivot_row + 1; sub_row < N; sub_row++) {

                //Calculate the elimination factor
                double elim_scale = A[sub_row * N + pivot_row];

                //Update the remaining elements 
                for (int col = pivot_row; col < N; col++) {
                    A[N * sub_row + col] -= A[N * pivot_row + col] * elim_scale;
                }

            }
        }
    }
   
}
