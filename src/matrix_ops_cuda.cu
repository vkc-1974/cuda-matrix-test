#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, \
                "ERROR: CUDA error: %s at %s:%d\n", \
                cudaGetErrorString(err), \
                __FILE__, \
                __LINE__); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(stat) do { \
    cublasStatus_t err__ = (stat); \
    if (err__ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, \
                "ERROR: cuBLAS error %d (%s) at %s:%d\n", \
                (int)err__, \
                cublasGetErrorString(err__), \
                __FILE__, \
                __LINE__); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUSOLVER(stat) do { \
    if (stat != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, \
                "ERROR: cuSOLVER error %d at %s:%d\n", \
                (int)stat, \
                __FILE__, \
                __LINE__); \
        exit(1); \
    } \
} while(0)

// Matrix dimensions, max value for GTX1650M 4Gb 896 CUDA cores
// (original OpenCL version: 8 * 110 (max value))
const unsigned int N = 8 * 560;  // Rows of A, rows/cols of C
const unsigned int M = 8 * 560;

#ifdef PRINT_SAMPLE
const unsigned int L = 8 * 200 /* * 110 */;
#endif //  PRINT_SAMPLE

typedef double FLOAT_TYPE;

const char* cublasGetErrorString(const cublasStatus_t status) {
    switch(status)
    {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown error";
}

// CPU Gauss-Jordan (invert a matrix)
void gauss_jordan_cpu(FLOAT_TYPE *C,
                      FLOAT_TYPE *C_inverse,
                      const unsigned int n) {
    FLOAT_TYPE *augmented = (FLOAT_TYPE*)malloc(n * (2 * n) * sizeof(FLOAT_TYPE));
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            augmented[i * (2 * n) + j] = C[i * n + j];
            augmented[i * (2 * n) + (n + j)] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (unsigned int pivot = 0; pivot < n; pivot++) {
        FLOAT_TYPE pivot_value = augmented[pivot * (2 * n) + pivot];
        if (fabs(pivot_value) < 1e-6) {
            fprintf(stderr,
                    "ERROR: Singular matrix at pivot %d\n",
                    pivot);
            free(augmented); exit(1);
        }
        for (unsigned int j = 0; j < 2 * n; j++) {
            augmented[pivot * (2 * n) + j] /= pivot_value;
        }
        for (unsigned int i = 0; i < n; i++) {
            if (i != pivot) {
                FLOAT_TYPE factor = augmented[i * (2 * n) + pivot];
                for (unsigned int j = 0; j < 2 * n; j++) {
                    augmented[i * (2 * n) + j] -= factor * augmented[pivot * (2 * n) + j];
                }
            }
        }
    }

    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            C_inverse[i * n + j] = augmented[i * (2 * n) + (n + j)];
        }
    }
    free(augmented);
}

// Invert a matrix on GPU
//   Input:  d_A   - original matrix on the device (dimension n x n),
//                   is destroyed
//   Output: d_inv - inverted matrix on the device (dimension n x n),
//                   should be allocated before the call
void compute_inverse_gpu(cusolverDnHandle_t cusolverH,
                         double *d_A,
                         double *d_inv,
                         const unsigned int n) {
    int *d_info = nullptr;
    int *d_pivot = nullptr;
    double *d_work = nullptr;
    int lwork = 0;

    // 1. Allocate auxiliary memory
    CHECK_CUDA(cudaMalloc(&d_info,
                          sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_pivot,
                          n * sizeof(int)));

    // 2. Ask the buffer for resulting matrix
    CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(cusolverH,
                                               n,
                                               n,
                                               d_A,
                                               n,
                                               &lwork));
    CHECK_CUDA(cudaMalloc(&d_work,
                          sizeof(double) * lwork));

    // 3. LU decomposition: A → L,U + pivot (d_A is rewritten!)
    CHECK_CUSOLVER(cusolverDnDgetrf(cusolverH,
                                    n,
                                    n,
                                    d_A,
                                    n,
                                    d_work,
                                    d_pivot,
                                    d_info));

    // Check calculation issues (singular or numerical issues)
    int h_info;
    CHECK_CUDA(cudaMemcpy(&h_info,
                          d_info,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr,
                "ERROE: LU factorization failed: info = %d (pivot %d)\n",
                h_info,
                -h_info);
        // TBD: process the issue or stop running
    }

    // 4. Prepare the right-hand side (the identity matrix) in `d_inv`
    // Fill `d_inv` with the identity matrix
    CHECK_CUDA(cudaMemset(d_inv,
                          0,
                          n * n * sizeof(double)));

    // Fill the diagonal with 1.0
    // TBD: for simplicity it is done on the host; a separate kernel can be done in the future
    double one = 1.0;
    for (unsigned int i = 0; i < n; ++i) {
        CHECK_CUDA(cudaMemcpy(d_inv + i * n + i,
                              &one,
                              sizeof(double),
                              cudaMemcpyHostToDevice));
    }

    // 5. Calculate A × X = I → X = A⁻¹
    // `getrs` does it for several right parts at the same time
    int nrhs = n;  // `n` right parts (columns I)
    CHECK_CUSOLVER(cusolverDnDgetrs(cusolverH,
                                    CUBLAS_OP_N,        // no transpose
                                    n,
                                    nrhs,
                                    d_A,                // LU factorized matrix
                                    n,
                                    d_pivot,
                                    d_inv,              // input — I, output — X = A⁻¹
                                    n,
                                    d_info));
    // Check resulting matrix again
    CHECK_CUDA(cudaMemcpy(&h_info,
                          d_info,
                          sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        fprintf(stderr,
                "ERROR: getrs failed: info = %d\n",
                h_info);
    }

    // Очистка
    cudaFree(d_info);
    cudaFree(d_pivot);
    cudaFree(d_work);
}

// Kernel to calculate max error value (residual = C * C_inv - I)
__global__ void compute_max_residual_kernel(const double *d_residual,
                                            int n,
                                            double *d_max_err) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    double val = 0.0;
    if (idx < n * n) {
        int i = idx / n;
        int j = idx % n;
        double expected = (i == j) ? 1.0 : 0.0;
        val = fabs(d_residual[idx] - expected);
    }

    sdata[tid] = val;
    __syncthreads();

    // Reduce в блоке
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((unsigned long long*)d_max_err, __double_as_longlong(sdata[0]));
    }
}

// Verification routine
bool verify_inverse_gpu(cublasHandle_t cublasH,
                        double *d_C,           // origunal matrix
                        double *d_C_inv,       // inverted matrix
                        int n,
                        double tol = 1e-8) {
    double *d_product = nullptr;
    double *d_residual = nullptr;
    double *d_max_err = nullptr;
    double h_max_err = 0.0;

    CHECK_CUDA(cudaMalloc(&d_product,
                          n*n*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_residual,
                          n*n*sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_max_err,
                          sizeof(double)));

    double alpha = 1.0;
    double beta  = 0.0;

    // product = C * C_inv
    CHECK_CUBLAS(cublasDgemm(cublasH,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             n,
                             n,
                             n,
                             &alpha,
                             d_C,
                             n,
                             d_C_inv,
                             n,
                             &beta,
                             d_product,
                             n));

    // residual = product - I
    // To simplify it just copy `product → residual` (not on the host but with using of a separate kernel)
    // To sub I, a separate kernel can be used (existing one can be updated)

    // Start kernel running on product (as residual)
    int threads = 256;
    int blocks = (n*n + threads - 1) / threads;

    CHECK_CUDA(cudaMemset(d_max_err,
                          0,
                          sizeof(double)));

    compute_max_residual_kernel<<<blocks, threads, threads*sizeof(double)>>>(d_product, n, d_max_err);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(&h_max_err,
                          d_max_err,
                          sizeof(double),
                          cudaMemcpyDeviceToHost));

    printf("Maximum absolute error for (%u x %u): |C * C⁻¹ - I|_∞ = %.3e\n",
           n,
           n,
           h_max_err);

    bool ok = (h_max_err <= tol);

    printf("%s (tol = %.1e)\n",
           (ok ? "Check has passed" : "Check has failed"),
           tol);

    cudaFree(d_product);
    cudaFree(d_residual);
    cudaFree(d_max_err);

    return ok;
}

// CUDA kernel: matrix multiplication C = A * B
__global__ void matrix_multiply_kernel(const FLOAT_TYPE *A,
                                       const FLOAT_TYPE *B,
                                       FLOAT_TYPE *C,
                                       const unsigned int N,
                                       const unsigned int M)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        FLOAT_TYPE sum = 0.0;
        for (unsigned int k = 0; k < M; k++) {
            sum += A[row * M + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CUDA kernel: normalize rows and transpose in-place (C becomes normalized transpose)
__global__ void normalize_transpose_inplace_kernel(FLOAT_TYPE *C,
                                                   const unsigned int N)
{
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N && i <= j) {
        // Compute row sums (only once per row)
        __shared__ FLOAT_TYPE row_sum_i[32];  // assume blockDim.y <= 32
        __shared__ FLOAT_TYPE row_sum_j[32];

        row_sum_i[threadIdx.y] = 0.0;
        row_sum_j[threadIdx.y] = 0.0;

        for (int k = 0; k < N; k++) {
            row_sum_i[threadIdx.y] += C[i * N + k];
            if (i != j) row_sum_j[threadIdx.y] += C[j * N + k];
        }

        __syncthreads();

        // Normalize diagonal
        if (i == j) {
            if (row_sum_i[threadIdx.y] > 1e-6) {
                C[i * N + i] /= row_sum_i[threadIdx.y];
            } else {
                C[i * N + i] = 0.0;
            }
        } else {
            FLOAT_TYPE norm_ij = (row_sum_i[threadIdx.y] > 1e-6) ? C[i * N + j] / row_sum_i[threadIdx.y] : 0.0;
            FLOAT_TYPE norm_ji = (row_sum_j[threadIdx.y] > 1e-6) ? C[j * N + i] / row_sum_j[threadIdx.y] : 0.0;
            C[j * N + i] = norm_ij;
            C[i * N + j] = norm_ji;
        }
    }
}

void check_inverse_simple(FLOAT_TYPE *C,
                          FLOAT_TYPE *C_inverse,
                          const unsigned int n) {
    printf("\nChecking C * C_inverse (should be identity matrix, %ux%u):\n",
           n,
           n);
    for (unsigned int i = 0; i < n; i++) {
        for (unsigned int j = 0; j < n; j++) {
            FLOAT_TYPE sum = 0.0;
            for (unsigned int k = 0; k < n; k++) {
                sum += C[i * n + k] * C_inverse[k * n + j];
            }
            printf("%.6f ", sum);
        }
        printf("\n");
    }
}

int main() {
    const size_t size_A = N * M * sizeof(FLOAT_TYPE);
    const size_t size_B = M * N * sizeof(FLOAT_TYPE);
    const size_t size_C = N * N * sizeof(FLOAT_TYPE);

    cublasHandle_t cublasH = nullptr;
    CHECK_CUBLAS(cublasCreate(&cublasH));

    cusolverDnHandle_t cusolverH = nullptr;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    
    // Host memory
    FLOAT_TYPE *h_A = (FLOAT_TYPE*)malloc(size_A);
    FLOAT_TYPE *h_B = (FLOAT_TYPE*)malloc(size_B);
    FLOAT_TYPE *h_C = (FLOAT_TYPE*)malloc(size_C);
    FLOAT_TYPE *h_C_inverse = (FLOAT_TYPE*)malloc(size_C);

    // Initialize matrices (same as before)
    srand(time(NULL));
    for (size_t i = 0; i < N * M; i++) {
        if (i / M == i % M) {
            h_A[i] = 10.0 + (FLOAT_TYPE)(rand() % 100) / 10.0;
        } else {
            h_A[i] = (FLOAT_TYPE)(rand() % 10) / 10.0;
        }
    }
    for (size_t i = 0; i < M * N; i++) {
        h_B[i] = (FLOAT_TYPE)(rand() % 100) / 10.0;
    }

    // Device memory
    FLOAT_TYPE *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A,
                          size_A));
    CHECK_CUDA(cudaMalloc(&d_B,
                          size_B));
    CHECK_CUDA(cudaMalloc(&d_C,
                          size_C));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_A,
                          h_A,
                          size_A,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B,
                          h_B,
                          size_B,
                          cudaMemcpyHostToDevice));

    // Kernel launch params
    dim3 block(16, 16);  // 256 threads/block — quite good for most amount of GPUs
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Matrix multiplication
#if 0
    matrix_multiply_kernel<<<grid, block>>>(d_A, d_B, d_C, N, M);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
#else
    double alpha = 1.0;
    double beta  = 0.0;

    CHECK_CUBLAS(cublasDgemm(cublasH,
                             CUBLAS_OP_N, CUBLAS_OP_N,   // A and B are not transposed
                             N,                          // m = rows of C and op(A)
                             N,                          // n = cols of C and op(B)
                             M,                          // k = cols of op(A) = rows of op(B)
                             &alpha,
                             d_A, M,                     // A, lda = leading dimension of A = M (row-major)
                             d_B, N,                     // B, ldb = N
                             &beta,
                             d_C, N));                   // C, ldc = N
#endif
    
    // Copy C back to host
    CHECK_CUDA(cudaMemcpy(h_C,
                          d_C,
                          size_C,
                          cudaMemcpyDeviceToHost));

    // 1. Allocate the memory for inverted matrix
    double *d_C_inv = nullptr;
    CHECK_CUDA(cudaMalloc(&d_C_inv,
                          size_C));

    // 2. Make a copy of original matrix as `getrf` destroys its input
    double *d_C_copy = nullptr;
    CHECK_CUDA(cudaMalloc(&d_C_copy,
                          size_C));
    CHECK_CUDA(cudaMemcpy(d_C_copy,
                          d_C,
                          size_C,
                          cudaMemcpyDeviceToDevice));

    // 3. Calculate inverted matrix on GPU
    compute_inverse_gpu(cusolverH,
                        d_C_copy,
                        d_C_inv,
                        N);

    // 4. Verify inverted matrix on GPU
    bool verification_ok = verify_inverse_gpu(cublasH,
                                              d_C,
                                              d_C_inv,
                                              N,
                                              1e-8);

    if (!verification_ok) {
        fprintf(stderr,
                "ERROR: Verification connot be passed "
                "as numerical instability or error is possible\n");
    }

#ifdef PRINT_SAMPLE
    // 5. (Optional): copy `C_inv` to host to print a sample
    CHECK_CUDA(cudaMemcpy(h_C_inverse,
                          d_C_inv,
                          size_C,
                          cudaMemcpyDeviceToHost));

    // 6. (Optional): print a sample Печать сэмпла
    printf("\nSample of inverse matrix (%ux%u):\n", L, L);
    for (unsigned int i = 0; i < L; i++) {
        for (unsigned int j = 0; j < L; j++) {
            printf("%.6f ", h_C_inverse[i * N + j]);
        }
        printf("\n");
    }
#endif //  PRINT_SAMPLE

    // Cleanup auxiliary buffers
    cudaFree(d_C_inv);
    cudaFree(d_C_copy);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_inverse);

    // Release libraries' handlers
    if (cusolverH) {
        cusolverDnDestroy(cusolverH);    
    }
    
    if (cublasH) {
        cublasDestroy(cublasH);
    }
    
    return 0;
}
