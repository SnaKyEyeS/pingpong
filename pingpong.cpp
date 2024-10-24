#include <assert.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// Macro for checking errors in CUDA API calls
#define m_cudart_check_errors(res)                                                              \
    do {                                                                                        \
        cudaError_t __err = res;                                                                \
        if (__err != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA RT error %s at %s:%d\n", cudaGetErrorString(__err), __FILE__, \
                    __LINE__);                                                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

int main(int argc, char *argv[]) {
    /* -------------------------------------------------------------------------------------------
            MPI Initialization
    --------------------------------------------------------------------------------------------*/
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the intra-node communicator
    const int host_id = gethostid();
    MPI_Comm node_comm;
    MPI_Comm_split(MPI_COMM_WORLD, host_id, world_rank, &node_comm);
    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);
    assert(node_size == world_size / 2);

    // Get the node ID
    int node_id = (world_rank == 0) ? 0 : 1;
    MPI_Bcast(&node_id, 1, MPI_INT, 0, node_comm);

    // Get the pair-wise communicators for the ping-pong
    MPI_Comm pair_comm;
    MPI_Comm_split(MPI_COMM_WORLD, node_rank, world_rank, &pair_comm);
    int pair_rank, pair_size;
    MPI_Comm_rank(pair_comm, &pair_rank);
    MPI_Comm_size(pair_comm, &pair_size);
    assert(pair_size == 2);

    // Map MPI ranks to GPUs
    int num_devices = -1;
    cudaDeviceProp prop;
    m_cudart_check_errors(cudaGetDeviceCount(&num_devices));
    m_cudart_check_errors(cudaSetDevice(node_rank % num_devices));
    m_cudart_check_errors(cudaGetDeviceProperties(&prop, node_rank % num_devices));

    // Get my GPU ID and print some info
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    const int gpu_id = atoi(getenv("CUDA_VISIBLE_DEVICES"));
    printf("[node #%d (%d, %s) - rank #%d] Visible devices: %d, got: #%d (%s)\n", node_id, host_id, hostname, node_rank, num_devices, gpu_id, prop.name);

    /* -------------------------------------------------------------------------------------------
            Loop from 8 B to 1 GB
    --------------------------------------------------------------------------------------------*/
    MPI_Request req;
    MPI_Status  stat;
    for (int i = 0; i <= 27; i++) {
        long int N = 1 << i;

        // Allocate memory for A on CPU
        double *A = (double *)malloc(N * sizeof(double));

        // Initialize all elements of A to random values
        for (int i = 0; i < N; i++) {
            A[i] = (double)rand() / (double)RAND_MAX;
        }

        double *d_A;
        m_cudart_check_errors(cudaMalloc(&d_A, N * sizeof(double)));
        m_cudart_check_errors(cudaMemcpy(d_A, A, N * sizeof(double), cudaMemcpyHostToDevice));

        const int tag1       = 10;
        const int tag2       = 20;
        const int loop_count = 50;

        // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
        double start_time, stop_time, elapsed_time = 0;
        for (int i = -5; i < loop_count; i++) {
            if (0 == node_id) {
                MPI_Irecv(d_A, N, MPI_DOUBLE, 1, tag1, pair_comm, &req);
                MPI_Barrier(MPI_COMM_WORLD);  // barrier is on world comm
                start_time = MPI_Wtime();
                MPI_Send(d_A, N, MPI_DOUBLE, 1, tag2, pair_comm);
                MPI_Wait(&req, &stat);
                stop_time = MPI_Wtime();
                // only count the non-warm-up loops
                if (i >= 0)
                    elapsed_time += stop_time - start_time;

            } else if (1 == node_id) {
                MPI_Irecv(d_A, N, MPI_DOUBLE, 0, tag2, pair_comm, &req);
                MPI_Barrier(MPI_COMM_WORLD);  // barrier is on world comm
                MPI_Wait(&req, &stat);
                MPI_Send(d_A, N, MPI_DOUBLE, 0, tag1, pair_comm);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (0 == node_id) {
            long int num_B                 = N * sizeof(double);
            long int B_in_GB               = 1 << 30;
            double   num_GB                = (double)num_B / (double)B_in_GB;
            double   avg_time_per_transfer = elapsed_time / (2.0 * (double)loop_count);

            printf("Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n", num_B, avg_time_per_transfer, num_GB / avg_time_per_transfer);
        }

        m_cudart_check_errors(cudaFree(d_A));
        free(A);
    }

    MPI_Finalize();

    return 0;
}

