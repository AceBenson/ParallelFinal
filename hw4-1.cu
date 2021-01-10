#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>
#include <omp.h>
#include <chrono>
#include <ctime>
#define INF 1073741823
#define BS 64
#define BS_DIM 32

using namespace std;

__device__ void floyd(int C[][BS], int A[][BS], int B[][BS], int ti, int tj) {
    int sum;
    for(int k=0; k<BS; k++) {
        for(int i=0; i<BS; i+=BS_DIM) {
            for(int j=0; j<BS; j+=BS_DIM) {
                sum = A[ti+i][k] + B[k][tj+j];
                if(C[ti+i][tj+j] > sum) {
                    C[ti+i][tj+j] = sum;
                }
            }
        }
        __syncthreads();
    }
}

//Wkk, Wkk, Wkk
__global__ void floyd_warshall_phase1(int* mat, int V, int k) {
    __shared__ int C[BS][BS];
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    int kti = k*BS+ti;
    int ktj = k*BS+tj;
    for(int i=0; i<BS; i+=BS_DIM) {
        for(int j=0; j<BS; j+=BS_DIM) {
            C[ti+i][tj+j] = mat[(kti+i)*V+ktj+j];
        }
    }
    
    __syncthreads();
    floyd(C, C, C, ti, tj);
    for(int i=0; i<BS; i+=BS_DIM) {
        for(int j=0; j<BS; j+=BS_DIM) {
            mat[(kti+i)*V+ktj+j] = C[ti+i][tj+j];
        }
    }
}

//Wkj, Wkk, Wkj
//Wik, Wik, Wkk
__global__ void floyd_warshall_phase2(int* mat, int V, int k) {
    __shared__ int C[BS][BS];
    __shared__ int A[BS][BS];
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    if(bidx==k) return;
    for(int i=0; i<BS; i+=BS_DIM) {
        for(int j=0; j<BS; j+=BS_DIM) {
            A[ti+i][tj+j] = mat[(k*BS+ti+i)*V+k*BS+tj+j];
        }
    }
    int base_i, base_j;
    if(bidy==0) {
        base_i = k*BS+ti;
        base_j = bidx*BS+tj;
        for(int i=0; i<BS; i+=BS_DIM) {
            for(int j=0; j<BS; j+=BS_DIM) {
                C[ti+i][tj+j] = mat[(k*BS+ti+i)*V+bidx*BS+tj+j];
            }
        }
        __syncthreads();
        floyd(C, A, C, ti, tj);
    }
    else {
        base_i = bidx*BS+ti;
        base_j = k*BS+tj;
        for(int i=0; i<BS; i+=BS_DIM) {
            for(int j=0; j<BS; j+=BS_DIM) {
                C[ti+i][tj+j] = mat[(bidx*BS+ti+i)*V+k*BS+tj+j];
            }
        }
        __syncthreads();
        floyd(C, C, A, ti, tj);
    }

    mat[base_i*V+base_j] = C[ti][tj];
    mat[base_i*V+base_j+BS_DIM] = C[ti][tj+BS_DIM];
    mat[(base_i+BS_DIM)*V+base_j] = C[ti+BS_DIM][tj];
    mat[(base_i+BS_DIM)*V+base_j+BS_DIM] = C[ti+BS_DIM][tj+BS_DIM];
}
//Wij, Wik, Wkj
__global__ void floyd_warshall_phase3(int* mat, int V, int k) {
    __shared__ int A[BS][BS];
    __shared__ int B[BS][BS];
    // __shared__ int C[BS][BS];
    int bi = blockIdx.y;
    int bj = blockIdx.x;
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    if(bj==k || bi==k) return;

    int biti = bi*BS+ti;
    int bjtj = bj*BS+tj;
    int kti  = k*BS+ti;
    int ktj  = k*BS+tj;

    for(int i=0; i<BS; i+=BS_DIM) {
        for(int j=0; j<BS; j+=BS_DIM) {
            A[ti+i][tj+j] = mat[(biti+i)*V+ktj+j];
        }
    }
    for(int i=0; i<BS; i+=BS_DIM) {
        for(int j=0; j<BS; j+=BS_DIM) {
            B[ti+i][tj+j] = mat[(kti+i)*V+bjtj+j];
        }
    }
    __syncthreads();
    int i0, i1, i2, i3;
    i0 = mat[biti*V+bjtj];
    i1 = mat[biti*V+bjtj+BS_DIM];
    i2 = mat[(biti+BS_DIM)*V+bjtj];
    i3 = mat[(biti+BS_DIM)*V+bjtj+BS_DIM];

    for(int kk=0; kk<BS; kk++) {
        int sum0 = A[ti][kk] + B[kk][tj];
        int sum1 = A[ti][kk] + B[kk][tj+BS_DIM];
        int sum2 = A[ti+BS_DIM][kk] + B[kk][tj];
        int sum3 = A[ti+BS_DIM][kk] + B[kk][tj+BS_DIM];
        if(i0 > sum0) {
            i0 = sum0;
        }
        if(i1 > sum1) {
            i1 = sum1;
        }
        if(i2 > sum2) {
            i2 = sum2;
        }
        if(i3 > sum3) {
            i3 = sum3;
        }
    }

    mat[biti*V + bjtj]                 = i0;
    mat[biti*V + bjtj+BS_DIM]          = i1;
    mat[(biti+BS_DIM)*V + bjtj]        = i2;
    mat[(biti+BS_DIM)*V + bjtj+BS_DIM] = i3;
}

int main(int argc, char** argv)
{
    auto start_time = chrono::steady_clock::now();
    int numOfVertex;
    int numOfEdge;
    ifstream in;
    ofstream out;
    in.open(argv[1], ios::binary | ios::in);
    in.read((char*)&numOfVertex, sizeof(int));
    in.read((char*)&numOfEdge, sizeof(int));

    const int int_size = sizeof(int);
    // padding
    const int size = ((numOfVertex+BS-1)/BS)*BS;
    int* mat = new int[size*size];
    for(int i=0; i<size; i++) {
        for(int j=0; j<size; j++) {
            if(i>=numOfVertex || j>=numOfVertex) mat[i*size+j] = INF;
            else if(i==j) mat[i*size+j] = 0;
            else mat[i*size+j] = INF;
        }
    }
    // auto start_time = chrono::steady_clock::now();
    for(int i=0; i<numOfEdge; i++) {
        int src, dest;
        in.read((char*)&src, int_size);
        in.read((char*)&dest, int_size);
        in.read((char*)&mat[src*size+dest], int_size);
    }
    auto end_time = chrono::steady_clock::now();
    auto time_span = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "read time: " << time_span.count() << endl;

    size_t mat_size = sizeof(int) * size * size;
    int* device_mat;
    cudaMalloc(&device_mat, mat_size);

    start_time = chrono::steady_clock::now();
    cudaMemcpy(device_mat, mat, mat_size, cudaMemcpyHostToDevice);
    end_time = chrono::steady_clock::now();
    time_span = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "copy time: " << time_span.count() << endl;

    start_time = chrono::steady_clock::now();
    int numOfBlock = size/BS;
    dim3 block_dim(BS_DIM, BS_DIM, 1);
    dim3 p2_grid(numOfBlock, 2, 1);
    dim3 grid_dim(numOfBlock, numOfBlock, 1);

    for (int k=0; k<numOfBlock; k++) {
        floyd_warshall_phase1<<<1, block_dim>>>(device_mat, size, k);
        floyd_warshall_phase2<<<p2_grid, block_dim>>>(device_mat, size, k);
        floyd_warshall_phase3<<<grid_dim, block_dim>>>(device_mat, size, k);
    }
    cudaDeviceSynchronize();

    end_time = chrono::steady_clock::now();
    time_span = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "computation time: " << time_span.count() << endl;

    start_time = chrono::steady_clock::now();
    cudaMemcpy(mat, device_mat, mat_size, cudaMemcpyDeviceToHost);
    end_time = chrono::steady_clock::now();
    time_span = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "copy time: " << time_span.count() << endl;

    cudaFree(device_mat);

    cudaError_t err = cudaGetLastError();

    if( err != cudaSuccess ) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    }
    // print(mat, numOfVertex);
    
    start_time = chrono::steady_clock::now();
    out.open(argv[2], ios::binary | ios::out);
    for(int i=0; i<numOfVertex; i++) {
        out.write((char*)&mat[i*size], numOfVertex*sizeof(int));
    }
    end_time = chrono::steady_clock::now();
    time_span = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "write time: " << time_span.count() << endl;


    return 0;
}