#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <zlib.h>
#include <png.h>
#include <stdbool.h>
#include <stdint.h>
#include <jpeglib.h>
#include <assert.h>
#include <string.h>
#include <chrono>
#include <ctime>

#define KERNEL_BS 8
#define BS 8

typedef struct {
    uint8_t *data;   // raw data
    uint32_t width;
    uint32_t height;
    uint32_t channels;     // color channels
} ImageData;

void write_png(const char* filename, ImageData* imageData) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, imageData->width, imageData->height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[imageData->height];
    for (int i = 0; i < imageData->height; ++ i) {
        row_ptr[i] = imageData->data + i * imageData->width * imageData->channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int read_png(const char* filename, ImageData* imageData) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, &imageData->width, &imageData->height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[imageData->height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    imageData->channels = (int) png_get_channels(png_ptr, info_ptr);

    if (((imageData->data) = (unsigned char *) malloc(rowbytes * imageData->height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < imageData->height;  ++i)
        row_pointers[i] = (imageData->data) + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

__constant__ double cos_lookup[8][8] = {
    {0.707107,  0.980785,  0.923880,  0.831470,  0.707107,  0.555570,  0.382683,  0.195090},
    {0.707107,  0.831470,  0.382683, -0.195090, -0.707107, -0.980785, -0.923880, -0.555570},
    {0.707107,  0.555570, -0.382684, -0.980785, -0.707107,  0.195090,  0.923880,  0.831469},
    {0.707107,  0.195090, -0.923880, -0.555570,  0.707107,  0.831469, -0.382684, -0.980785},
    {0.707107, -0.195090, -0.923880,  0.555570,  0.707107, -0.831470, -0.382683,  0.980785},
    {0.707107, -0.555570, -0.382683,  0.980785, -0.707107, -0.195090,  0.923879, -0.831470},
    {0.707107, -0.831470,  0.382684,  0.195090, -0.707107,  0.980785, -0.923880,  0.555571},
    {0.707107, -0.980785,  0.923880, -0.831470,  0.707107, -0.555571,  0.382684, -0.195092}
};

__constant__ float qy[8][8] = {{16, 11, 10, 16, 24, 40, 51, 61},
                  {12, 12, 14, 19, 26, 58, 60, 55},
                  {14, 13, 16, 24, 40, 57, 69, 56},
                  {14, 17, 22, 29, 51, 87, 80, 82},
                  {18, 22, 37, 56, 68, 109, 103, 77},
                  {24, 35, 55, 64, 81, 104, 113, 92},
                  {99, 64, 78, 87, 103, 121, 120, 101},
                  {72, 92, 95, 98, 112, 100, 103, 99}};
__constant__ float quv[8][8] = {{17, 18, 24, 47, 99, 99, 99, 99},
                  {18, 21, 26, 66, 99, 99, 99, 99},
                  {24, 26, 56, 99, 99, 99, 99, 99},
                  {47, 66, 99, 99, 99, 99, 99, 99},
                  {99, 99, 99, 99, 99, 99, 99, 99},
                  {99, 99, 99, 99, 99, 99, 99, 99},
                  {99, 99, 99, 99, 99, 99, 99, 99},
                  {99, 99, 99, 99, 99, 99, 99, 99}};

__device__ void rgb_to_yuv(float R[][KERNEL_BS], float G[][KERNEL_BS], float B[][KERNEL_BS], int i, int j) {
    float newR = 0.299   * R[i][j] + 0.587  * G[i][j] + 0.114  * B[i][j];
    float newG = -0.1687 * R[i][j] - 0.3313 * G[i][j] + 0.5    * B[i][j] + 128;
    float newB = 0.5     * R[i][j] - 0.4187 * G[i][j] - 0.0813 * B[i][j] + 128;
    R[i][j] = newR;
    G[i][j] = newG;
    B[i][j] = newB;
}

__device__ void dct(float A[][KERNEL_BS], int i, int j) {
    int x, y;
    float tmp = 0;
    // float u_cs, v_cs, Pi=3.1415927;
    for (x = 0; x < KERNEL_BS; x++) {
        for (y = 0; y < KERNEL_BS; y++) {
            tmp += 0.25 * A[y][x] * cos_lookup[x][i] * cos_lookup[y][j];
            // tmp += 0.25 * A[y][x];
        }
    }
    __syncthreads();
    A[j][i] = tmp;
}

__device__ void quantize_y(float A[][KERNEL_BS], int i, int j) {
    A[i][j] = round(A[i][j] / qy[i][j]);
}

__device__ void quantize_uv(float A[][KERNEL_BS], int i, int j) {
    A[i][j] = round(A[i][j] / quv[i][j]);
}

__device__ void yuv_to_rgb(float R[][KERNEL_BS], float G[][KERNEL_BS], float B[][KERNEL_BS], int i, int j) {
    float newR = R[i][j] + 1.402 * (B[i][j] - 128);
    float newG = R[i][j] - 0.34414 * (G[i][j] - 128) - 0.71414 * (B[i][j] - 128);
    float newB = R[i][j] + 1.772 * (G[i][j] - 128);
    R[i][j] = newR;
    G[i][j] = newG;
    B[i][j] = newB;
}

__device__ void inv_dct(float A[][KERNEL_BS], int i, int j) {
    int x, y;
    float tmp = 0;
    for (x = 0; x < KERNEL_BS; x++) {
        for (y = 0; y < KERNEL_BS; y++) {
            tmp += 0.25 * A[y][x] * cos_lookup[i%BS][x%BS] * cos_lookup[j%BS][y%BS];
            // tmp += 0.25 * A[y][x];
        }
    }
    __syncthreads();
    A[j][i] = tmp;
}

__device__ void dequantize_y(float A[][KERNEL_BS], int i, int j) {
    A[i][j] = round(A[i][j] * qy[i%BS][j%BS]);
}

__device__ void dequantize_uv(float A[][KERNEL_BS], int i, int j) {
    A[i][j] = round(A[i][j] * quv[i%BS][j%BS]);
}

__global__ void compress(float* r, float* g, float* b, int width) {

    // Start JPEG Compress
    
    __shared__ float R[KERNEL_BS][KERNEL_BS];
    __shared__ float G[KERNEL_BS][KERNEL_BS];
    __shared__ float B[KERNEL_BS][KERNEL_BS];
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    R[ti][tj] = r[width*(bidy*KERNEL_BS+ti) + bidx*KERNEL_BS+tj];
    G[ti][tj] = g[width*(bidy*KERNEL_BS+ti) + bidx*KERNEL_BS+tj];
    B[ti][tj] = b[width*(bidy*KERNEL_BS+ti) + bidx*KERNEL_BS+tj];
    rgb_to_yuv(R, G, B, ti, tj);
    __syncthreads();
    dct(R, ti, tj);
    dct(G, ti, tj);
    dct(B, ti, tj);
    quantize_y(R, ti, tj);
    quantize_uv(G, ti, tj);
    quantize_uv(B, ti, tj);
    // __syncthreads();

    dequantize_y(R, ti, tj);
    dequantize_uv(G, ti, tj);
    dequantize_uv(B, ti, tj);
    __syncthreads();

    inv_dct(R, ti, tj);
    inv_dct(G, ti, tj);
    inv_dct(B, ti, tj);
    __syncthreads();

    yuv_to_rgb(R, G, B, ti, tj);
    // __syncthreads();
    r[width*(bidy*KERNEL_BS+ti) + bidx*KERNEL_BS+tj] = R[ti][tj];
    g[width*(bidy*KERNEL_BS+ti) + bidx*KERNEL_BS+tj] = G[ti][tj];
    b[width*(bidy*KERNEL_BS+ti) + bidx*KERNEL_BS+tj] = B[ti][tj];
}

using namespace std;

void imageProcessing_BlockByBlock(ImageData* imageData) {
    
    const int padWidth = ((imageData->width+BS-1)/KERNEL_BS)*KERNEL_BS;
    const int padHeight = ((imageData->height+BS-1)/KERNEL_BS)*KERNEL_BS;

    cout << padWidth << ", " << padHeight << ", " << imageData->channels << endl;

    float *h_r = new float[padWidth * padHeight];
    float *h_g = new float[padWidth * padHeight];
    float *h_b = new float[padWidth * padHeight];

    // Copy data from imageData
    for (int i=0; i<padHeight; ++i) {
        for (int j=0; j<padWidth; ++j) {
            if(i<imageData->height && j<imageData->width) {
                h_r[i*imageData->width + j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 0];
                h_g[i*imageData->width + j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 1];
                h_b[i*imageData->width + j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 2];
            }
            else {
                h_r[i*imageData->width + j] = 0;
                h_g[i*imageData->width + j] = 0;
                h_b[i*imageData->width + j] = 0;
            }
        }
    }

    float* d_r;
    float* d_g;
    float* d_b;
    auto start_time = chrono::steady_clock::now();
    size_t size = sizeof(float) * padWidth * padHeight;
    cudaMalloc(&d_r, size);
    cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_g, size);
    cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b, size);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    auto end_time = chrono::steady_clock::now();
    auto time_span = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "copy time: " << time_span.count() << endl;

    int widthBlockNum = padWidth/KERNEL_BS;
    int heightBlockNum = padHeight/KERNEL_BS;

    dim3 block_dim(KERNEL_BS, KERNEL_BS, 1);
    dim3 grid_dim(widthBlockNum, heightBlockNum, 1);
    start_time = chrono::steady_clock::now();
    compress<<<grid_dim, block_dim>>>(d_r, d_g, d_b, padWidth);
    cudaDeviceSynchronize();
    end_time = chrono::steady_clock::now();
    time_span = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "computation time: " << time_span.count() << endl;


    start_time = chrono::steady_clock::now();
    cudaMemcpy(h_r, d_r, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, d_g, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    end_time = chrono::steady_clock::now();
    time_span = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "copy time: " << time_span.count() << endl;

    // // Copy Data back
    for (int i=0; i<imageData->height; ++i) {
        for (int j=0; j<imageData->width; ++j) {
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 0] = h_r[i*imageData->width + j];
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 1] = h_g[i*imageData->width + j];
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 2] = h_b[i*imageData->width + j];
        }
    }
}

int main(int argc, char** argv)
{
    assert(argc == 3);
    auto start_time = chrono::steady_clock::now();
    char* srcName = argv[1];
    char* dstName = argv[2];
    ImageData imageData;
    read_png(srcName, &imageData);
    auto end_time = chrono::steady_clock::now();
    auto time_span = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "read time: " << time_span.count() << endl;

    imageProcessing_BlockByBlock(&imageData);

    cudaError_t err = cudaGetLastError();

    if( err != cudaSuccess ) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    }
    start_time = chrono::steady_clock::now();
    write_png(dstName, &imageData);
    end_time = chrono::steady_clock::now();
    time_span = chrono::duration_cast<chrono::duration<double>>(end_time - start_time);
    cout << "write time: " << time_span.count() << endl;

}