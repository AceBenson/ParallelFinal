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

__device__ void rgb_to_yuv(float R[][BS], float G[][BS], float B[][BS], int i, int j) {
    R[i][j] = 0.299   * R[i][j] + 0.587  * G[i][j] + 0.114  * B[i][j];
    G[i][j] = -0.1687 * R[i][j] - 0.3313 * G[i][j] + 0.5    * B[i][j] + 128;
    B[i][j] = 0.5     * R[i][j] - 0.4187 * G[i][j] - 0.0813 * B[i][j] + 128;
}

__global__ void compress(float* r, float* g, float* b, int width) {

    // Start JPEG Compress
    // int roundX = imageData->width/8;
    // int roundY = imageData->height/8;
    
    __shared__ float R[BS][BS];
    __shared__ float G[BS][BS];
    __shared__ float B[BS][BS];
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int ti = threadIdx.y;
    int tj = threadIdx.x;
    R[ti][tj] = r[width*(bidy*BS+ti) + bidx*BS+tj];
    G[ti][tj] = g[width*(bidy*BS+ti) + bidx*BS+tj];
    B[ti][tj] = b[width*(bidy*BS+ti) + bidx*BS+tj];
    rgb_to_yuv(R, G, B, ti, tj);
    dct(r_out, r_in);
    dct(g_out, g_in);
    dct(b_out, b_in);

    r[width*(bidy*BS+ti) + bidx*BS+tj] = R[ti][tj];
    g[width*(bidy*BS+ti) + bidx*BS+tj] = G[ti][tj];
    b[width*(bidy*BS+ti) + bidx*BS+tj] = B[ti][tj];
    // for (int i=0; i<roundY; ++i) {
    //     for (int j=0; j<roundX; ++j) {
    //         // Deal with 8*8 matrix
    //         float r_in[8][8], g_in[8][8], b_in[8][8];
    //         float r_out[8][8], g_out[8][8], b_out[8][8];
    //         for (int x=0; x<8; ++x) {
    //             for (int y=0; y<8; ++y) {
    //                 r_in[x][y] = image_r[i*8+x][j*8+y];
    //                 g_in[x][y] = image_g[i*8+x][j*8+y];
    //                 b_in[x][y] = image_b[i*8+x][j*8+y];
    //             }
    //         }
    //         // printf("1. Convert to YUV space...\n");
    //         rgb_to_yuv(r_in, g_in, b_in, r_out, g_out, b_out);

    //         // printf("2. DCT...\n");
    //         dct(r_out, r_in);
    //         dct(g_out, g_in);
    //         dct(b_out, b_in);

    //         // printf("3. Quantization...\n");
    //         quantize(r_in, r_out, 0);
    //         quantize(g_in, g_out, 1);
    //         quantize(b_in, b_out, 1);

    //         if (i == 10 && j == 10) {
    //             printf("Check Compress Data:\n");
    //             for (int x=0; x<8; ++x) {
    //                 for (int y=0; y<8; ++y) {
    //                     printf("%6.1f ", r_out[x][y]);
    //                 }
    //                 printf("\t");
    //                 for (int y=0; y<8; ++y) {
    //                     printf("%6.1f ", g_out[x][y]);
    //                 }
    //                 printf("\t");
    //                 for (int y=0; y<8; ++y) {
    //                     printf("%6.1f ", b_out[x][y]);
    //                 }
    //                 printf("\n");
    //             }
    //         }

    //         // printf("4. Dequantization...\n");
    //         dequantize(r_out, r_in, 0);
    //         dequantize(g_out, g_in, 1);
    //         dequantize(b_out, b_in, 1);

    //         // printf("5. Inv DCT...\n");
    //         inv_dct(r_in, r_out);
    //         inv_dct(g_in, g_out);
    //         inv_dct(b_in, b_out);

    //         // printf("6. Convert to RGB space...\n");
    //         yuv_to_rgb(r_out, g_out, b_out, r_in, g_in, b_in);

    //         for (int x=0; x<8; ++x) {
    //             for (int y=0; y<8; ++y) {
    //                 image_r[i*8+x][j*8+y] = r_in[x][y];
    //                 image_g[i*8+x][j*8+y] = g_in[x][y];
    //                 image_b[i*8+x][j*8+y] = b_in[x][y];
    //             }
    //         }
    //     }
    // }
}

void imageProcessing_BlockByBlock(ImageData* imageData) {
    // Allocate 2D Array
    float *h_r = new float[imageData->width * imageData->height];
    float *h_g = new float[imageData->width * imageData->height];
    float *h_b = new float[imageData->width * imageData->height];

    // Copy data from imageData
    for (int i=0; i<imageData->height; ++i) {
        for (int j=0; j<imageData->width; ++j) {
            h_r[i*imageData->width + j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 0];
            h_g[i*imageData->width + j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 1];
            h_b[i*imageData->width + j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 2];
        }
    }

    float* d_r;
    float* d_g;
    float* d_b;

    size_t size = sizeof(float) * imageData->width * imageData->height;
    cudaMalloc(&d_r, size);
    cudaMemcpy(d_r, h_r, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_g, size);
    cudaMemcpy(d_g, h_g, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_b, size);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int widthBlockNum = imageData->width/BS;
    int heightBlockNum = imageData->width/BS;

    dim3 block_dim(8, 8, 1);
    dim3 grid_dim(widthBlockNum, heightBlockNum, 1);
    compress<<<grid_dim, block_dim>>>(d_r, d_g, d_b, imageData->width);

    cudaMemcpy(h_r, d_r, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, d_g, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b, d_b, size, cudaMemcpyDeviceToHost);

    // // Copy Data back
    for (int i=0; i<imageData->height; ++i) {
        for (int j=0; j<imageData->width; ++j) {
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 0] = h_r[i*imageData->width + j];
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 1] = h_g[i*imageData->width + j];
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 2] = h_b[i*imageData->width + j];
        }
    }
}

using namespace std;

int main()
{
    char srcName[30] = "large-candy.png";
    char dstName[30] = "out.png";
    ImageData imageData;
    read_png(srcName, &imageData);

    imageProcessing_BlockByBlock(&imageData);

    cudaError_t err = cudaGetLastError();

    if( err != cudaSuccess ) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    }

    write_png(dstName, &imageData);

}