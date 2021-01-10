#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>
#include <cmath>

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8
#define BS 4

/* Hint 7 */
// this variable is used by device
__device__ int mask[MASK_N][MASK_X][MASK_Y] = { 
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0}, 
     {  2,  8, 12,  8,  2}, 
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1}, 
     { -4, -8,  0,  8,  4}, 
     { -6,-12,  0, 12,  6}, 
     { -4, -8,  0,  8,  4}, 
     { -1, -2,  0,  2,  1}} 
};

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

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
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__global__ void Sobel(unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    int  v, u;
    int  R, G, B;
    int size = blockDim.y*blockDim.x;
    double val0, val1, val2;
    int adjustX, adjustY, xBound, yBound;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int y = blockIdx.y*blockDim.y + ty;
    int x = blockIdx.x*blockDim.x + tx;
    int tz = threadIdx.z;
    int idx = blockDim.y*blockDim.x*tz + ty*blockDim.x + tx;
    adjustX = 1;
    adjustY = 1;
    xBound = 2;
    yBound = 2;

    val2 = 0.0;
    val1 = 0.0;
    val0 = 0.0;

    for (v = -yBound; v < yBound + adjustY; ++v) {
        for (u = -xBound; u < xBound + adjustX; ++u) {
            if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
                R = s[channels * (width * (y+v) + (x+u)) + 2];
                G = s[channels * (width * (y+v) + (x+u)) + 1];
                B = s[channels * (width * (y+v) + (x+u)) + 0];
                val2 += R * mask[tz][u + xBound][v + yBound];
                val1 += G * mask[tz][u + xBound][v + yBound];
                val0 += B * mask[tz][u + xBound][v + yBound];
            }    
        }
    }

    __shared__ double totalR[BS*BS*2];
    __shared__ double totalG[BS*BS*2];
    __shared__ double totalB[BS*BS*2];
    totalR[idx] = val2 * val2;
    totalG[idx] = val1 * val1;
    totalB[idx] = val0 * val0;

    __syncthreads();

    if(idx<size) {
        totalR[idx] += totalR[idx+size];
        totalG[idx] += totalG[idx+size];
        totalB[idx] += totalB[idx+size];
        totalR[idx] = sqrt(totalR[idx]) / SCALE;
        totalG[idx] = sqrt(totalG[idx]) / SCALE;
        totalB[idx] = sqrt(totalB[idx]) / SCALE;
        const unsigned char cR = (totalR[idx] > 255.0) ? 255 : totalR[idx];
        const unsigned char cG = (totalG[idx] > 255.0) ? 255 : totalG[idx];
        const unsigned char cB = (totalB[idx] > 255.0) ? 255 : totalB[idx];

        t[channels * (width * y + x) + 2] = cR;
        t[channels * (width * y + x) + 1] = cG;
        t[channels * (width * y + x) + 0] = cB;
    }
}

int main(int argc, char** argv) {

    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char* host_s = NULL;
    read_png(argv[1], &host_s, &height, &width, &channels);
    int N = height * width * channels;
    unsigned char* host_t = (unsigned char*) malloc(N * sizeof(unsigned char));
    
    unsigned char* d_s, *d_t;
    /* Hint 1 */
    // cudaMalloc(...) for device src and device dst
    cudaMalloc((void**)&d_s, N * sizeof(unsigned char));
    cudaMalloc((void**)&d_t, N * sizeof(unsigned char));

    /* Hint 2 */
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(d_s, host_s, N * sizeof(unsigned char), cudaMemcpyHostToDevice);
    /* Hint 3 */
    // acclerate this function
    // sobel(host_s, host_t, height, width, channels);
    dim3 blockDim(width/BS, height/BS);
    dim3 threadDim(BS, BS, 2);
    Sobel<<<blockDim, threadDim, BS*BS*2*3*sizeof(double)>>>(d_s, d_t, height, width, channels);
    
    /* Hint 4 */
    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(host_t, d_t, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_s);
    cudaFree(d_t);
    write_png(argv[2], host_t, height, width, channels);

    return 0;
}
