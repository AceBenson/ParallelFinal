#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8
#define BX 32
#define BY 1

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

__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int y = blockIdx.y*blockDim.y + ty;
    int x = blockIdx.x*blockDim.x + tx;
    float val0, val1, val2;
    float val00, val11, val22;

    val2 = 0.0;
    val1 = 0.0;
    val0 = 0.0;
    val22 = 0.0;
    val11 = 0.0;
    val00 = 0.0;
    // printf("%d\n", y);

    // __shared__ unsigned char local[3*(BX+2*xBound)*(BY+2*yBound)];
    // int w = BX+2*xBound;
    // int istart = blockIdx.y*blockDim.y;
    // int iend = istart + BY + 2*yBound;
    // int jstart = blockIdx.x*blockDim.x;
    // int jend = jstart+w;
    // if(tx==0) {
    //     if(istart==0 && jstart==0) {
    //         printf("%d %d %d %d ", jstart, jstart+1, jend-xBound, jend-xBound+1);
    //     }
    //     for(int i=istart; i<istart+BY + 2*yBound; i++) {
    //         for(int j=jstart; j<jstart+xBound; j++) {
    //             if (bound_check(j-xBound, 0, width) && bound_check(i-yBound, 0, height)) {
    //                 local[channels * (w * (i-istart) + (j-jstart)) + 2] = s[channels * (width * (i-yBound) + (j-xBound)) + 2];
    //                 local[channels * (w * (i-istart) + (j-jstart)) + 1] = s[channels * (width * (i-yBound) + (j-xBound)) + 1];
    //                 local[channels * (w * (i-istart) + (j-jstart)) + 0] = s[channels * (width * (i-yBound) + (j-xBound)) + 0];
    //             }
    //         }
    //         // for(int j=jstart+xBound; j<jend-xBound; j++) {
    //         //     if (bound_check(j-xBound, 0, width) && bound_check(i-yBound, 0, height)) {
    //         //         local[channels * (w * (i-istart) + (j-jstart)) + 2] = s[channels * (width * (i-yBound) + (j-xBound)) + 2];
    //         //         local[channels * (w * (i-istart) + (j-jstart)) + 1] = s[channels * (width * (i-yBound) + (j-xBound)) + 1];
    //         //         local[channels * (w * (i-istart) + (j-jstart)) + 0] = s[channels * (width * (i-yBound) + (j-xBound)) + 0];
    //         //     }
    //         // }
    //         for(int j=jend-xBound; j<jend; j++) {
    //             if (bound_check(j-xBound, 0, width) && bound_check(i-yBound, 0, height)) {
    //                 local[channels * (w * (i-istart) + (j-jstart)) + 2] = s[channels * (width * (i-yBound) + (j-xBound)) + 2];
    //                 local[channels * (w * (i-istart) + (j-jstart)) + 1] = s[channels * (width * (i-yBound) + (j-xBound)) + 1];
    //                 local[channels * (w * (i-istart) + (j-jstart)) + 0] = s[channels * (width * (i-yBound) + (j-xBound)) + 0];
    //             }
    //         }
    //     }
    // }
    // int j = x+xBound;
    // if(istart==0 && jstart==0) {
    //     printf("%d ", j);
    // }
    // for(int i=istart; i<istart+BY + 2*yBound; i++) {
    //     if (bound_check(j-xBound, 0, width) && bound_check(i-yBound, 0, height)) {
    //         local[channels * (w * (i-istart) + (j-jstart)) + 2] = s[channels * (width * (i-yBound) + (j-xBound)) + 2];
    //         local[channels * (w * (i-istart) + (j-jstart)) + 1] = s[channels * (width * (i-yBound) + (j-xBound)) + 1];
    //         local[channels * (w * (i-istart) + (j-jstart)) + 0] = s[channels * (width * (i-yBound) + (j-xBound)) + 0];
    //     }
    // }
    // __syncthreads();
    /* Y and X axis of mask */
    for (int v = -yBound; v <= yBound; ++v) {
        for (int u = -xBound; u <= xBound; ++u) {
            if (bound_check(x + u, 0, width) && bound_check(y + v, 0, height)) {
                const unsigned char R = s[channels * (width * (y + v) + (x + u)) + 2];
                const unsigned char G = s[channels * (width * (y + v) + (x + u)) + 1];
                const unsigned char B = s[channels * (width * (y + v) + (x + u)) + 0];
                // printf("%d %d\n", ty+v+yBound, tx+u+xBound);
                // const unsigned char R = local[channels * (w * (ty + v+yBound) + (tx + u+xBound)) + 2];
                // const unsigned char G = local[channels * (w * (ty + v+yBound) + (tx + u+xBound)) + 1];
                // const unsigned char B = local[channels * (w * (ty + v+yBound) + (tx + u+xBound)) + 0];
                val2 += R * mask[0][u + xBound][v + yBound];
                val1 += G * mask[0][u + xBound][v + yBound];
                val0 += B * mask[0][u + xBound][v + yBound];
                val22 += R * mask[1][u + xBound][v + yBound];
                val11 += G * mask[1][u + xBound][v + yBound];
                val00 += B * mask[1][u + xBound][v + yBound];
            }
        }
    }

    float dR = val2 * val2 + val22 * val22;
    float dG = val1 * val1 + val11 * val11;
    float dB = val0 * val0 + val00 * val00;
    dR = sqrt(dR) / SCALE;
    dG = sqrt(dG) / SCALE;
    dB = sqrt(dB) / SCALE;
    const unsigned char cR = (dR > 255.0) ? 255 : dR;
    const unsigned char cG = (dG > 255.0) ? 255 : dG;
    const unsigned char cB = (dB > 255.0) ? 255 : dB;

    t[channels * (width * y + x) + 2] = cR;
    t[channels * (width * y + x) + 1] = cG;
    t[channels * (width * y + x) + 0] = cB;
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    unsigned char *dsrc, *ddst;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    // cudaMemcpy(...) copy source image to device (mask matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // decide to use how many blocks and threads
    dim3 num_blocks(width/BX, height/BY);
    dim3 num_threads(BX, BY, 1);

    // launch cuda kernel
    sobel <<<num_blocks, num_threads>>> (dsrc, ddst, height, width, channels);
    // cudaMemcpy(...) copy result image to host
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaError_t err = cudaGetLastError();

    if( err != cudaSuccess ) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err)); 
    }


    write_png(argv[2], dst, height, width, channels);
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    return 0;
}

