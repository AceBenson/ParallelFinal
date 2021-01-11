#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <zlib.h>
#include <png.h>
#include <stdbool.h>
#include <stdint.h>
#include <jpeglib.h>
#include <assert.h>
#include <string.h>

// gcc -lm main.c -lpng -ljpeg

typedef struct {
    uint8_t *data;   // raw data
    uint32_t width;
    uint32_t height;
    uint32_t channels;     // color channels
} ImageData;

bool endsWith(const char *str, const char *suffix) {
    if (!str || !suffix)
        return 0;
    size_t lenstr = strlen(str);
    size_t lensuffix = strlen(suffix);
    if (lensuffix >  lenstr)
        return 0;
    return strncmp(str + lenstr - lensuffix, suffix, lensuffix) == 0;
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
bool read_jpeg(const char *srcfile, ImageData *jpegData, struct jpeg_error_mgr *jerr) {
    // 1. create JPEG decompression object
    struct jpeg_decompress_struct cinfo;
    jpeg_create_decompress(&cinfo);
    cinfo.err = jpeg_std_error(jerr);

    FILE *fp = fopen(srcfile, "rb");
    if (fp == NULL) {
        printf("Error: failed to open %s\n", srcfile);
        return false;
    }
    // 2. specify source data
    jpeg_stdio_src(&cinfo, fp);

    // 3. read JPEG header
    jpeg_read_header(&cinfo, TRUE);

    // 4. set parameters (omitted)
    // 5. start decompression
    jpeg_start_decompress(&cinfo);

    jpegData->width  = cinfo.image_width;
    jpegData->height = cinfo.image_height;
    jpegData->channels     = cinfo.num_components;

    // alloc_jpeg(jpegData);
    jpegData->data = (uint8_t*) malloc(sizeof(uint8_t) * jpegData->width * jpegData->height * jpegData->channels);

    // 6. scan lines (read line by line)
    uint8_t *row = jpegData->data;
    const uint32_t stride = jpegData->width * jpegData->channels;
    for (int y = 0; y < jpegData->height; y++) {
        jpeg_read_scanlines(&cinfo, &row, 1);
        row += stride;
    }

    // 7. finish decompression
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(fp);

    return true;
}
bool write_jpeg(const char *dstfile, const ImageData *jpegData, struct jpeg_error_mgr *jerr) {
    // 1. create JPEG compression object
    struct jpeg_compress_struct cinfo;
    jpeg_create_compress(&cinfo);
    cinfo.err = jpeg_std_error(jerr);

    FILE *fp = fopen(dstfile, "wb");
    if (fp == NULL) {
        printf("Error: failed to open %s\n", dstfile);
        return false;
    }
    // 2. specify destination data
    jpeg_stdio_dest(&cinfo, fp);

    // 3. set parameters
    cinfo.image_width      = jpegData->width;
    cinfo.image_height     = jpegData->height;
    cinfo.input_components = jpegData->channels;
    cinfo.in_color_space   = JCS_RGB;
    jpeg_set_defaults(&cinfo);

    // 4. start compression
    jpeg_start_compress(&cinfo, TRUE);

    // 5. scan lines
    uint8_t *row = jpegData->data;
    const uint32_t stride = jpegData->width * jpegData->channels;
    for (int y = 0; y < jpegData->height; y++) {
        jpeg_write_scanlines(&cinfo, &row, 1);
        row += stride;
    }

    // 6. finish compression
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(fp);

    return true;
}

// Input RGB
float q0[8][8] = {{16, 11, 10, 16, 24, 40, 51, 61},
                  {12, 12, 14, 19, 26, 58, 60, 55},
                  {14, 13, 16, 24, 40, 57, 69, 56},
                  {14, 17, 22, 29, 51, 87, 80, 82},
                  {18, 22, 37, 56, 68, 109, 103, 77},
                  {24, 35, 55, 64, 81, 104, 113, 92},
                  {99, 64, 78, 87, 103, 121, 120, 101},
                  {72, 92, 95, 98, 112, 100, 103, 99}};
float q1[8][8] = {{17, 18, 24, 47, 99, 99, 99, 99},
                  {18, 21, 26, 66, 99, 99, 99, 99},
                  {24, 26, 56, 99, 99, 99, 99, 99},
                  {47, 66, 99, 99, 99, 99, 99, 99},
                  {99, 99, 99, 99, 99, 99, 99, 99},
                  {99, 99, 99, 99, 99, 99, 99, 99},
                  {99, 99, 99, 99, 99, 99, 99, 99},
                  {99, 99, 99, 99, 99, 99, 99, 99}};

void rgb_to_yuv(float r[8][8], float g[8][8], float b[8][8], float y[8][8], float u[8][8], float v[8][8])
{
    int i, j;
    for (i = 0; i < 8; i++)
    {
        for (j = 0; j < 8; j++)
        {
            y[i][j] = 0.299 * r[i][j] + 0.587 * g[i][j] + 0.114 * b[i][j];
            u[i][j] = -0.1687 * r[i][j] - 0.3313 * g[i][j] + 0.5 * b[i][j] + 128;
            v[i][j] = 0.5 * r[i][j] - 0.4187 * g[i][j] - 0.0813 * b[i][j] + 128;
        }
    }
}
void yuv_to_rgb(float y[8][8], float u[8][8], float v[8][8], float r[8][8], float g[8][8], float b[8][8])
{
    int i, j;
    for (i = 0; i < 8; i++)
    {
        for (j = 0; j < 8; j++)
        {
            r[i][j] = y[i][j] + 1.402 * (v[i][j] - 128);
            g[i][j] = y[i][j] - 0.34414 * (u[i][j] - 128) - 0.71414 * (v[i][j] - 128);
            b[i][j] = y[i][j] + 1.772 * (u[i][j] - 128);
        }
    }
}

void dct(float pic_in[8][8], float enc_out[8][8])
{
    int u, v, x, y;
    float u_cs, v_cs, Pi;
    Pi = 3.1415927;
    for (u = 0; u < 8; u++)
    {
        for (v = 0; v < 8; v++)
        {
            enc_out[v][u] = 0;
            for (x = 0; x < 8; x++)
            {
                for (y = 0; y < 8; y++)
                {
                    u_cs = cos(((2 * x + 1) * u * Pi) / 16); //WHY?
                    if (u == 0)
                        u_cs = (1 / (sqrt(2)));
                    v_cs = cos(((2 * y + 1) * v * Pi) / 16);
                    if (v == 0)
                        v_cs = (1 / (sqrt(2)));
                    enc_out[v][u] += 0.25 * pic_in[y][x] * u_cs * v_cs;
                    // enc_out[v][u] += 0.25 * pic_in[y][x];
                }
            }
        }
    }
}
void inv_dct(float enc_in[8][8], float rec_out[8][8])
{
    int u, v, x, y;
    float u_cs, v_cs, Pi;
    Pi = 3.1415927;
    for (x = 0; x < 8; x++)
    {
        for (y = 0; y < 8; y++)
        {
            rec_out[y][x] = 0;
            for (u = 0; u < 8; u++)
            {
                for (v = 0; v < 8; v++)
                {
                    u_cs = cos(((2 * x + 1) * u * Pi) / 16);
                    if (u == 0)
                        u_cs = (1 / (sqrt(2)));
                    v_cs = cos(((2 * y + 1) * v * Pi) / 16);
                    if (v == 0)
                        v_cs = (1 / (sqrt(2)));
                    rec_out[y][x] += 0.25 * enc_in[v][u] * u_cs * v_cs;
                    // rec_out[y][x] += 0.25 * enc_in[v][u];
                }
            }
        }
    }
}

void quantize(float dctb[8][8], float qb[8][8], int n)
{
    int u, v;
    for (v = 0; v < 8; v++)
    {
        for (u = 0; u < 8; u++)
        {
            if (n == 0)
                qb[v][u] = round(dctb[v][u] / q0[v][u]);
            else
                qb[v][u] = round(dctb[v][u] / q1[v][u]);
        }
    }
}
void dequantize(float qb[8][8], float dctb[8][8], int n)
{
    int u, v;
    for (v = 0; v < 8; v++)
    {
        for (u = 0; u < 8; u++)
        {
            if (n == 0)
                dctb[v][u] = round(qb[v][u] * q0[v][u]);
            else
                dctb[v][u] = round(qb[v][u] * q1[v][u]);
        }
    }
}

void imageProcessing_BlockByBlock(ImageData* imageData) {
    // Allocate 2D Array
    float *image_r[imageData->height];
    float *image_g[imageData->height];
    float *image_b[imageData->height];
    for (int i=0; i<imageData->height; ++i) {
        image_r[i] = (float *) malloc (imageData->width * sizeof(float));
        image_g[i] = (float *) malloc (imageData->width * sizeof(float));
        image_b[i] = (float *) malloc (imageData->width * sizeof(float));
    }

    // Copy data from imageData
    for (int i=0; i<imageData->height; ++i) {
        for (int j=0; j<imageData->width; ++j) {
            image_r[i][j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 0];
            image_g[i][j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 1];
            image_b[i][j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 2];
        }
    }

    // Start JPEG Compress
    int roundX = imageData->width/8;
    int roundY = imageData->height/8;
    
    for (int i=0; i<roundY; ++i) {
        for (int j=0; j<roundX; ++j) {
            // Deal with 8*8 matrix
            float r_in[8][8], g_in[8][8], b_in[8][8];
            float r_out[8][8], g_out[8][8], b_out[8][8];
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    r_in[x][y] = image_r[i*8+x][j*8+y];
                    g_in[x][y] = image_g[i*8+x][j*8+y];
                    b_in[x][y] = image_b[i*8+x][j*8+y];
                }
            }
            // printf("1. Convert to YUV space...\n");
            rgb_to_yuv(r_in, g_in, b_in, r_out, g_out, b_out);

            // printf("2. DCT...\n");
            dct(r_out, r_in);
            dct(g_out, g_in);
            dct(b_out, b_in);

            // printf("3. Quantization...\n");
            quantize(r_in, r_out, 0);
            quantize(g_in, g_out, 1);
            quantize(b_in, b_out, 1);

            // if (i == 10 && j == 10) {
            //     printf("Check Compress Data:\n");
            //     for (int x=0; x<8; ++x) {
            //         for (int y=0; y<8; ++y) {
            //             printf("%6.1f ", r_out[x][y]);
            //         }
            //         printf("\t");
            //         for (int y=0; y<8; ++y) {
            //             printf("%6.1f ", g_out[x][y]);
            //         }
            //         printf("\t");
            //         for (int y=0; y<8; ++y) {
            //             printf("%6.1f ", b_out[x][y]);
            //         }
            //         printf("\n");
            //     }
            // }

            // printf("4. Dequantization...\n");
            dequantize(r_out, r_in, 0);
            dequantize(g_out, g_in, 1);
            dequantize(b_out, b_in, 1);

            // // printf("5. Inv DCT...\n");
            inv_dct(r_in, r_out);
            inv_dct(g_in, g_out);
            inv_dct(b_in, b_out);

            // // printf("6. Convert to RGB space...\n");
            yuv_to_rgb(r_out, g_out, b_out, r_in, g_in, b_in);

            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    image_r[i*8+x][j*8+y] = r_in[x][y];
                    image_g[i*8+x][j*8+y] = g_in[x][y];
                    image_b[i*8+x][j*8+y] = b_in[x][y];
                }
            }
        }
    }

    // Copy Data back
    for (int i=0; i<imageData->height; ++i) {
        for (int j=0; j<imageData->width; ++j) {
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 0] = image_r[i][j];
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 1] = image_g[i][j];
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 2] = image_b[i][j];
        }
    }
}
void imageProcessing_StepByStep(ImageData* imageData) {
    // Allocate 2D Array
    float *image_r[imageData->height];
    float *image_g[imageData->height];
    float *image_b[imageData->height];
    for (int i=0; i<imageData->height; ++i) {
        image_r[i] = (float *) malloc (imageData->width * sizeof(float));
        image_g[i] = (float *) malloc (imageData->width * sizeof(float));
        image_b[i] = (float *) malloc (imageData->width * sizeof(float));
    }

    // Copy data from imageData
    for (int i=0; i<imageData->height; ++i) {
        for (int j=0; j<imageData->width; ++j) {
            image_r[i][j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 0];
            image_g[i][j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 1];
            image_b[i][j] = imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 2];
        }
    }

    // Start JPEG Compress
    int roundX = imageData->width/8;
    int roundY = imageData->height/8;

    // 1. Convert to YUV space
    printf("1. Convert to YUV space...\n");
    for (int i=0; i<roundY; ++i) {
        for (int j=0; j<roundX; ++j) {
            // Deal with 8*8 matrix
            float r_in[8][8], g_in[8][8], b_in[8][8];
            float y_out[8][8], u_out[8][8], v_out[8][8];
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    r_in[x][y] = image_r[i*8+x][j*8+y];
                    g_in[x][y] = image_g[i*8+x][j*8+y];
                    b_in[x][y] = image_b[i*8+x][j*8+y];
                }
            }
            rgb_to_yuv(r_in, g_in, b_in, y_out, u_out, v_out);
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    image_r[i*8+x][j*8+y] = y_out[x][y];
                    image_g[i*8+x][j*8+y] = u_out[x][y];
                    image_b[i*8+x][j*8+y] = v_out[x][y];
                }
            }
        }
    }

    // 2. DCT
    printf("2. DCT...\n");
    for (int i=0; i<roundY; ++i) {
        for (int j=0; j<roundX; ++j) {
            // Deal with 8*8 matrix
            float y_in[8][8], u_in[8][8], v_in[8][8];
            float dct_y_out[8][8], dct_u_out[8][8], dct_v_out[8][8];
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    y_in[x][y] = image_r[i*8+x][j*8+y];
                    u_in[x][y] = image_g[i*8+x][j*8+y];
                    v_in[x][y] = image_b[i*8+x][j*8+y];
                    dct_y_out[x][y] = 0;
                    dct_u_out[x][y] = 0;
                    dct_v_out[x][y] = 0;
                }
            }
            dct(y_in, dct_y_out);
            dct(u_in, dct_u_out);
            dct(v_in, dct_v_out);
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    image_r[i*8+x][j*8+y] = dct_y_out[x][y];
                    image_g[i*8+x][j*8+y] = dct_u_out[x][y];
                    image_b[i*8+x][j*8+y] = dct_v_out[x][y];
                }
            }
        }
    }

    // 3. Quantization
    printf("3. Quantization...\n");
    for (int i=0; i<roundY; ++i) {
        for (int j=0; j<roundX; ++j) {
            // Deal with 8*8 matrix
            float y_in[8][8], u_in[8][8], v_in[8][8];
            float y_out[8][8], u_out[8][8], v_out[8][8];
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    y_in[x][y] = image_r[i*8+x][j*8+y];
                    u_in[x][y] = image_g[i*8+x][j*8+y];
                    v_in[x][y] = image_b[i*8+x][j*8+y];
                }
            }
            quantize(y_in, y_out, 0);
            quantize(u_in, u_out, 1);
            quantize(v_in, v_out, 1);
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    image_r[i*8+x][j*8+y] = y_out[x][y];
                    image_g[i*8+x][j*8+y] = u_out[x][y];
                    image_b[i*8+x][j*8+y] = v_out[x][y];
                }
            }
        }
    }

    // 4. Dequantization
    printf("4. Dequantization...\n");
    for (int i=0; i<roundY; ++i) {
        for (int j=0; j<roundX; ++j) {
            // Deal with 8*8 matrix
            float y_in[8][8], u_in[8][8], v_in[8][8];
            float y_out[8][8], u_out[8][8], v_out[8][8];
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    y_in[x][y] = image_r[i*8+x][j*8+y];
                    u_in[x][y] = image_g[i*8+x][j*8+y];
                    v_in[x][y] = image_b[i*8+x][j*8+y];
                }
            }
            dequantize(y_in, y_out, 0);
            dequantize(u_in, u_out, 1);
            dequantize(v_in, v_out, 1);
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    image_r[i*8+x][j*8+y] = y_out[x][y];
                    image_g[i*8+x][j*8+y] = u_out[x][y];
                    image_b[i*8+x][j*8+y] = v_out[x][y];
                }
            }
        }
    }

    // 5. Inv DCT
    printf("5. Inv DCT...\n");
    for (int i=0; i<roundY; ++i) {
        for (int j=0; j<roundX; ++j) {
            // Deal with 8*8 matrix
            float y_in[8][8], u_in[8][8], v_in[8][8];
            float dct_y_out[8][8], dct_u_out[8][8], dct_v_out[8][8];
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    y_in[x][y] = image_r[i*8+x][j*8+y];
                    u_in[x][y] = image_g[i*8+x][j*8+y];
                    v_in[x][y] = image_b[i*8+x][j*8+y];
                    dct_y_out[x][y] = 0;
                    dct_u_out[x][y] = 0;
                    dct_v_out[x][y] = 0;
                }
            }
            inv_dct(y_in, dct_y_out);
            inv_dct(u_in, dct_u_out);
            inv_dct(v_in, dct_v_out);
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    image_r[i*8+x][j*8+y] = dct_y_out[x][y];
                    image_g[i*8+x][j*8+y] = dct_u_out[x][y];
                    image_b[i*8+x][j*8+y] = dct_v_out[x][y];
                }
            }
        }
    }

    // 6. Convert to RGB space
    printf("6. Convert to RGB space...\n");
    for (int i=0; i<roundY; ++i) {
        for (int j=0; j<roundX; ++j) {
            // Deal with 8*8 matrix
            float y_in[8][8], u_in[8][8], v_in[8][8];
            float r_out[8][8], g_out[8][8], b_out[8][8];
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    y_in[x][y] = image_r[i*8+x][j*8+y];
                    u_in[x][y] = image_g[i*8+x][j*8+y];
                    v_in[x][y] = image_b[i*8+x][j*8+y];
                }
            }
            yuv_to_rgb(y_in, u_in, v_in, r_out, g_out, b_out);
            for (int x=0; x<8; ++x) {
                for (int y=0; y<8; ++y) {
                    image_r[i*8+x][j*8+y] = r_out[x][y];
                    image_g[i*8+x][j*8+y] = g_out[x][y];
                    image_b[i*8+x][j*8+y] = b_out[x][y];
                }
            }
        }
    }

    // Copy Data back
    for (int i=0; i<imageData->height; ++i) {
        for (int j=0; j<imageData->width; ++j) {
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 0] = image_r[i][j];
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 1] = image_g[i][j];
            imageData->data[(i*imageData->width)*imageData->channels + j*imageData->channels + 2] = image_b[i][j];
        }
    }
}

int main(int argc, char** argv) {
    assert(argc == 3);
    ImageData imageData;
    struct jpeg_error_mgr jerr;
    char* srcName = argv[1];
    char* dstName = argv[2];

    // Read Image
    printf("Read: %s\n", srcName);
    if (endsWith(srcName, ".png"))
        read_png(srcName, &imageData);
    else if (endsWith(srcName, ".jpg"))
        read_jpeg(srcName, &imageData, &jerr);
    else {
        printf("Error: failed to open %s\n", srcName);
        return -1;
    }

    printf("Src Width: %d\n", imageData.width);
    printf("Src Height: %d\n", imageData.height);

    imageProcessing_BlockByBlock(&imageData);

    // Write Image
    printf("Write: %s\n", dstName);
    if (endsWith(dstName, ".png"))
        write_png(dstName, &imageData);
    else if (endsWith(dstName, ".jpg"))
        write_jpeg(dstName, &imageData, &jerr);
    else {
        printf("Error: failed to open %s\n", dstName);
        return -1;
    }
}
