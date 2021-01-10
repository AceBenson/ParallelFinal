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

using namespace std;

int main()
{
    char srcName[30] = "read-write-test/src.png";
    ImageData imageData;
    read_png(srcName, &imageData);
    cout << imageData.width << ", " << imageData.height << ", " << imageData.channels << endl;
    
}