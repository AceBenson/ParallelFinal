#include <stdio.h>
#include <stdlib.h>
#include <zlib.h>
#include <png.h>
#include <stdbool.h>
#include <stdint.h>
#include <jpeglib.h>
#include <assert.h>
#include <string.h>

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

void dataProcessing(ImageData imageData) {
    int size = imageData.width * imageData.height * imageData.channels;
    for (int i = 0; i < size; i++) {
        imageData.data[i] = ~imageData.data[i];
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

    // Perform Data Processing
    dataProcessing(imageData);

    // Write Image
    printf("Write: %s\n", dstName);
    if (endsWith(srcName, ".png"))
        write_png(dstName, &imageData);
    else if (endsWith(srcName, ".jpg"))
        write_jpeg(dstName, &imageData, &jerr);
    else {
        printf("Error: failed to open %s\n", dstName);
        return -1;
    }
}