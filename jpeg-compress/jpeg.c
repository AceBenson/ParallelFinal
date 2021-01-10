#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Input RGB
float r_in[8][8] = {{224, 192, 160, 128, 96, 64, 32, 16},
                    {192, 160, 128, 96, 64, 32, 16, 8},
                    {160, 128, 96, 64, 32, 16, 8, 4},
                    {128, 96, 64, 32, 16, 8, 4, 2},
                    {96, 64, 32, 16, 8, 4, 2, 1},
                    {64, 32, 16, 8, 4, 2, 1, 0},
                    {32, 16, 8, 4, 2, 1, 0, 0},
                    {16, 8, 4, 2, 1, 0, 0, 0}};

float g_in[8][8] = {{16, 8, 4, 2, 1, 0, 0, 0},
                    {32, 16, 8, 4, 2, 1, 0, 0},
                    {64, 32, 16, 8, 4, 2, 1, 0},
                    {96, 64, 32, 16, 8, 4, 2, 1},
                    {128, 96, 64, 32, 16, 8, 4, 2},
                    {160, 128, 96, 64, 32, 16, 8, 4},
                    {192, 160, 128, 96, 64, 32, 16, 8},
                    {224, 192, 160, 128, 96, 64, 32, 16}};

float b_in[8][8] = {{224, 192, 160, 128, 96, 64, 32, 16},
                    {192, 160, 128, 96, 64, 32, 16, 8},
                    {160, 128, 96, 64, 32, 16, 8, 4},
                    {128, 96, 64, 32, 16, 8, 4, 2},
                    {96, 64, 32, 16, 8, 4, 2, 1},
                    {64, 32, 16, 8, 4, 2, 1, 0},
                    {32, 16, 8, 4, 2, 1, 0, 0},
                    {16, 8, 4, 2, 1, 0, 0, 0}};
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

float y[8][8], u[8][8], v[8][8];
float dct_y[8][8], dct_u[8][8], dct_v[8][8];
float quant_y[8][8], quant_u[8][8], quant_v[8][8];
float iquant_y[8][8], iquant_u[8][8], iquant_v[8][8];
float idct_y[8][8], idct_u[8][8], idct_v[8][8];
float iy[8][8], iu[8][8], iv[8][8];
float r_out[8][8], g_out[8][8], b_out[8][8];
float o_rll_y[96], o_rll_u[96], o_rll_v[96];
int cnt_y, cnt_u, cnt_v;

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

void rll(float quant_in[8][8], float rll_out[96], int *buffsize)
{
    int buffcnt, u, v, zcount;
    int waszero;
    buffcnt = 0;
    u = 0;
    v = 0;
    zcount = 0;
    waszero = 0; // initialize variables
    for (u = 0; u < 8; u++)
    {
        for (v = 0; v < 8; v++)
        {
            if ((int)quant_in[u][v] == 0)
            {
                if ((u == 7) && (v == 7))
                {
                    zcount++;
                    rll_out[buffcnt] = 0;
                    buffcnt++;
                    rll_out[buffcnt] = zcount;
                } //end-if (u,v)==7
                else
                {
                    if (zcount < 14)
                    {
                        zcount++;
                        rll_out[buffcnt] = 0;
                        rll_out[buffcnt + 1] = zcount;
                        waszero = 1;
                    } //end-if zcount<14
                    else
                    {
                        rll_out[buffcnt] = 0;
                        buffcnt++;
                        rll_out[buffcnt] = zcount;
                        buffcnt++;
                        zcount = 0;
                        waszero = 0;
                    } //end-else zcount<14
                }     //end-else (u,v)==7
            }         //end-if quant_in==0?
            else
            {
                zcount = 0;
                if (waszero == 1)
                {
                    buffcnt += 2;
                    rll_out[buffcnt] = quant_in[u][v];
                    buffcnt++;
                    waszero = 0;
                } //end-if waszero=1?
                else
                {
                    rll_out[buffcnt] = quant_in[u][v];
                    buffcnt++;
                } //end-else waszero==1?
            }     //end-else quant==0?
        }         //end-for v
    }             //end-for u
    *buffsize = buffcnt;
} //end rll
void trace_rll(float tracebuff[96], int maxcount)
{
    int i, j, k;
    printf("%d\n", maxcount);
    i = 0;
    j = 0;
    for (i = 0; i < (maxcount + 1); i++)
    {
        printf("i= %i. Buffer= %4.0f ||", i, tracebuff[i]);
        j++;
        if (j == 4)
        {
            printf("\n");
            j = 0;
        }
    }
    printf("\n");
}

void trace3(float data_to_dump1[8][8], float data_to_dump2[8][8], float data_to_dump3[8][8])
{
    int u, v;
    for (u = 0; u < 8; u++)
    {
        for (v = 0; v < 8; v++)
        {
            printf("%4.1f ", data_to_dump1[u][v]);
        }
        printf(" ");
        for (v = 0; v < 8; v++)
        {
            printf("%4.1f ", data_to_dump2[u][v]);
        }
        printf(" ");
        for (v = 0; v < 8; v++)
        {
            printf("%4.1f ", data_to_dump3[u][v]);
        }
        printf(" ");
        printf("\n");
    }
    printf("\n\n");
}

int main()
{
    printf("\n\n\n\n=========================================================================================================");
    printf("\n\n\n**** Starting JPEG-Compression **** \n\n");

    printf("- Processing: show_source-color...\n");
    trace3(r_in, g_in, b_in);

    rgb_to_yuv(r_in, g_in, b_in, y, u, v);
    printf("- Processing: RGBtoYUV-color...\n");
    trace3(y, u, v);

    dct(y, dct_y);
    dct(u, dct_u);
    dct(v, dct_v);
    printf("\n\n- Processing: dct...\n");
    trace3(dct_y, dct_u, dct_v);

    quantize(dct_y, quant_y, 0);
    quantize(dct_u, quant_u, 1);
    quantize(dct_v, quant_v, 1);
    printf("\n\n- Processing: quantize...\n");
    trace3(quant_y, quant_u, quant_v);

    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
        {
            iquant_y[i][j] = quant_y[i][j];
            iquant_u[i][j] = quant_u[i][j];
            iquant_v[i][j] = quant_v[i][j];
        }

    // rll(quant_y, o_rll_y, &cnt_y);
    // rll(quant_u, o_rll_u, &cnt_u);
    // rll(quant_v, o_rll_v, &cnt_v);
    // trace_rll(o_rll_y, cnt_y);
    // trace_rll(o_rll_u, cnt_u);
    // trace_rll(o_rll_v, cnt_v);

    dequantize(iquant_y, idct_y, 0);
    dequantize(iquant_u, idct_u, 1);
    dequantize(iquant_v, idct_v, 1);
    printf("\n\n- Processing: dequantize...\n");
    trace3(idct_y, idct_u, idct_v);

    inv_dct(idct_y, iy);
    inv_dct(idct_u, iu);
    inv_dct(idct_v, iv);
    printf("\n\n- Processing: inv_dct...\n");
    trace3(iy, iu, iv);

    yuv_to_rgb(iy, iu, iv, r_out, g_out, b_out);
    printf("\n\n- Processing: YUVtoRGB-color...\n");

    trace3(r_out, g_out, b_out);
    printf("**** ... finished! ****\n\n");
}
