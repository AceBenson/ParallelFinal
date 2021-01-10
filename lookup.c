#include <math.h>
#include <stdio.h>

int main()
{
    double cos_lookup[8][8];
	int i, j;
	for (i=0; i<8; i++)
		for (j=0; j<8; j++)
		{
			cos_lookup[i][j] = cos( (2*i+1)*j*M_PI/16 );
            if(j==0) printf("%lf ", 1/sqrt(2));
            // else printf("%lf ", cos_lookup[i][j]);
		}
    return 0;
}