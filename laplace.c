#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<unistd.h>
#include<math.h>

#define XSIZE   512
#define YSIZE   512
#define ERR 1.0e-6
#define PI  3.1415927

typedef struct matrix{
    double **u;
}Matrix;

double second();
void init(int x, int y, Matrix m);

int main(){

    double start, end;
    double err, diff;
    int i, j;
    Matrix uM;
    Matrix uuM;
    
    double *uptr  = malloc(XSIZE * YSIZE * sizeof(double));
    double *uuptr = malloc(XSIZE * YSIZE * sizeof(double));
    double **u  = malloc(YSIZE * sizeof(double));
    double **uu = malloc(YSIZE * sizeof(double));

    if(!(uptr && uuptr && u && uu)){
        puts("malloc error");
        exot(-1);
    }

    for(i=0; i<YSIZE; i++){
        u[i]  = uptr + (i * XSIZE);
        uu[i] = uuptr + (i * XSIZE);
    }

    uM.u = u;
    uuM.u = uu;

    init(XSIZE, YSIZE, uM);

    for(i=0; i<YSIZE; i++){
        for(j=0; j<XSIZE; j++){
            uu[i][j] = u[i][j];
        }
    }
   
/*
    // debug
    for(i=0; i<YSIZE; i++){
        for(j=0; j<XSIZE; j++){
            printf("%lf ", u[i][j]);
        }
        puts("");
    }
*/

    start = second();
#pragma omp parallel private(i, j, diff)
    do{

#pragma omp for
        for(i = 1; i < YSIZE - 1; i++)
            for(j = 1; j < XSIZE - 1; j++)
                uu[i][j] = u[i][j];

#pragma omp for
        for(i = 1; i < YSIZE - 1; i++)
            for(j = 1; j < XSIZE - 1; j++)
                u[i][j] = (  uu[i - 1][j]
                           + uu[i + 1][j]
                           + uu[i][j - 1] 
                           + uu[i][j + 1]) / 4.0;
#pragma omp single
        { err = 0.0; }
#pragma omp for reduction(+:err)
        for(i = 1; i < YSIZE - 1; i++)
            for(j = 1; j < XSIZE - 1; j++){
                diff = uu[i][j] - u[i][j];
                err += diff * diff;
            }
    } while(err > ERR);

    end = second();

/*
    puts("");
    for(i=0; i<XSIZE; i++){
        for(j=0; j<YSIZE; j++){
            printf("%lf ", u[i][j]);
        }
        puts("");
    }
*/
    printf("time = %f [s]\n", end - start);


    free(u);
    free(uu);
    free(uptr);
    free(uuptr);
}

double second(){
    struct timeval tm;
    double t;

    gettimeofday(&tm,NULL);
    t = (double) (tm.tv_sec) + ((double) (tm.tv_usec))/1.0e6;
    return t;
}


void init(int x, int y, Matrix m){
    int i, j;
    double **f = m.u;

    for(i = 0; i < y; i++){
        for(j = 0; j < x; j++){
            f[i][j] = sin((double)j / XSIZE * PI) + cos((double)i / YSIZE * PI);
        }
    }
}
