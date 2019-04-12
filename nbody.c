#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/stat.h>
#include<sys/time.h>
#include<math.h>
#include<mpi.h>
#include<immintrin.h>
#include "nbody.h"

double get_time();

int main(int argc, char **argv){

    double *xi __attribute__((aligned(64)));
    double *yi __attribute__((aligned(64)));
    double *zi __attribute__((aligned(64)));
    double *xj __attribute__((aligned(64)));
    double *yj __attribute__((aligned(64)));
    double *zj __attribute__((aligned(64)));
    double *m  __attribute__((aligned(64)));
    double ax, ay, az;
    double *vx, *vy, *vz;
    double r;

    //initialize
    xi = _mm_malloc(sizeof(double)*N, 64);
    yi = _mm_malloc(sizeof(double)*N, 64);
    zi = _mm_malloc(sizeof(double)*N, 64);

    xj = _mm_malloc(sizeof(double)*N, 64);
    yj = _mm_malloc(sizeof(double)*N, 64);
    zj = _mm_malloc(sizeof(double)*N, 64);

    vx = _mm_malloc(sizeof(double)*N, 64);
    vy = _mm_malloc(sizeof(double)*N, 64);
    vz = _mm_malloc(sizeof(double)*N, 64);

    m = _mm_malloc(sizeof(double)*N, 64);

    if(!(xi && yi && zi && xj && yj && zj && vx && vy && vz && m)){
        puts("malloc error");
        exit(-1);
    }

    double start, end;

    int i,j, step;
    double rx, ry, rz, M, vxx, vyy, vzz;

    int  rank;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size, split, myrank;

    if(rank==0){
        for(i=0; i<N; i++) xi[i] = (-1 * (i+1)) * rand() * 1.0E-6;
        for(i=0; i<N; i++) yi[i] = (-1 * (i+2)) * rand() * 1.0E-6;
        for(i=0; i<N; i++) zi[i] = (-1 * (i+1)) * rand() * 1.0E-6;
        for(i=0; i<N; i++) m[i] = rand() * 1.0E-9;
    }


    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(N%size == 0){
        split = N / size;
    }
    else{
        puts("Invalid process size");
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MPI_Bcast(xi, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    memcpy(xj, xi, N*sizeof(double));

    MPI_Bcast(yi, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    memcpy(yj, yi, N*sizeof(double));

    MPI_Bcast(zi, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    memcpy(zj, zi, N*sizeof(double));


    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();

    for(step = 0; step < 10; step++){

        if(rank==0){
            printf("step: %d\n", step);
        }

        for(i = myrank*split; i < (myrank+1)*split; i++){

            ax = ay = az = 0.0;

            #pragma omp parallel for private(j,M,rx,ry,rz,r,vxx,vyy,vzz) reduction(+:ax,ay,az)
            for(j = 0; j < N; j++){

                if(i != j){
                    
                    rx = xi[j] - xi[i];
                    ry = yi[j] - yi[i];
                    rz = zi[j] - zi[i];
                    
                    r = sqrt(rx * rx + ry * ry + rz * rz);
                    
                    M = m[j];

                    ax += G * ( M * rx ) / ( r * r * r );
                    ay += G * ( M * ry ) / ( r * r * r );
                    az += G * ( M * rz ) / ( r * r * r );	  
                }
            }

            vxx = vx[i] += ax*dt;
            vyy = vy[i] += ay*dt;
            vzz = vz[i] += az*dt;
            
            xj[i] += vxx*dt;
            yj[i] += vyy*dt;
            zj[i] += vzz*dt;      

        }

        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allgather(xj+(myrank*split), split, MPI_DOUBLE, xi, split, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(yj+(myrank*split), split, MPI_DOUBLE, yi, split, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(zj+(myrank*split), split, MPI_DOUBLE, zi, split, MPI_DOUBLE, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        memcpy(xj, xi, N*sizeof(double));
        memcpy(yj, yi, N*sizeof(double));
        memcpy(zj, zi, N*sizeof(double));


    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    if(rank == 0) printf("%lf [s]\n", end - start);

    _mm_free(m);
    _mm_free(xi);
    _mm_free(yi);
    _mm_free(zi);
    _mm_free(xj);
    _mm_free(yj);
    _mm_free(zj);
    _mm_free(vx);
    _mm_free(vy);
    _mm_free(vz);

    MPI_Finalize();
    return 0;

}




double get_time(){
    struct timeval t;
    gettimeofday(&t,NULL);
    return (double)t.tv_sec+(double)t.tv_usec*1e-6;
}
