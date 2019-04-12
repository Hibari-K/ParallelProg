#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<sys/stat.h>
#include<sys/time.h>
#include<math.h>
#include<mpi.h>
#include<immintrin.h>
#include "nbody.h"


// for debug
void print_vec(__m512d a){
    
    double *x = (double*)&a;

    int i;
    for(i=7; i>=0; i--) printf("%lf ", x[i]);
    puts("");
}

double get_time();


int main(int argc, char **argv){

    double *xi __attribute__((aligned(64)));
    double *yi __attribute__((aligned(64)));
    double *zi __attribute__((aligned(64)));
    double *xj __attribute__((aligned(64)));
    double *yj __attribute__((aligned(64)));
    double *zj __attribute__((aligned(64)));
    double *m  __attribute__((aligned(64))); 
    double ax[8] __attribute__((aligned(64)));
    double ay[8] __attribute__((aligned(64)));
    double az[8] __attribute__((aligned(64)));
    double *vx, *vy, *vz;
    double r;

    //initialize
    xi = _mm_malloc(sizeof(double)*N, 64);
    yi = _mm_malloc(sizeof(double)*N, 64);
    zi = _mm_malloc(sizeof(double)*N, 64);
    
    if(!(xi && yi && zi)){
        puts("malloc error");
        exit(-1);
    }

    xj = _mm_malloc(sizeof(double)*N, 64);
    yj = _mm_malloc(sizeof(double)*N, 64);
    zj = _mm_malloc(sizeof(double)*N, 64);

    if(!(xj && yj && zj)){
        puts("malloc error");
        _mm_free(xi);
        _mm_free(yi);
        _mm_free(zi);
        exit(-1);
    }

    vx = _mm_malloc(sizeof(double)*N, 64);
    vy = _mm_malloc(sizeof(double)*N, 64);
    vz = _mm_malloc(sizeof(double)*N, 64);
    m = _mm_malloc(sizeof(double)*N, 64);

    if(!(vx && vy && vz && m)){
        puts("malloc error");
        _mm_free(xi);
        _mm_free(yi);
        _mm_free(zi);
        _mm_free(xj);
        _mm_free(yj);
        _mm_free(zj);
        exit(-1);
    }

    __m512d zxi, zyi, zzi;
    __m512d zvx, zvy, zvz;
    __m512d zax, zay, zaz;

    __m512d zG  = _mm512_set1_pd(G);
    __m512d zdt = _mm512_set1_pd(dt);
    __m512d one = _mm512_set1_pd(1.0);
    __m512d zero = _mm512_setzero_pd();


    double start, mid, end;
    int i,j, step;
    double rx, ry, rz;

    int  rank;
    MPI_Status status;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size, split, myrank;
    
    if(rank == 0){
        
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
  
        for(i = myrank*split; i < (myrank+1)*split; i += NUM_PD){

            zax = _mm512_setzero_pd();
            zay = _mm512_setzero_pd();
            zaz = _mm512_setzero_pd();

            zxi = _mm512_load_pd(xi + i);
            zyi = _mm512_load_pd(yi + i);
            zzi = _mm512_load_pd(zi + i);

            #pragma omp parallel for private(j,zxi,zyi,zzi,zax,zay,zaz) reduction(+:ax,ay,az)
            for(j = 0; j < N; j+=4){

                __m512d ztmp, zsqr, zr;
                __mmask8 k1;
                __m512d zxj, zyj, zzj;
                __m512d zrx, zry, zrz;
                __m512d zm;
                __m512d denom, numer, fixed;

                zxj = _mm512_set1_pd(xi[j]);
                zyj = _mm512_set1_pd(yi[j]);
                zzj = _mm512_set1_pd(zi[j]);
                
                zrx = _mm512_sub_pd(zxi, zxj);
                zry = _mm512_sub_pd(zyi, zyj);
                zrz = _mm512_sub_pd(zzi, zzj);

                ztmp = _mm512_add_pd(_mm512_mul_pd(zrx, zrx),
                                     _mm512_fmadd_pd(zry, zry, _mm512_mul_pd(zrz, zrz)));

                zm = _mm512_set1_pd(m[j]);

                if(unlikely(i <= j && j < i+8)){
                    k1 = _mm512_cmpeq_pd_mask(zero, ztmp);
                    ztmp = _mm512_mask_mov_pd(ztmp, k1, one);
                    zm = _mm512_mask_mov_pd(zm, k1, zero);
                }
                
                zsqr = _mm512_rsqrt28_pd(ztmp);
                zr   = _mm512_rcp28_pd(ztmp);

                denom = _mm512_mul_pd(zr, zsqr);    // r * r * r
                numer = _mm512_mul_pd(zG, zm);      // G * m[j]
                fixed = _mm512_mul_pd(denom, numer);

                zax = _mm512_fmadd_pd(fixed, zrx, zax);
                zay = _mm512_fmadd_pd(fixed, zry, zay);
                zaz = _mm512_fmadd_pd(fixed, zrz, zaz);

                // 2
                zxj = _mm512_set1_pd(xi[j+1]);
                zyj = _mm512_set1_pd(yi[j+1]);
                zzj = _mm512_set1_pd(zi[j+1]);
                
                zrx = _mm512_sub_pd(zxi, zxj);
                zry = _mm512_sub_pd(zyi, zyj);
                zrz = _mm512_sub_pd(zzi, zzj);

                ztmp = _mm512_add_pd(_mm512_mul_pd(zrx, zrx),
                                     _mm512_fmadd_pd(zry, zry, _mm512_mul_pd(zrz, zrz)));

                zm = _mm512_set1_pd(m[j+1]);

                if(unlikely(i <= j && j < i+8)){
                    k1 = _mm512_cmpeq_pd_mask(zero, ztmp);
                    ztmp = _mm512_mask_mov_pd(ztmp, k1, one);
                    zm = _mm512_mask_mov_pd(zm, k1, zero);
                }
                
                zsqr = _mm512_rsqrt28_pd(ztmp);
                zr   = _mm512_rcp28_pd(ztmp);

                denom = _mm512_mul_pd(zr, zsqr);    // r * r * r
                numer = _mm512_mul_pd(zG, zm);      // G * m[j]
                fixed = _mm512_mul_pd(denom, numer);

                zax = _mm512_fmadd_pd(fixed, zrx, zax);
                zay = _mm512_fmadd_pd(fixed, zry, zay);
                zaz = _mm512_fmadd_pd(fixed, zrz, zaz);

                // 3
                zxj = _mm512_set1_pd(xi[j+2]);
                zyj = _mm512_set1_pd(yi[j+2]);
                zzj = _mm512_set1_pd(zi[j+2]);
                
                zrx = _mm512_sub_pd(zxi, zxj);
                zry = _mm512_sub_pd(zyi, zyj);
                zrz = _mm512_sub_pd(zzi, zzj);

                ztmp = _mm512_add_pd(_mm512_mul_pd(zrx, zrx),
                                     _mm512_fmadd_pd(zry, zry, _mm512_mul_pd(zrz, zrz)));

                zm = _mm512_set1_pd(m[j+2]);

                if(unlikely(i <= j && j < i+8)){
                    k1 = _mm512_cmpeq_pd_mask(zero, ztmp);
                    ztmp = _mm512_mask_mov_pd(ztmp, k1, one);
                    zm = _mm512_mask_mov_pd(zm, k1, zero);
                }
                
                zsqr = _mm512_rsqrt28_pd(ztmp);
                zr   = _mm512_rcp28_pd(ztmp);

                denom = _mm512_mul_pd(zr, zsqr);    // r * r * r
                numer = _mm512_mul_pd(zG, zm);      // G * m[j]
                fixed = _mm512_mul_pd(denom, numer);

                zax = _mm512_fmadd_pd(fixed, zrx, zax);
                zay = _mm512_fmadd_pd(fixed, zry, zay);
                zaz = _mm512_fmadd_pd(fixed, zrz, zaz);

                // 4
                zxj = _mm512_set1_pd(xi[j+3]);
                zyj = _mm512_set1_pd(yi[j+3]);
                zzj = _mm512_set1_pd(zi[j+3]);
                
                zrx = _mm512_sub_pd(zxi, zxj);
                zry = _mm512_sub_pd(zyi, zyj);
                zrz = _mm512_sub_pd(zzi, zzj);

                ztmp = _mm512_add_pd(_mm512_mul_pd(zrx, zrx),
                                     _mm512_fmadd_pd(zry, zry, _mm512_mul_pd(zrz, zrz)));

                zm = _mm512_set1_pd(m[j+3]);

                if(unlikely(i <= j && j < i+8)){
                    k1 = _mm512_cmpeq_pd_mask(zero, ztmp);
                    ztmp = _mm512_mask_mov_pd(ztmp, k1, one);
                    zm = _mm512_mask_mov_pd(zm, k1, zero);
                }
                
                zsqr = _mm512_rsqrt28_pd(ztmp);
                zr   = _mm512_rcp28_pd(ztmp);

                denom = _mm512_mul_pd(zr, zsqr);    // r * r * r
                numer = _mm512_mul_pd(zG, zm);      // G * m[j]
                fixed = _mm512_mul_pd(denom, numer);

                zax = _mm512_fmadd_pd(fixed, zrx, zax);
                zay = _mm512_fmadd_pd(fixed, zry, zay);
                zaz = _mm512_fmadd_pd(fixed, zrz, zaz);

                _mm512_store_pd(ax, zax);
                _mm512_store_pd(ay, zay);
                _mm512_store_pd(az, zaz);
            }
      
            zvx = _mm512_load_pd(vx + i);
            zvy = _mm512_load_pd(vy + i);
            zvz = _mm512_load_pd(vz + i);
            zax = _mm512_load_pd(ax);
            zay = _mm512_load_pd(ay);
            zaz = _mm512_load_pd(az);
         
            zvx = _mm512_fmadd_pd(zax, zdt, zvx);
            zvy = _mm512_fmadd_pd(zay, zdt, zvy);
            zvz = _mm512_fmadd_pd(zaz, zdt, zvz);

            zxi = _mm512_fmadd_pd(zvx, zdt, zxi);
            zyi = _mm512_fmadd_pd(zvy, zdt, zyi);
            zzi = _mm512_fmadd_pd(zvz, zdt, zzi);

            _mm512_store_pd(vx + i, zvx);
            _mm512_store_pd(vy + i, zvy);
            _mm512_store_pd(vz + i, zvz);
            _mm512_store_pd(xj + i, zxi);
            _mm512_store_pd(yj + i, zyi);
            _mm512_store_pd(zj + i, zzi);
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
