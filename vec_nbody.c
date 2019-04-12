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
                __m512d ztmp2, zsqr2, zr2;
                __m512d ztmp3, zsqr3, zr3;
                __m512d ztmp4, zsqr4, zr4;
                __mmask8 k1, k2, k3, k4;
                __m512d zxj, zyj, zzj;
                __m512d zxj2, zyj2, zzj2;
                __m512d zxj3, zyj3, zzj3;
                __m512d zxj4, zyj4, zzj4;
                __m512d zrx, zry, zrz;
                __m512d zrx2, zry2, zrz2;
                __m512d zrx3, zry3, zrz3;
                __m512d zrx4, zry4, zrz4;
                __m512d zm, zm2, zm3, zm4;
                __m512d denom, numer, fixed;
                __m512d denom2, numer2, fixed2;
                __m512d denom3, numer3, fixed3;
                __m512d denom4, numer4, fixed4;
                __m512d tmp11, tmp21, tmp31;
                __m512d tmp12, tmp22, tmp32;
                __m512d tmp13, tmp23, tmp33;
                __m512d tmp14, tmp24, tmp34;

                
                if(unlikely(i <= j && j < i+8)){

                    zxj = _mm512_set1_pd(xi[j]);
                    zyj = _mm512_set1_pd(yi[j]);
                    zzj = _mm512_set1_pd(zi[j]);
                    
                    zxj2 = _mm512_set1_pd(xi[j+1]);
                    zyj2 = _mm512_set1_pd(yi[j+1]);
                    zzj2 = _mm512_set1_pd(zi[j+1]);
                    
                    zxj3 = _mm512_set1_pd(xi[j+2]);
                    zyj3 = _mm512_set1_pd(yi[j+2]);
                    zzj3 = _mm512_set1_pd(zi[j+2]);
                    
                    zxj4 = _mm512_set1_pd(xi[j+3]);
                    zyj4 = _mm512_set1_pd(yi[j+3]);
                    zzj4 = _mm512_set1_pd(zi[j+3]);
                    
                    
                    zrx = _mm512_sub_pd(zxi, zxj);
                    zry = _mm512_sub_pd(zyi, zyj);
                    zrz = _mm512_sub_pd(zzi, zzj);

                    zrx2 = _mm512_sub_pd(zxi, zxj2);
                    zry2 = _mm512_sub_pd(zyi, zyj2);
                    zrz2 = _mm512_sub_pd(zzi, zzj2);

                    zrx3 = _mm512_sub_pd(zxi, zxj3);
                    zry3 = _mm512_sub_pd(zyi, zyj3);
                    zrz3 = _mm512_sub_pd(zzi, zzj3);

                    zrx4 = _mm512_sub_pd(zxi, zxj4);
                    zry4 = _mm512_sub_pd(zyi, zyj4);
                    zrz4 = _mm512_sub_pd(zzi, zzj4);

                    
                    tmp11 = _mm512_mul_pd(zrx, zrx);
                    tmp21 = _mm512_mul_pd(zry, zry);
                    tmp31 = _mm512_mul_pd(zrz, zrz);
                    
                    tmp12 = _mm512_mul_pd(zrx2, zrx2);
                    tmp22 = _mm512_mul_pd(zry2, zry2);
                    tmp32 = _mm512_mul_pd(zrz2, zrz2);
                    
                    tmp13 = _mm512_mul_pd(zrx3, zrx3);
                    tmp23 = _mm512_mul_pd(zry3, zry3);
                    tmp33 = _mm512_mul_pd(zrz3, zrz3);
                    
                    tmp14 = _mm512_mul_pd(zrx4, zrx4);
                    tmp24 = _mm512_mul_pd(zry4, zry4);
                    tmp34 = _mm512_mul_pd(zrz4, zrz4);
                    
                    
                    tmp11 = _mm512_add_pd(tmp11, tmp21);
                    tmp12 = _mm512_add_pd(tmp12, tmp22);
                    tmp13 = _mm512_add_pd(tmp13, tmp23);
                    tmp14 = _mm512_add_pd(tmp14, tmp24);
                    
                    ztmp  = _mm512_add_pd(tmp11, tmp31);
                    ztmp2 = _mm512_add_pd(tmp12, tmp32);
                    ztmp3 = _mm512_add_pd(tmp13, tmp33);
                    ztmp4 = _mm512_add_pd(tmp14, tmp34);
                    
                    
                    
                    //ztmp = _mm512_add_pd(_mm512_mul_pd(zrx, zrx),
                    //                     _mm512_fmadd_pd(zry, zry, _mm512_mul_pd(zrz, zrz)));

                    
                    // if  i <= j < i+8
                    k1 = _mm512_cmpeq_pd_mask(zero, ztmp);
                    k2 = _mm512_cmpeq_pd_mask(zero, ztmp2);
                    ztmp  = _mm512_mask_mov_pd(ztmp, k1, one);
                    ztmp2 = _mm512_mask_mov_pd(ztmp2, k2, one);
                    zm =  _mm512_mask_mov_pd(zm, k1, zero);
                    zm2 = _mm512_mask_mov_pd(zm2, k2, zero);
                    
                    k3 = _mm512_cmpeq_pd_mask(zero, ztmp3);
                    k4 = _mm512_cmpeq_pd_mask(zero, ztmp4);
                    ztmp3  = _mm512_mask_mov_pd(ztmp3, k3, one);
                    ztmp4 = _mm512_mask_mov_pd(ztmp4, k4, one);
                    zm3 = _mm512_mask_mov_pd(zm3, k3, zero);
                    zm4 = _mm512_mask_mov_pd(zm4, k4, zero);
                    // endif
                    
                    zsqr = _mm512_rsqrt28_pd(ztmp);
                    zsqr2 = _mm512_rsqrt28_pd(ztmp2);
                    zsqr3 = _mm512_rsqrt28_pd(ztmp3);
                    zsqr4 = _mm512_rsqrt28_pd(ztmp4);
                    
                    zr   = _mm512_rcp28_pd(ztmp);
                    zr2  = _mm512_rcp28_pd(ztmp2);
                    zr3  = _mm512_rcp28_pd(ztmp3);
                    zr4  = _mm512_rcp28_pd(ztmp4);

                    
                    zm  = _mm512_set1_pd(m[j]);
                    zm2 = _mm512_set1_pd(m[j+1]);
                    zm3 = _mm512_set1_pd(m[j+2]);
                    zm4 = _mm512_set1_pd(m[j+3]);
                    

                    denom = _mm512_mul_pd(zr, zsqr);    // r * r * r
                    numer = _mm512_mul_pd(zG, zm);      // G * m[j]
                    fixed = _mm512_mul_pd(denom, numer);

                    denom2 = _mm512_mul_pd(zr2, zsqr2);    // r * r * r
                    numer2 = _mm512_mul_pd(zG, zm2);      // G * m[j]
                    fixed2 = _mm512_mul_pd(denom2, numer2);

                    denom3 = _mm512_mul_pd(zr3, zsqr3);    // r * r * r
                    numer3 = _mm512_mul_pd(zG, zm3);      // G * m[j]
                    fixed3 = _mm512_mul_pd(denom3, numer3);

                    denom4 = _mm512_mul_pd(zr4, zsqr4);    // r * r * r
                    numer4 = _mm512_mul_pd(zG, zm4);      // G * m[j]
                    fixed4 = _mm512_mul_pd(denom4, numer4);


                    tmp11 = _mm512_mul_pd(fixed, zrx);
                    tmp21 = _mm512_mul_pd(fixed, zry);
                    tmp31 = _mm512_mul_pd(fixed, zrz);
                    
                    tmp12 = _mm512_mul_pd(fixed2, zrx2);
                    tmp22 = _mm512_mul_pd(fixed2, zry2);
                    tmp32 = _mm512_mul_pd(fixed2, zrz2);
                    
                    tmp13 = _mm512_mul_pd(fixed3, zrx3);
                    tmp23 = _mm512_mul_pd(fixed3, zry3);
                    tmp33 = _mm512_mul_pd(fixed3, zrz3);
                    
                    tmp14 = _mm512_mul_pd(fixed4, zrx4);
                    tmp24 = _mm512_mul_pd(fixed4, zry4);
                    tmp34 = _mm512_mul_pd(fixed4, zrz4);
                    

                    
                    zax = _mm512_add_pd(tmp11, zax);
                    zax = _mm512_add_pd(tmp12, zax);
                    zax = _mm512_add_pd(tmp13, zax);
                    zax = _mm512_add_pd(tmp14, zax);
                    
                    zay = _mm512_add_pd(tmp21, zay);
                    zay = _mm512_add_pd(tmp22, zay);
                    zay = _mm512_add_pd(tmp23, zay);
                    zay = _mm512_add_pd(tmp24, zay);
                    
                    zaz = _mm512_add_pd(tmp31, zaz);
                    zaz = _mm512_add_pd(tmp32, zaz);
                    zaz = _mm512_add_pd(tmp33, zaz);
                    zaz = _mm512_add_pd(tmp34, zaz);
                    
                    _mm512_store_pd(ax, zax);
                    _mm512_store_pd(ay, zay);
                    _mm512_store_pd(az, zaz);
                
                }
                else{

                    zxj = _mm512_set1_pd(xi[j]);
                    zyj = _mm512_set1_pd(yi[j]);
                    zzj = _mm512_set1_pd(zi[j]);
                    
                    zxj2 = _mm512_set1_pd(xi[j+1]);
                    zyj2 = _mm512_set1_pd(yi[j+1]);
                    zzj2 = _mm512_set1_pd(zi[j+1]);
                    
                    zxj3 = _mm512_set1_pd(xi[j+2]);
                    zyj3 = _mm512_set1_pd(yi[j+2]);
                    zzj3 = _mm512_set1_pd(zi[j+2]);
                    
                    zxj4 = _mm512_set1_pd(xi[j+3]);
                    zyj4 = _mm512_set1_pd(yi[j+3]);
                    zzj4 = _mm512_set1_pd(zi[j+3]);
                    
                    
                    zrx = _mm512_sub_pd(zxi, zxj);
                    zry = _mm512_sub_pd(zyi, zyj);
                    zrz = _mm512_sub_pd(zzi, zzj);

                    zrx2 = _mm512_sub_pd(zxi, zxj2);
                    zry2 = _mm512_sub_pd(zyi, zyj2);
                    zrz2 = _mm512_sub_pd(zzi, zzj2);

                    zrx3 = _mm512_sub_pd(zxi, zxj3);
                    zry3 = _mm512_sub_pd(zyi, zyj3);
                    zrz3 = _mm512_sub_pd(zzi, zzj3);

                    zrx4 = _mm512_sub_pd(zxi, zxj4);
                    zry4 = _mm512_sub_pd(zyi, zyj4);
                    zrz4 = _mm512_sub_pd(zzi, zzj4);

                    
                    tmp11 = _mm512_mul_pd(zrx, zrx);
                    tmp21 = _mm512_mul_pd(zry, zry);
                    tmp31 = _mm512_mul_pd(zrz, zrz);
                    
                    tmp12 = _mm512_mul_pd(zrx2, zrx2);
                    tmp22 = _mm512_mul_pd(zry2, zry2);
                    tmp32 = _mm512_mul_pd(zrz2, zrz2);
                    
                    tmp13 = _mm512_mul_pd(zrx3, zrx3);
                    tmp23 = _mm512_mul_pd(zry3, zry3);
                    tmp33 = _mm512_mul_pd(zrz3, zrz3);
                    
                    tmp14 = _mm512_mul_pd(zrx4, zrx4);
                    tmp24 = _mm512_mul_pd(zry4, zry4);
                    tmp34 = _mm512_mul_pd(zrz4, zrz4);
                    
                    
                    tmp11 = _mm512_add_pd(tmp11, tmp21);
                    tmp12 = _mm512_add_pd(tmp12, tmp22);
                    tmp13 = _mm512_add_pd(tmp13, tmp23);
                    tmp14 = _mm512_add_pd(tmp14, tmp24);
                    
                    ztmp  = _mm512_add_pd(tmp11, tmp31);
                    ztmp2 = _mm512_add_pd(tmp12, tmp32);
                    ztmp3 = _mm512_add_pd(tmp13, tmp33);
                    ztmp4 = _mm512_add_pd(tmp14, tmp34);
                    
                    
                    //ztmp = _mm512_add_pd(_mm512_mul_pd(zrx, zrx),
                    //                     _mm512_fmadd_pd(zry, zry, _mm512_mul_pd(zrz, zrz)));

                    zsqr = _mm512_rsqrt28_pd(ztmp);
                    zsqr2 = _mm512_rsqrt28_pd(ztmp2);
                    zsqr3 = _mm512_rsqrt28_pd(ztmp3);
                    zsqr4 = _mm512_rsqrt28_pd(ztmp4);
                    
                    zr   = _mm512_rcp28_pd(ztmp);
                    zr2  = _mm512_rcp28_pd(ztmp2);
                    zr3  = _mm512_rcp28_pd(ztmp3);
                    zr4  = _mm512_rcp28_pd(ztmp4);

                    
                    zm  = _mm512_set1_pd(m[j]);
                    zm2 = _mm512_set1_pd(m[j+1]);
                    zm3 = _mm512_set1_pd(m[j+2]);
                    zm4 = _mm512_set1_pd(m[j+3]);
                    

                    denom = _mm512_mul_pd(zr, zsqr);    // r * r * r
                    numer = _mm512_mul_pd(zG, zm);      // G * m[j]
                    fixed = _mm512_mul_pd(denom, numer);

                    denom2 = _mm512_mul_pd(zr2, zsqr2);    // r * r * r
                    numer2 = _mm512_mul_pd(zG, zm2);      // G * m[j]
                    fixed2 = _mm512_mul_pd(denom2, numer2);

                    denom3 = _mm512_mul_pd(zr3, zsqr3);    // r * r * r
                    numer3 = _mm512_mul_pd(zG, zm3);      // G * m[j]
                    fixed3 = _mm512_mul_pd(denom3, numer3);

                    denom4 = _mm512_mul_pd(zr4, zsqr4);    // r * r * r
                    numer4 = _mm512_mul_pd(zG, zm4);      // G * m[j]
                    fixed4 = _mm512_mul_pd(denom4, numer4);


                    tmp11 = _mm512_mul_pd(fixed, zrx);
                    tmp21 = _mm512_mul_pd(fixed, zry);
                    tmp31 = _mm512_mul_pd(fixed, zrz);
                    
                    tmp12 = _mm512_mul_pd(fixed2, zrx2);
                    tmp22 = _mm512_mul_pd(fixed2, zry2);
                    tmp32 = _mm512_mul_pd(fixed2, zrz2);
                    
                    tmp13 = _mm512_mul_pd(fixed3, zrx3);
                    tmp23 = _mm512_mul_pd(fixed3, zry3);
                    tmp33 = _mm512_mul_pd(fixed3, zrz3);
                    
                    tmp14 = _mm512_mul_pd(fixed4, zrx4);
                    tmp24 = _mm512_mul_pd(fixed4, zry4);
                    tmp34 = _mm512_mul_pd(fixed4, zrz4);
                    

                    
                    zax = _mm512_add_pd(tmp11, zax);
                    zax = _mm512_add_pd(tmp12, zax);
                    zax = _mm512_add_pd(tmp13, zax);
                    zax = _mm512_add_pd(tmp14, zax);
                    
                    zay = _mm512_add_pd(tmp21, zay);
                    zay = _mm512_add_pd(tmp22, zay);
                    zay = _mm512_add_pd(tmp23, zay);
                    zay = _mm512_add_pd(tmp24, zay);
                    
                    zaz = _mm512_add_pd(tmp31, zaz);
                    zaz = _mm512_add_pd(tmp32, zaz);
                    zaz = _mm512_add_pd(tmp33, zaz);
                    zaz = _mm512_add_pd(tmp34, zaz);
                    

                    _mm512_store_pd(ax, zax);
                    _mm512_store_pd(ay, zay);
                    _mm512_store_pd(az, zaz);
                }

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
