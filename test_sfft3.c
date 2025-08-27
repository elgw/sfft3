#include <assert.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <getopt.h>

#include <omp.h>
#include <fftw3.h>
#include "sfft3.h"

typedef int64_t i64;
typedef float f32;

static float timespec_diff(struct timespec* end, struct timespec * start)
{
    double elapsed = (end->tv_sec - start->tv_sec);
    elapsed += (end->tv_nsec - start->tv_nsec) / 1000000000.0;
    return elapsed;
}

fftwf_plan gen_fftwf_plan_r2c(i64 M, i64 N, i64 P,
                              int FFTW_PLANNER_FLAGS)
{
    fftwf_plan plan_r2c_inplace = NULL;

    fftwf_complex * T = fftwf_malloc( (1+M/2)*2*N*P*sizeof(fftwf_complex));
    memset(T, 0, (1+M/2)*2*N*P*sizeof(fftwf_complex));
    plan_r2c_inplace = fftwf_plan_dft_r2c_3d(P, N, M,
                                             (float*) T, (fftwf_complex*) T,
                                             FFTW_PLANNER_FLAGS);
    free(T);

    assert(plan_r2c_inplace != NULL);
    return plan_r2c_inplace;
}

fftwf_plan gen_fftwf_plan_c2r(i64 M, i64 N, i64 P,
                              int FFTW_PLANNER_FLAGS)
{
    fftwf_plan plan_c2r_inplace = NULL;

    fftwf_complex * T = fftwf_malloc( (1+M/2)*2*N*P*sizeof(fftwf_complex));
    memset(T, 0, (1+M/2)*2*N*P*sizeof(fftwf_complex));
    plan_c2r_inplace = fftwf_plan_dft_c2r_3d(P, N, M,
                                             T, (float*) T,
                                             FFTW_PLANNER_FLAGS);
    free(T);

    assert(plan_c2r_inplace != NULL);
    return plan_c2r_inplace;
}

static float * fft_pad(const float * restrict X,
                       const i64 M,
                       const i64 N,
                       const i64 P)
{
    f32 * Y = fftwf_malloc(2*(1+M/2)*N*P*sizeof(float));
    const i64 nchunk = N*P;
    const i64 chunk_size = M*sizeof(float);
    const i64 pM = (1+M/2)*2; // padded number of floats per first dimension

#pragma omp parallel for schedule(static)
    for(i64 c = 0; c < nchunk; c++)
    {
        memcpy(Y+c*pM,
               X + c*M,
               chunk_size);
    }

    return Y;
}

static float * fft_unpad(const float * restrict X,
                         const size_t M,
                         const size_t N,
                         const size_t P)
{
    f32 * Y = fftwf_malloc(M*N*P*sizeof(float));
    i64 nchunk = N*P;

    i64 pM = (1+M/2)*2; // padded number of floats per first dimension

#pragma omp parallel for
    for(i64 c = 0; c < nchunk; c++)
    {
        memmove(Y+c*M,
                X+c*pM,
                M*sizeof(float));
    }
    return Y;
}

double complex_rel_err(fftwf_complex A, fftwf_complex B)
{
    double n = sqrt( pow(A[0], 2) + pow(A[1], 2));
    double e = sqrt( pow(A[0]-B[0], 2) + pow(A[1]-B[1], 2));
    return  e/n;
}

double float_rel_err(float A, float B)
{
    float d = A;
    fabs(B) > fabs(A) ?  d = B : 0;
    return fabs(A-B)/fabs(d);
}

static void show_usage(void)
{
    printf("A small test program that benchmarks sfft3 vs fftw3 for 3D problems\n"
           "sfft3 still use 1D fft functions from fftw3\n"
           "The program will warm up for --warmup seconds, then it will\n"
           "run benchmarks for --benchmark seconds\n"
           "Timings will be stored to M_N_P_results.csv\n"
           "\n");

    printf("Usage:\n");
    printf("--verbose v\n\t"
           "set verbosity level\n");
    printf("--help\n\t"
           "show this message\n");
    printf("--warmup s\n\t"
           "specify the warmup time in seconds\n");
    printf("--benchmark s\n\t"
           "specify the benchmark time in seconds\n");
    printf("--m M\n\t"
           "Set the number of elements in the first dimension\n");
    printf("--n N\n\t"
           "Set the number of elements in the 2nd dimension\n");
    printf("--p P\n\t"
           "Set the number of elements in the 3rd dimension\n");
    printf("--estimate\n\t"
           "Sets FFTW_ESTIMATE for planning (default FFTW_MEASURE)\n");
    printf("--patient\n\t"
           "Sets FFTW_PATIENT for planning (default FFTW_MEASURE)\n");
    printf("nofftw\n\t"
           "Disable FFTW3 benchmark\n");
    printf("nosfft\n\t"
           "Disable SFFTW3 benchmark\n");
    return;
}

int main(int argc, char ** argv)
{
    i64 M = 512;
    i64 N = 256;
    i64 P = 128;
    float warmup_s = 10;
    float benchmark_s = 5;
    int verbose = 1;
    int threads = omp_get_max_threads();
    int flags = 0;
    int use_fftw = 1;
    int use_sfft = 1;

    static struct option long_options[] = {
        {"verbose", required_argument, 0,  'v' },
        {"warmup",  required_argument, 0,  'w' },
        {"benchmark", required_argument, 0, 'b'},
        {"m",       required_argument, 0,  'm' },
        {"n",       required_argument, 0,  'n' },
        {"p",       required_argument, 0,  'p' },
        {"help",    no_argument,       0,  'h' },
        {"estimate", no_argument,       0,  'E' },
        {"patient", no_argument,       0,  'P' },
        {"nofftw", no_argument,       0,  'F' },
        {"nosfft", no_argument,       0,  'S' },
        {0,         0,                 0,  0 }
    };

    while(1)
    {
        int option_index = 0;
        int c = getopt_long(argc, argv, "v:w:b:m:n:p:hEPFS", long_options, &option_index);
        if(c == -1)
            break;

        switch(c){
        case 'v':
            verbose = atoi(optarg);
            break;
        case 'w':
            warmup_s = atof(optarg);
            break;
        case 'b':
            benchmark_s = atof(optarg);
            break;
        case 'm':
            M = atoi(optarg);
            break;
        case 'n':
            N = atoi(optarg);
            break;
        case 'p':
            P = atoi(optarg);
            break;
        case 'E':
            flags = flags | FFTW_ESTIMATE;
            break;
        case 'P':
            flags = flags | FFTW_PATIENT;
            break;
        case 'F':
            use_fftw = 0;
            break;
        case 'S':
            use_sfft = 0;
            break;
        default:
            show_usage();
            exit(EXIT_FAILURE);
        }
    }

    char * outfile = malloc(1024);
    printf("Built for L1=%d kB, L2=%d kB\n", SFFT3_L1, SFFT3_L2);
    sprintf(outfile, "results_%ld_%ld_%ld_%dth.csv", M, N, P, threads);
    if(verbose > 0)
    {
        printf("Writing to %s\n", outfile);
    }
    if(verbose > 1)
    {
        printf("omp_get_max_threads -> %d (control with OMP_NUM_THREADS)\n", omp_get_max_threads());
        printf("Planner flags: %d\n", flags);
    }
    float * X0 = malloc(M*N*P*sizeof(float));
    assert(X0 != NULL);

    /* Generate pseudo random data in [1, 2]*/
#pragma omp parallel
    {
        unsigned int seed = 42*omp_get_thread_num();

#pragma omp for
        for(i64 kk = 0; kk < M*N*P; kk++)
        {
            X0[kk] = 1.0 + rand_r(&seed) / (float) RAND_MAX;
        }
    }

    FILE * fid = fopen(outfile, "w");
    free(outfile);
    fprintf(fid, "method, time\n");



    /* Prepare data by padding */
    float * X = fft_pad(X0, M, N, P);
    i64 cM = (1+M/2);

    if(verbose > 0)
    {
        printf("Planning\n");
    }
    /* Create FFTW3 plans */
    fftwf_plan plan_r2c_inplace = NULL;
    fftwf_plan plan_c2r_inplace = NULL;
    if(use_fftw)
    {
        fftwf_plan_with_nthreads(omp_get_max_threads());
        plan_r2c_inplace = gen_fftwf_plan_r2c(M, N, P,
                                              flags);

        plan_c2r_inplace = gen_fftwf_plan_c2r(M, N, P,
                                              flags);
    }
    /* Create sfft3 plans (with extra workspace) */
    sfft_plan * plan = NULL;
    if(use_sfft)
    {
        plan = sfft3_create_plan(M, N, P,
                                 omp_get_max_threads(), // threads
                                 0, // verbose
                                 flags); // for 1D transforms
    }

    if(use_fftw && use_sfft)
    {
        if(verbose > 0)
        {
            printf("Checking results\n");
        }
        {
            float * X1 = malloc( cM*N*P*sizeof(fftwf_complex));
            memcpy(X1, X, cM*N*P*sizeof(fftwf_complex) );
            fftwf_execute_dft_r2c(plan_r2c_inplace, X, (fftwf_complex *) X);
            sfft3_execute_dft_r2c(plan, X1);
            // Equal in complex domain
            fftwf_complex * cX = (fftwf_complex *) X;
            fftwf_complex * cX1 = (fftwf_complex *) X1;
            i64 err = 0;
            printf(" -> Validating the r2c implementation\n");
            for(i64 kk = 0; kk < cM*N*P; kk++)
            {
                if( complex_rel_err(cX[kk],  cX1[kk]) > 1e-2)
                {
                    err++;
                    if(err < 5)
                    {
                        printf("%ld %f %f -- %f %f (%e)\n",
                               kk,
                               cX[kk][0], cX[kk][1],
                               cX1[kk][0], cX1[kk][1],
                               complex_rel_err(cX[kk], cX1[kk]));
                    }
                }
            }
            printf(" -> Validating the c2r implementation\n");
            // Equal to input when transformed back
            fftwf_execute_dft_c2r(plan_c2r_inplace, cX, X);
            sfft3_execute_dft_c2r(plan, cX1);
            float * XX = (float*) cX;
            float * XX1 = (float*) cX1;
            float * uXX = fft_unpad(XX, M, N, P);
            float * uXX1 = fft_unpad(XX1, M, N, P);
            free(X1);
            err = 0;
            float scale = (float) (M * N * P);
#pragma omp parallel for
            for(i64 kk = 0; kk < M*N*P; kk++)
            {
                uXX[kk] /= scale;
                uXX1[kk] /= scale;
            }

            for(i64 kk = 0; kk < M*N*P; kk++)
            {
                if(float_rel_err(X0[kk], uXX1[kk]) > 0.1)
                {
                    err ++;
                    if(err < 5)
                    {
                        printf("X0[%ld]=%f, sfft->=%f, fftw3->%f\n", kk, X0[kk], uXX1[kk], uXX[kk]);
                    }
                }
            }
            if(err > 0)
            {
                printf("%ld errors\n", err);
            }
            free(uXX);
            free(uXX1);
        }
    }

    free(X0);

    float dt = 0;
    struct timespec tstart, t0, t1;
    clock_gettime(CLOCK_REALTIME, &tstart);
    if(verbose > 0 && warmup_s > 0)
    {
        printf("warming up for %.0f s\n", warmup_s);
    }
    while(dt < warmup_s)
    {
        if(use_fftw)
        {
            fftwf_execute_dft_r2c(plan_r2c_inplace, X, (fftwf_complex *) X);
            fftwf_execute_dft_c2r(plan_r2c_inplace, (fftwf_complex *) X, X);
        }
        if(use_sfft)
        {
            sfft3_execute_dft_r2c(plan, X);
            sfft3_execute_dft_c2r(plan, (fftwf_complex *) X);
        }
        clock_gettime(CLOCK_REALTIME, &t1);
        dt = timespec_diff(&t1, &tstart);
    }

    dt = 0;
    clock_gettime(CLOCK_REALTIME, &tstart);
    if(verbose > 0 && benchmark_s > 0)
    {
        printf("benchmarking for %.0f s\n", benchmark_s);
    }

    double t_fftw = 0;
    double t_ifftw = 0;
    double t_sfft = 0;
    double t_isfft = 0;
    i64 iter = 0;
    while(dt < benchmark_s)
    {
        if(use_fftw)
        {
            clock_gettime(CLOCK_REALTIME, &t0);
            fftwf_execute_dft_r2c(plan_r2c_inplace, X, (fftwf_complex *) X);
            clock_gettime(CLOCK_REALTIME, &t1);
            dt = timespec_diff(&t1, &t0);
            t_fftw += dt;
            fprintf(fid, "fftwf_r2c, %.5f\n", dt);
            if(verbose > 2)
            {
                printf("fftwf_r2c, %.5f\n", dt);
            }

            clock_gettime(CLOCK_REALTIME, &t0);
            fftwf_execute_dft_c2r(plan_c2r_inplace, (fftwf_complex *) X,  X);
            clock_gettime(CLOCK_REALTIME, &t1);
            dt = timespec_diff(&t1, &t0);
            t_ifftw += dt;
            fprintf(fid, "fftwf_c2r, %.5f\n", dt);
            if(verbose > 2)
            {
                printf("fftwf_c2r, %.5f\n", dt);
            }
        }

        if(use_sfft)
        {
            clock_gettime(CLOCK_REALTIME, &t0);
            sfft3_execute_dft_r2c(plan, X);
            clock_gettime(CLOCK_REALTIME, &t1);
            dt = timespec_diff(&t1, &t0);
            t_sfft += dt;
            fprintf(fid, "sfft3_r2c, %.5f\n", dt);
            if(verbose > 2)
            {
                printf("sfft3_r2c, %.5f\n", dt);
            }

            clock_gettime(CLOCK_REALTIME, &t0);
            sfft3_execute_dft_c2r(plan, (fftwf_complex *) X);
            clock_gettime(CLOCK_REALTIME, &t1);
            dt = timespec_diff(&t1, &t0);
            t_isfft += dt;
            fprintf(fid, "sfft3_c2r, %.5f\n", dt);
            if(verbose > 2)
            {
                printf("sfft3_c2r, %.5f\n", dt);
            }
        }
        dt = timespec_diff(&t1, &tstart);
        iter++;
    }

    if(verbose > 0)
    {
        printf("Average times:\n");
        if(use_fftw)
        {
            printf("FFTW3:  %.3e\n", (t_fftw+t_ifftw) / (double) iter / 2.0);
        }
        if(use_sfft)
        {
            printf("SFFT3:  %.3e\n", (t_sfft+t_isfft) / (double) iter / 2.0);
        }
    }

    if(verbose > 1)
    {
        printf("FFTW3 forward:  %.3e\n", t_fftw / (double) iter);
        printf("FFTW3 backward: %.3e\n", t_ifftw/ (double) iter);
        printf("FFTW3 backward: %.3e\n", t_ifftw/ (double) iter);
        printf("SFFT3 forward:  %.3e\n", t_sfft / (double) iter);
        printf("SFFT3 backward: %.3e\n", t_isfft/ (double) iter);
    }

    free(X);
    if(use_fftw)
    {
        fftwf_destroy_plan(plan_r2c_inplace);
        fftwf_destroy_plan(plan_c2r_inplace);
    }
    if(use_sfft)
    {
        sfft3_destroy_plan(plan);
    }

    fclose(fid);
    fftwf_cleanup();

    return EXIT_SUCCESS;
}
