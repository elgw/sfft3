#include "sfft3.h"

#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

typedef int64_t i64;
typedef float f32;

static int nomp(i64 n)
{
    if(n <= 0)
    {
        return 1;
    }
    i64 m = omp_get_max_threads();
    i64 r = (n-1)/m + 1;
    i64 t = (n-1)/r + 1;

    return t;
}

static i64 block_side(void)
{
    i64 bs = 16*floor(sqrt(SFFT3_L1/8)/16/2);
    bs < 16 ? bs = 16 : 0;
    return bs;
}

static void
fft_Y_plane_large(fftwf_complex * restrict _X,
                  fftwf_complex * restrict _Y,
                  i64 M, i64 N,
                  fftwf_plan restrict plan2,
                  fftwf_plan restrict plan2_aligned)
{
    double * X = (double*) _X;
    double * Y = (double*) _Y;
    i64 bs = block_side();

    i64 nl = bs * (SFFT3_L2 / (N*bs*8 ) / 4);
    nl < bs ? nl = bs : 0;

    double buff[bs*bs] __attribute__ ((aligned (64)));

    for(i64 M0 = 0; M0 < M; M0+=nl)
    {
        i64 M1 = M0 + nl;
        M1 > M ? M1 = M : 0;

        for(i64 m0 = M0; m0 < M1; m0+=bs)
        {
            for(i64 n0 = 0; n0 < N; n0+=bs)
            {

                i64 m1 = m0+bs;
                m1 > M1 ? m1 = M1 : 0;

                i64 n1 = n0+bs;
                n1 > N ? n1 = N : 0;

                // record
                for(i64 n = n0; n < n1; n++)
                {
                    for(i64 m = m0; m < m1; m++)
                    {
                        buff[m-m0 + bs*(n-n0)] = X[m + n*M];
                    }
                }

                // write back transposed
                for(i64 m = m0; m < m1; m++)
                {
                    for(i64 n = n0; n < n1; n++)
                    {
                        Y[n + (m-M0)*N] = buff[m-m0 + bs*(n-n0)];
                    }
                }
            }
        }

        for(i64 m = 0; m < M1-M0; m++)
        {
            fftwf_complex* L = (fftwf_complex*) Y + m*N;
            if( (ptrdiff_t) L % 16 == 0)
            {
                fftwf_execute_dft(plan2_aligned, L, L);
            } else {
                fftwf_execute_dft(plan2, L, L);
            }
        }

        for(i64 n0 = 0; n0 < N; n0+=bs)
        {
            for(i64 m0 = M0; m0 < M1; m0+=bs)
            {

                i64 m1 = m0+bs;
                m1 > M1 ? m1 = M1 : 0;

                i64 n1 = n0+bs;
                n1 > N ? n1 = N : 0;

                // record
                for(i64 m = m0; m < m1; m++)
                {
                    for(i64 n = n0; n < n1; n++)
                    {
                        buff[m-m0 + bs*(n-n0)] = Y[n + (m-M0)*N];
                    }
                }

                // write back transposed
                for(i64 n = n0; n < n1; n++)
                {
                    for(i64 m = m0; m < m1; m++)
                    {
                        X[m + n*M] = buff[m-m0 + bs*(n-n0)];
                    }
                }
            }

        }
    }


    return;
}


static void
fft_Y_plane(fftwf_complex * restrict X,
            fftwf_complex * restrict Y,
            i64 M, i64 N,
            fftwf_plan restrict plan2,
            fftwf_plan restrict plan2_aligned)
{
        fft_Y_plane_large(X, Y,
                          M, N,
                          plan2, plan2_aligned);
}

static void
fft_Z_large(double * restrict X,
            double * restrict * restrict W0, // temp work space of size M*N
            i64 M, i64 N, i64  P,
            fftwf_plan plan,
            fftwf_plan aligned_plan)
{
    const i64 bs = block_side();
    i64 nl = bs * (SFFT3_L2 / (P*bs*8 ) / 4);
    nl < bs ? nl = bs : 0;

#pragma omp parallel for collapse(2) schedule(static)
    for(i64 n = 0; n < N; n++)
    {

        for(i64 M0 = 0; M0 < M; M0+=nl)
        {
            i64 M1 = M0 + nl;
            M1 > M ? M1 = M : 0;

            double * W = W0[omp_get_thread_num()];
            double buff[bs*bs];

            for(i64 p0 = 0; p0 < P; p0+=bs)
            {
                for(i64 m0 = M0; m0 < M1; m0+=bs)
                {
                    i64 p1 = p0 + bs;
                    p1 > P ? p1 = P : 0;
                    i64 m1 = m0 + bs;
                    m1 > M1 ? m1 = M1 : 0;

                    for(i64 p = p0; p < p1; p++)
                    {
                        for(i64 m = m0; m < m1; m++)
                        {
                            buff[m-m0 + (p-p0)*bs] = X[m + n*M + p*M*N];
                        }
                    }

                    for(i64 m = m0; m < m1; m++)
                    {
                        for(i64 p = p0; p < p1; p++)
                        {
                            W[p + (m-M0)*P] = buff[m-m0 + (p-p0)*bs];
                        }
                    }
                }
            }

            for(i64 m = 0; m < M1-M0; m++)
            {
                if( (ptrdiff_t) (W+m*P) % 16 == 0)
                {
                    fftwf_execute_dft(aligned_plan,
                                      (fftwf_complex*) W+m*P,
                                      (fftwf_complex*) W+m*P);
                } else {
                    fftwf_execute_dft(plan,
                                      (fftwf_complex*) W+m*P,
                                      (fftwf_complex*) W+m*P);
                }
            }


            for(i64 p0 = 0; p0 < P; p0+=bs)
            {
                for(i64 m0 = M0; m0 < M1; m0+=bs)
                {
                    i64 p1 = p0 + bs;
                    p1 > P ? p1 = P : 0;
                    i64 m1 = m0 + bs;
                    m1 > M1 ? m1 = M1 : 0;

                    for(i64 m = m0; m < m1; m++)
                    {
                        for(i64 p = p0; p < p1; p++)
                        {
                            buff[m-m0 + (p-p0)*bs] = W[p + (m-M0)*P];
                        }
                    }

                    for(i64 p = p0; p < p1; p++)
                    {
                        for(i64 m = m0; m < m1; m++)
                        {
                            X[m + n*M + p*M*N] = buff[m-m0 + (p-p0)*bs];
                        }
                    }
                }
            }

        }
    }
    return;
}

static void
fft_Z(double * restrict X,
      double * restrict * restrict W0, // temp work space of size M*N
      i64 M, i64 N, i64  P,
      fftwf_plan plan,
      fftwf_plan aligned_plan)
{
        fft_Z_large(X, W0, M, N, P, plan, aligned_plan);
}

void sfft3_execute_dft_r2c(sfft_plan * plan, float * X)
{
    i64 M = plan->M;
    i64 N = plan->N;
    i64 P = plan->P;

    i64 cM = (1+M/2); /* complex per line in the first dimension */
    i64 sM = cM*2; /* floats per line in the first dimension after padding*/

    i64 workspace = cM*N;
    P > N ? workspace = cM*P : 0;
    double * BB[omp_get_max_threads()];

    for(int kk = 0; kk < omp_get_max_threads(); kk++)
    {
        BB[kk] = plan->workspace + kk*workspace;
    }

    /* FFT in X and Y */
#pragma omp parallel for schedule(static) num_threads(nomp(P))
    for(i64 p = 0; p < P; p++)
    {
        for(i64 n = 0; n < N; n++)
        {
            float * L = X + sM*n + p*sM*N;
            if( (ptrdiff_t) L % 16 == 0)
            {
                fftwf_execute_dft_r2c(plan->forward_a_M, L, (fftwf_complex*) L);
            } else {
                fftwf_execute_dft_r2c(plan->forward_M, L, (fftwf_complex*) L);
            }
        }


        fft_Y_plane((fftwf_complex*) (X+p*sM*N),
                    (fftwf_complex*) BB[omp_get_thread_num()], cM, N,
                    plan->forward_N,
                    plan->forward_a_N);

    }
    fft_Z((double*) X, BB, cM, N, P, plan->forward_P, plan->forward_a_P);
    return;
}

void sfft3_execute_dft_c2r(sfft_plan * plan, fftwf_complex * CX)
{
    const i64 M = plan->M;
    const i64 N = plan->N;
    const i64 P = plan->P;
    const i64 cM = (1+M/2);     /* Number of complex per line in the first dimension */

    i64 workspace = cM*N;
    P > N ? workspace = cM*P : 0;
    double * BB[omp_get_max_threads()];

    for(int kk = 0; kk < omp_get_max_threads(); kk++)
    {
        BB[kk] = plan->workspace + kk*workspace;
    }

    /* iFFT along Z */
    fft_Z((double*) CX, BB, cM, N, P, plan->backward_P, plan->backward_a_P);

    /* FFT in X and Y */
#pragma omp parallel for schedule(static) num_threads(nomp(P))
    for(i64 p = 0; p < P; p++)
    {
        fft_Y_plane(CX + p*cM*N,
                    (fftwf_complex*) BB[omp_get_thread_num()], cM, N,
                    plan->backward_N, plan->backward_a_N);

        for(i64 n = 0; n < N; n++)
        {
            fftwf_complex * L = CX + cM*n + p*cM*N;
            if( (ptrdiff_t) L % 16 == 0)
            {
                fftwf_execute_dft_c2r(plan->backward_a_M, L, (float*) L);
            } else {
                fftwf_execute_dft_c2r(plan->backward_M, L, (float*) L);
            }
        }
    }


    if(plan->scale)
    {
        float * X = (float*) CX;
        float scale = (float) (M * N * P);
#pragma omp parallel for
        for(i64 kk = 0; kk < M*N*P; kk++)
        {
            X[kk] /= scale;
        }
    }
    return;
}

void sfft3_destroy_plan(sfft_plan * plan)
{
    if(plan == NULL)
    {
        return;
    }
    fftwf_destroy_plan(plan->forward_M);
    fftwf_destroy_plan(plan->forward_N);
    fftwf_destroy_plan(plan->forward_P);
    fftwf_destroy_plan(plan->forward_a_M);
    fftwf_destroy_plan(plan->forward_a_N);
    fftwf_destroy_plan(plan->forward_a_P);
    fftwf_destroy_plan(plan->backward_M);
    fftwf_destroy_plan(plan->backward_N);
    fftwf_destroy_plan(plan->backward_P);
    fftwf_destroy_plan(plan->backward_a_M);
    fftwf_destroy_plan(plan->backward_a_N);
    fftwf_destroy_plan(plan->backward_a_P);
    free(plan->workspace);
    free(plan);
}

static fftwf_plan gen_fftwf_1d_plan(i64 M, int flags, int direction)
{

    fftwf_complex * CX = fftwf_malloc((M+2)*sizeof(fftwf_complex));

    fftwf_plan plan = fftwf_plan_dft_1d(M,
                                        CX, CX,
                                        direction,
                                        flags);
    fftwf_free(CX);
    if(plan == NULL)
    {
        fprintf(stderr, "fftwf_plan_dft_1d failed on line %d\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    return plan;
}

static fftwf_plan gen_fftwf_1d_plan_c2r(i64 M, int flags)
{
#ifndef NDEBUG
    printf("Planning for %ld elements r2c\n", M);
#endif
    fftwf_complex * X = fftwf_malloc(M*sizeof(fftwf_complex));

    fftwf_plan plan = fftwf_plan_dft_c2r_1d(M,
                                            X, (float*) X,
                                            flags);
    if(plan == NULL)
    {
        fprintf(stderr, "fftwf_plan_dft_c2r_1d failed on line %d\n", __LINE__);
        exit(EXIT_FAILURE);
    }

    fftwf_free(X);

    return plan;
}

static fftwf_plan gen_fftwf_1d_plan_r2c(i64 M, int flags)
{
#ifndef NDEBUG
    printf("Planning for %ld elements r2c\n", M);
#endif
    fftwf_complex * X = fftwf_malloc((M+2)*sizeof(fftwf_complex));

    fftwf_plan plan = fftwf_plan_dft_r2c_1d(M,
                                            (float *) X, X,
                                            flags);
    if(plan == NULL)
    {
        fprintf(stderr, "fftwf_plan_dft_r2c_1d failed on line %d\n", __LINE__);
        exit(EXIT_FAILURE);
    }
    fftwf_free(X);

    return plan;
}

static i64 maxi64(i64 a, i64 b)
{
    if(a > b)
        return a;
    return b;
}

sfft_plan * sfft3_create_plan(i64 M, i64 N, i64 P,
                              int threads,
                              int verbose,
                              int FFTW_PLANNER_FLAGS)
{
    fftwf_plan_with_nthreads(1);
    sfft_plan * plan = calloc(1, sizeof(sfft_plan));
    assert(plan != NULL);
    plan->M = M;
    plan->N = N;
    plan->P = P;
    plan->verbose = verbose;
    plan->threads = threads;

    i64 tws = maxi64(M*N*sizeof(float), P*M*sizeof(fftwf_complex));

    plan->workspace_size = threads*tws;
    assert(sizeof(double) == sizeof(fftwf_complex));
    plan->workspace = (double*) calloc(plan->workspace_size * plan->threads, sizeof(double));
    assert(plan->workspace != NULL);

    plan->forward_M = gen_fftwf_1d_plan_r2c(M, FFTW_PLANNER_FLAGS | FFTW_UNALIGNED);
    plan->forward_N = gen_fftwf_1d_plan(N, FFTW_PLANNER_FLAGS | FFTW_UNALIGNED, FFTW_FORWARD);
    plan->forward_P = gen_fftwf_1d_plan(P, FFTW_PLANNER_FLAGS | FFTW_UNALIGNED, FFTW_FORWARD);
    plan->forward_a_M = gen_fftwf_1d_plan_r2c(M, FFTW_PLANNER_FLAGS);
    plan->forward_a_N = gen_fftwf_1d_plan(N, FFTW_PLANNER_FLAGS, FFTW_FORWARD);
    plan->forward_a_P = gen_fftwf_1d_plan(P, FFTW_PLANNER_FLAGS, FFTW_FORWARD);

    plan->backward_M = gen_fftwf_1d_plan_c2r(M, FFTW_PLANNER_FLAGS | FFTW_UNALIGNED);
    plan->backward_N = gen_fftwf_1d_plan(N, FFTW_PLANNER_FLAGS | FFTW_UNALIGNED, FFTW_BACKWARD);
    plan->backward_P = gen_fftwf_1d_plan(P, FFTW_PLANNER_FLAGS | FFTW_UNALIGNED, FFTW_BACKWARD);
    plan->backward_a_M = gen_fftwf_1d_plan_c2r(M, FFTW_PLANNER_FLAGS);
    plan->backward_a_N = gen_fftwf_1d_plan(N, FFTW_PLANNER_FLAGS, FFTW_BACKWARD);
    plan->backward_a_P = gen_fftwf_1d_plan(P, FFTW_PLANNER_FLAGS, FFTW_BACKWARD);

    return plan;
}
