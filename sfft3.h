#pragma once

/* Copyright (C) 2025 Erik Wernersson
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <stdint.h>
#include <fftw3.h>

#ifndef SFFT3_L1
#define SFFT3_L1 64000
#endif
#ifndef SFFT3_L2
#define SFFT3_L2 512000
#endif

typedef struct {
    double * workspace;
    int64_t workspace_size; /* Number of bytes allocated for the workspace */
    int verbose;
    int threads;
    int scale; /* Set to 1 to rescale after c2r */
    int M; /* Problem size. M: size of non-strided dimension */
    int N;
    int P;
    fftwf_plan forward_M;
    fftwf_plan forward_N;
    fftwf_plan forward_P;
    fftwf_plan forward_a_M;
    fftwf_plan forward_a_N;
    fftwf_plan forward_a_P;
    fftwf_plan backward_M;
    fftwf_plan backward_N;
    fftwf_plan backward_P;
    fftwf_plan backward_a_M;
    fftwf_plan backward_a_N;
    fftwf_plan backward_a_P;
} sfft_plan;

sfft_plan * sfft3_create_plan(int64_t M, int64_t N, int64_t P,
                             int threads,
                             int verbose,
                             int FFT_PLANNER_FLAGS);

void sfft3_destroy_plan(sfft_plan *);

/* in-place 3D FFT, requires that the input array is padded for 3D in
 * the same way as fftwf_execute_dft_r2c.
 *
 * M is the size of the non strided dimension, i.e. the first in
 * fortran order
 */
void
sfft3_execute_dft_r2c(sfft_plan * plan, float * X);

/* Inverse 3D FFT.
 * corresponds to fftwf_execute_dft_c2r.
 * M, N, P: The real size of the data
 */
void
sfft3_execute_dft_c2r(sfft_plan * plan, fftwf_complex * CX);
