# SFFT3

This repo contains boiler plate code (~500 SLOC or approximately the size of
a FFTW plan in nerd format) to lift a 1D FFT into 3D using straight
forward code with no explicit vectorization or auto tuning. It should
be considered proof-of-concept since it is neither properly tested nor
used anywhere else at the moment. That said, it should be simple to
swap the 1D transformation routines with those from another library.

SFFT3 uses the 1D FFT routines from [fftw3](https://www.fftw.org/) to
build a semi-inplace 3D FFT. For that purpose per-thread workspace
buffers are used, at most the size of the largest plane each. The 3D
routines in FFTW3 use less memory, hence timings are not directly
comparable/fair.

It was a fun exercise to write this and I learned/confirmed a lot
Including:

1. It is not trivial at all to write fast code for in-place
   transpositions (slow code safely tucked away in another repository).

2. Even out of place transpositions are hard to make fast.

3. Results do not transfer between machines. Self-tuning algorithms
   (like FFTW uses) is a brilliant idea.

Eventually I decided to transposes the data planes in chunks,
processes them and then store back the results. Only tiny
planes/slices are transposed in full.

## Method

1. For each XY-plane.
   1. FFT for each line
   2. For a few Y lines at a time: transpose to the buffer, calculate
      the FFT, transpose back and store in the original location.
2. For each XZ-plane
   1. For a few Z lines at a time: transpose to the buffer, calculate
      the FFT, transpose back and store in the original location.

where _a few_ means what fits into L2 memory. The transposition
routine use an intermediate buffer that fits into L1 memory. The L1
and L2 sizes are compile time constants.

## Results

The tables below shows average execution times in seconds for pairs
of forward and reverse transforms, comparing the specialized 3D FFT
routines in FFTW3 to SFFT3. The code was built with gcc, although
clang can be used as well.

**-E** denotes `FFTW_ESTIMATE` **-M** stands for `FFTW_MEASURE`, and
**-P** is `FFTW_PATIENT`. The time consumption for data padding and
planning (by FFTW) is not included.

In all cases one threads per physical core is used. I've linked
against `-lfftw3d_omp` since I have a faint memory of that being
faster than `-fftw3f_threads`.

### 8-core AMD Ryzen 3700X

Ubuntu 24.04.2

| Size           | FFTW3-E   | FFTW3-P       | SFFT3-P       |
|----------------|-----------|---------------|---------------|
| 64x64x64       | 1.285e-04 | **1.185e-04** | 1.821e-04     |
| 128x128x128    | 9.580e-04 | **7.054e-04** | 1.060e-03     |
| 256x256x256    | 3.926e-02 | 1.213e-02     | **1.112e-02** |
| 512x256x128    | 1.461e-02 | 1.208e-02     | **1.021e-02** |
| 512x512x512    | 4.492e-01 | 1.009e-01     | **8.428e-02** |
| 1009x829x211   | 7.434e-01 | 5.754e-01     | **5.738e-01** |
| 1024x1024x256  | 8.985e-01 | 2.022e-01     | **1.825e-01** |
| 1024x1024x1024 | 5.973e+00 | 1.011e+00     | **7.733e-01** |
| 2100x2100x121  | 9.959e-01 | 5.254e-01     | **5.115e-01** |
|                |           |               |               |

<details><summary>Details about the benchmark</summary>

``` shell
make
args="--warmup 60 --benchmark 240  --verbose 2 --patient"
th=8
OMP_NUM_THREADS=${th} ./test_sfft3 --m 64 --n 64 --p 64 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 128 --n 128 --p 128 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 256 --n 256 --p 256 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 512 --n 256 --p 128 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 512 --n 512 --p 512 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 1009 --n 829 --p 211 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 1024 --n 1024 --p 256 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 1024 --n 1024 --p 1024 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 2100 --n 2100 --p 121 ${args}
```
</details>


## 4-core Intel 6700k

Ubuntu 22.04.5

| Size           | FFTW3-M       | SFFT3-M       |
|----------------|---------------|---------------|
| 128x128x128    | **1.517e-03** | 2.717e-03     |
| 256x256x256    | **2.152e-02** | 2.254e-02     |
| 512x256x128    | **1.982e-02** | 2.179e-02     |
| 512x512x512    | 1.960e-01     | **1.820e-01** |
| 1009x829x211   | 1.853e+00     | **1.832e+00** |
| 1024x1024x256  | 3.949e-01     | **3.723e-01** |
| 1024x1024x1024 | 1.705e+00     | **1.640e+00** |
| 2100x2100x121  | **9.696e-01** | 1.110e+00     |
| 2048x2048x1024 | 8.092e+00     | **7.360e+00** |




<details><summary>Details about the benchmark</summary>

``` shell
CFLAGS="-DSFFT3_L2=256000" make -B
args="--warmup 10 --benchmark 240"
th=4
OMP_NUM_THREADS=${th} ./test_sfft3 --m 128 --n 128 --p 128 ${args} 
OMP_NUM_THREADS=${th} ./test_sfft3 --m 256 --n 256 --p 256 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 512 --n 256 --p 128 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 512 --n 512 --p 512 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 1009 --n 829 --p 211 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 1024 --n 1024 --p 256 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 1024 --n 1024 --p 1024 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 2100 --n 2100 --p 121 ${args}
# Large sizes can be split up save memory
OMP_NUM_THREADS=${th} ./test_sfft3 --nofftw  --m 2048 --n 2048 --p 1024
OMP_NUM_THREADS=${th} ./test_sfft3 --nosfft  --m 2048 --n 2048 --p 1024
```

</details>

# Final words

- This is a proof-of-concept, and nothing more is planned.

- Although this code use a larger workspace than FFTW it is still
  surprisingly fast, given that there is no explicit vectorization and
  that it is not even using
  [`fftwf_plan_many`](https://www.fftw.org/doc/Advanced-Complex-DFTs.html)
  which should be faster. On the other hand it is only run on two,
  quite similar, machines and I don't expect that the results
  generalize to other hardware.

- For the smallest problem, $`64\times64\times64`$, SFFT3 is actually faster when
  using only one thread. A reasonable library should figure out :)

- Already here, with only a few parameters in the algorithms it is
  clear that self-tuning could be of use.
  
- Now I'l go back and optimize that separable convolution which was
  on the table before I got [nerd sniped](https://xkcd.com/356/).
