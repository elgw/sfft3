# SFFT3

This repo contains boiler plate code (<400 SLOC) to lift an 1D FFT
into 3D using straight forward code with no explicit vectorization or
auto tuning. It should be considered proof-of-concept since it is neither
properly tested nor used anywhere else at the moment. That said, it
should be simple to swap the 1D transformation routines with those
from another library.

SFFT3 uses the 1D FFT routines from [fftw3](https://www.fftw.org/) to
build a semi-inplace 3D FFT. For that purpose per-thread workspace
buffers are used which, at most the size of the largest plane
each. The 3D routines in FFTW3 use less memory, hence timings are not
directly comparable/fair.

It was a fun exercise to write this and I learned a lot Including:

1. It is not trivial at all to write fast code for in-place
   transpositions (slow code safely tucked away in another repository).

2. Even out of place transpositions are hard to make fast.

Eventually it turned out that it wasn't a good idea to transpose the
full data set, not even a full plane at a time. Current code
transposes the data planes in chunks, processes them and then store
back the results.

## Results

The table below shows average execution times, in seconds, for pairs
of forward and reverse transforms, thus comparing the specialized 3D
FFT routines in FFTW3 to SFFT3. An 8-core AMD Ryzen 3700X machine with
Ubuntu 24.04.2 and gcc 13.3 was used (it can be built with clang as
well). **FFTW3-I.E.** denotes FFTW3 in-place with `FFTW_ESTIMATE`,
**FFTW3-I.M.** denotes FFTW3 in-place with `FFTW_MEASURE`, which is
the default setting. Data padding and FFTW planning times are not
included. In all cases 8 threads were used.

| Size           | FFTW3-I.E. | FFTW3-I.M.    | SFFT3         |
|----------------|------------|---------------|---------------|
| 128x128x128    | 9.580e-04  | **6.982e-04** | 1.031e-03     |
| 256x256x256    | 3.926e-02  | 1.300e-02     | **1.053e-02** |
| 512x256x128    | 1.461e-02  | 1.285e-02     | **1.095e-02** |
| 512x512x512    | 4.492e-01  | 1.012e-01     | **9.628e-02** |
| 1009x829x211   | 7.434e-01  | **6.229e-01** | 6.254e-01     |
| 1024x1024x256  | 8.985e-01  | 2.189e-01     | **2.171e-01** |
| 1024x1024x1024 | 5.973e+00  | 9.239e-01     | **8.581e-01** |
| 2100x2100x121  | 9.959e-01  | 6.485e-01     | **5.560e-01** |
|                |            |               |               |

<details><summary>4-core Intel 6700k benchmark results</summary>

| Size           | FFTW3-I.M. | SFFT3 |
|----------------|------------|-------|
| 128x128x128    |            |       |
| 256x256x256    |            |       |
| 512x256x128    |            |       |
| 512x512x512    |            |       |
| 1009x829x211   |            |       |
| 1024x1024x256  |            |       |
| 1024x1024x1024 |            |       |
| 2100x2100x121  |            |       |


</details>

<details><summary>Benchmark code</summary>

``` shell
SFFT3_L1=64000 SFFT_L3=32000000 make
args="--warmup 0.1 --benchmark 20  --verbose 2"
# add --estimate to use FFTW_ESTIMATE instead of FFTW_MEASURE
th=8
OMP_NUM_THREADS=${th} ./test_sfft3 --m 128 --n 128 --p 128 ${args} --warmup 10
OMP_NUM_THREADS=${th} ./test_sfft3 --m 256 --n 256 --p 256 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 512 --n 256 --p 128 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 512 --n 512 --p 512 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 1009 --n 829 --p 211 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 1024 --n 1024 --p 256 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 1024 --n 1024 --p 1024 ${args}
OMP_NUM_THREADS=${th} ./test_sfft3 --m 2100 --n 2100 --p 121 ${args}
```
</details>

- Possibly the fastest fourier transform in the west (FFTW3) can be
  even faster using a larger workspace. For in-place transforms there
  is no flag for that. For out-of-place transforms there is the option
  `FFTW_DESTROY_INPUT` which I've not tested. I

- For small sizes my code is slow since the code for transpositions
  use intermediate buffers, requiring extra reads and writes, which
  are beneficial for large problems only.

- The code here was tested on a single CPU and hence buffer sizes etc
  are all right for that machine, but will probably be bad choices for
  other systems.
