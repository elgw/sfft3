# TODO:

# SFFT3

This repo contains boiler plate code to lift an 1D FFT into 3D using
straight forward code with no explicit vectorization or auto
tuning. It is a proof-of-concept project which is not tested enough to
be trusted yet. That said, it should be simple to swap the 1D
transformation routines with those from other library.

SFFT3 uses the 1D FFT routines from [fftw3](https://www.fftw.org/) to
build an 3D FFT using semi in-place transpositions. For that purpose
it use a per-thread workspace buffer, each the size of the largest
plane. That is more than FFTW3 what uses, hence timings are not
directly comparable/fair.

It was a fun exercise to write this and I learned a lot :) Including:

- It is not trivial at all to write fast code for in-place
  transpositions (slow code safely tucked away in another repository).

- Even out of place transpositions are hard to make fast. And it
  turned out that it was never a good idea to transpose the full data
  set, not even a full plane at a time. Eventually the solution that I
  took was to transpose the data planes in chunks, processes them and
  then store back the results.

## Results

Below are some results (average execution for pairs of forward and
reverse transforms, time given in seconds) comparing the specialized
3D FFT routines in FFTW3 to SFFT3. An 8-core AMD Ryzen 3700X machine
with Ubuntu 24.04.2 with gcc 13.3 was used. FFTW3-I.E. denotes FFTW3
in-place with `FFTW_ESTIMATE`, FFTW3-I.M. denotes FFTW3 in-place with
`FFTW_MEASURE`. In all cases 8 threads were used.

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

<details><summary>Benchmark code</summary>

``` shell
args="--warmup 0.1 --benchmark 20  --verbose 2"
# add --estimate to use FFTW_ESTIMATE instead of FFTW_MEASURE
th=1
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

- Does the results suggests that the fastest fourier
  transform in the west (FFTW3) can be even faster if it just use a
  little more memory?

- For small sizes my code is slow since the transposing parts use
  intermediate buffers which only benefits large problems.

- FFTW has the flag `FFTW_DESTROY_INPUT` which I've not tested. In
  that case it is allowed to use a whole extra image as buffer.

- The code here was tested on a single CPU and hence buffer sizes etc
  are all right for that machine, but will probably be bad choices for
  other systems.
