# SFFT3

This repo contains boiler plate code (~500 SLOC or approximately the size of
a FFTW plan in nerd format) to lift an 1D FFT into 3D using straight
forward code with no explicit vectorization or auto tuning. It should
be considered proof-of-concept since it is neither properly tested nor
used anywhere else at the moment. That said, it should be simple to
swap the 1D transformation routines with those from another library.

SFFT3 uses the 1D FFT routines from [fftw3](https://www.fftw.org/) to
build a semi-inplace 3D FFT. For that purpose per-thread workspace
buffers are used which, at most the size of the largest plane
each. The 3D routines in FFTW3 use less memory, hence timings are not
directly comparable/fair.

It was a fun exercise to write this and I learned/confirmed a lot
Including:

1. It is not trivial at all to write fast code for in-place
   transpositions (slow code safely tucked away in another repository).

2. Even out of place transpositions are hard to make fast.

3. Results do not transfer between machines. Self-tuning algorithms
   (like FFTW uses) is a brilliant idea.

Eventually I decided to transposes the data planes in chunks,
processes them and then store back the results. Only tiny plane are
transposed in full.

## Results

The table below shows average execution times, in seconds, for pairs
of forward and reverse transforms, comparing the specialized 3D FFT
routines in FFTW3 to SFFT3. An 8-core AMD Ryzen 3700X machine with
Ubuntu 24.04.2 and gcc 13.3 was used (it can be built with clang as
well). **-E** denotes FFTW3 `FFTW_ESTIMATE` and **-P** denotes FFTW3
`FFTW_PATIENT`. Data padding and FFTW planning times are not
included. In all cases 8 threads were used. The OMP version of fftw3
(`-lfftw3d_omp`) is used since I've found that to be faster than the
threads version (`-fftw3f_threads`).

| Size           | FFTW3-E   | FFTW3-P       | SFFT3-P       |
|----------------|-----------|---------------|---------------|
| 64x64x64       | 1.285e-04 | **1.183e-04** | 1.935e-04     |
| 128x128x128    | 9.580e-04 | **7.151e-04** | 1.077e-03     |
| 256x256x256    | 3.926e-02 | 1.236e-02     | **1.041e-02** |
| 512x256x128    | 1.461e-02 | 1.330e-02     | **1.085e-02** |
| 512x512x512    | 4.492e-01 | 9.952e-02     | **9.351e-02** |
| 1009x829x211   | 7.434e-01 | **5.799e-01** | 5.832e-01     |
| 1024x1024x256  | 8.985e-01 | **2.027e-01** | 2.127e-01     |
| 1024x1024x1024 | 5.973e+00 | **8.495e-01** | 8.664e-01     |
| 2100x2100x121  | 9.959e-01 | **4.934e-01** | 5.420e-01     |
|                |           |               |               |

<details><summary>Details about the benchmark</summary>

``` shell
make
args="--warmup 10 --benchmark 30  --verbose 2 --patient"
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


<details><summary>4-core Intel 6700k benchmark results</summary>

This machine has 256 kB L2. `-M` stands for `FFTW_MEASURE`.

| Size           | FFTW3-M       | SFFT3-M       |
|----------------|---------------|---------------|
| 128x128x128    | **1.517e-03** | 2.717e-03 |
| 256x256x256    | **2.152e-02** | 2.254e-02 |
| 512x256x128    | **1.982e-02** | 2.179e-02 |
| 512x512x512    | 1.960e-01     | **1.820e-01** |
| 1009x829x211   | 1.853e+00     | **1.832e+00** |
| 1024x1024x256  | 3.949e-01     | **3.723e-01** |
| 1024x1024x1024 | 1.705e+00     | **1.640e+00** |
| 2100x2100x121  | **9.696e-01** | 1.110e+00  |
|                |               |  |

compiled and ran with:

``` shell
CFLAGS="-DSFFT3_L2=256000" make -B
args="--warmup 5 --benchmark 20"
th=4
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
  even faster using a larger workspace/and/or including another
  sniplet. For in-place transforms there is no flag for that. For
  out-of-place transforms there is the option `FFTW_DESTROY_INPUT`
  which I've not tested.

- The code here was tested on a single CPU and hence buffer sizes etc
  are all right for that machine, but will probably be bad choices for
  other systems.

- For the smallest problem, 64x64x64, SFFT3 is actually faster when
  using only one thread. A reasonable library should figure out :)
