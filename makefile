CC = gcc
CFLAGS=-Wall -Wextra -pedantic -std=gnu11 -Wno-unused-function

CFLAGS+=-fopenmp
LDFLAGS+=-lm

CFLAGS+=`pkg-config fftw3f --cflags`
LDFLAGS+=`pkg-config fftw3f --libs`
LDFLAGS+=-lfftw3f_omp

DEBUG?=0
ifeq ($(DEBUG),1)
CFLAGS += -O0 -g3
else
CFLAGS+=-mtune=native -march=native
CFLAGS += -O3 -DNDEBUG
LDFLAGS+=-flto=auto
CFLAGS+=-fno-schedule-insns # turned on by O2 but it is faster without.
endif

FSAN?=0
ifeq ($(FSAN),1)
CFLAGS+=-fsanitize=address
endif

test_sfft3: sfft3.o test_sfft3.c
	$(CC) $(CFLAGS) test_sfft3.c sfft3.o $(LDFLAGS) -o test_sfft3

sfft3.o: sfft3.c
	$(CC) -c $(CFLAGS) sfft3.c
