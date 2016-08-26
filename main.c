/*
 * Copyright (c) 2016 Sugizaki Yukimasa
 * All rights reserved.
 *
 * This software is licensed under a Modified (3-Clause) BSD License.
 * You should have received a copy of this license along with this
 * software. If not, contact the copyright holder above.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <vc4vec.h>

const unsigned code[] = {
#include "sgemm.qhex"
};
const int unif_len = 1024;

static void unif_add_unsigned(const unsigned val, unsigned **p)
{
	memcpy(*p, &val, sizeof(val));
	(*p)++;
}

static void unif_add_float(const float val, unsigned **p)
{
	memcpy(*p, &val, sizeof(val));
	(*p)++;
}

static void mf_srandom()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	srandom(tv.tv_sec ^ tv.tv_usec);
}

static void mf_init_random(float *p, const int height, const int width)
{
	int i, j;

	for (i = 0; i < height; i ++)
		for (j = 0; j < width; j ++)
			p[i * width + j] = (random() % 10) / 10.0;
}

static void mf_init_seq(float *p, const int height, const int width)
{
	int i, j;

	for (i = 0; i < height; i ++)
		for (j = 0; j < width; j ++)
			p[i * width + j] = (i + 10) + (j + 10) / 1000.0;
}

static void mf_print(float *p, const int height, const int width)
{
	int i, j;

	for (i = 0; i < height; i ++) {
		printf("%3d:", i);
		for (j = 0; j < width; j ++)
			printf(" %7.3f", p[i * width + j]);
		printf("\n");
	}
}

static void mu_print_hex(unsigned *p, const int height, const int width)
{
	int i, j;
	for (i = 0; i < height; i ++) {
		printf("%3d:", i);
		for (j = 0; j < width; j ++)
			printf(" %08x", p[i * width + j]);
		printf("\n");
	}
}

static float mf_minimum_absolute_error(float *C1, float *C2, const int P, const int R)
{
	int i, j;
	float minimum_error = +INFINITY;
	for (i = 0; i < P; i ++) {
		for (j = 0; j < R; j ++) {
			float error = fabs(C1[i * R + j] - C2[i * R + j]);
			if (error < minimum_error)
				minimum_error = error;
		}
	}
	return minimum_error;
}

static float mf_maximum_absolute_error(float *C1, float *C2, const int P, const int R)
{
	int i, j;
	float maximum_error = 0.0;
	for (i = 0; i < P; i ++) {
		for (j = 0; j < R; j ++) {
			float error = fabs(C1[i * R + j] - C2[i * R + j]);
			if (error > maximum_error)
				maximum_error = error;
		}
	}
	return maximum_error;
}

static float mf_minimum_relative_error(float *C1, float *C2, const int P, const int R)
{
	int i, j;
	float minimum_error = +INFINITY;
	for (i = 0; i < P; i ++) {
		for (j = 0; j < R; j ++) {
			float error = fabs((C1[i * R + j] - C2[i * R + j]) / C2[i * R + j]);
			if (error < minimum_error)
				minimum_error = error;
		}
	}
	return minimum_error;
}

static float mf_maximum_relative_error(float *C1, float *C2, const int P, const int R)
{
	int i, j;
	float maximum_error = 0.0;
	for (i = 0; i < P; i ++) {
		for (j = 0; j < R; j ++) {
			float error = fabs((C1[i * R + j] - C2[i * R + j]) / C2[i * R + j]);
			if (error > maximum_error)
				maximum_error = error;
		}
	}
	return maximum_error;
}

static void mf_sgemm(float *A, float *B, float *C, const int P, const int Q, const int R, const float ALPHA, const float BETA)
{
	int i;

#pragma omp parallel for
	for (i = 0; i < P; i ++) {
		int j;
		for (j = 0; j < R; j ++) {
			int k;
			float sum = 0.0;
			for (k = 0; k < Q; k ++) {
				sum += A[i * Q + k] * B[k * R + j];
			}
			C[i * R + j] = ALPHA * sum + BETA * C[i * R + j];
		}
	}
}

int main()
{
	struct vc4vec_mem mem_unif, mem_code, A_BASE, B_BASE, C_BASE, C_ref;
	unsigned *p = NULL;
	int exitcode = EXIT_SUCCESS;
	struct timeval start, end;

	const unsigned NREG = 16;
	const unsigned P = 96;
	const unsigned Q = 363;
	const unsigned R = 3072;
	const float ALPHA = 3.3;
	const float BETA = 2.1;
	unsigned A_STRIDE_K, B_STRIDE_K, B_STRIDE_J, C_STRIDE_J, A_STRIDE_I, C_STRIDE_I;

	if (sizeof(unsigned) != (32 / 8)) {
		fprintf(stderr, "error: size of unsigned is not 4\n");
		exitcode = EXIT_FAILURE;
		goto cleanup;
	}

	if (sizeof(float) != (32 / 8)) {
		fprintf(stderr, "error: size of float is not 4\n");
		exitcode = EXIT_FAILURE;
		goto cleanup;
	}

	if (P % 16 != 0) {
		fprintf(stderr, "error: P must be a multiple of 16\n");
		exitcode = EXIT_FAILURE;
		goto cleanup;
	}

	if (R % NREG != 0) {
		fprintf(stderr, "error: R must be a multiple of NREG\n");
		exitcode = EXIT_FAILURE;
		goto cleanup;
	}

	vc4vec_init();
	vc4vec_mem_alloc(&mem_unif, unif_len * (32 / 8));
	vc4vec_mem_alloc(&mem_code, sizeof(code));
	vc4vec_mem_alloc(&A_BASE, P * Q * (32 / 8));
	vc4vec_mem_alloc(&B_BASE, Q * R * (32 / 8));
	vc4vec_mem_alloc(&C_BASE, P * R * (32 / 8));
	vc4vec_mem_alloc(&C_ref,  P * R * (32 / 8));

	memcpy(mem_code.cpu_addr, code, sizeof(code));
#if 1
	mf_srandom(); mf_init_random(A_BASE.cpu_addr, P, Q); mf_init_random(B_BASE.cpu_addr, Q, R); mf_init_random(C_BASE.cpu_addr, P, R);
#else
	mf_init_seq(A_BASE.cpu_addr, P, Q); mf_init_seq(B_BASE.cpu_addr, Q, R); mf_init_seq(C_BASE.cpu_addr, P, R);
#endif
	memcpy(C_ref.cpu_addr, C_BASE.cpu_addr, P * R * (32 / 8));

	A_STRIDE_K = (32 / 8);
	B_STRIDE_K = R * (32 / 8);
	B_STRIDE_J = NREG * (32 / 8);
	C_STRIDE_J = NREG * (32 / 8);
	A_STRIDE_I = 16 * Q * (32 / 8);
	C_STRIDE_I = 16 * R * (32 / 8);

	printf("NREG: %d\n", NREG);
	printf("P: %d\n", P);
	printf("Q: %d\n", Q);
	printf("R: %d\n", R);
	printf("ALPHA: %f\n", ALPHA);
	printf("BETA: %f\n", BETA);
	printf("A_STRIDE_K: %d\n", A_STRIDE_K);
	printf("B_STRIDE_K: %d\n", B_STRIDE_K);
	printf("B_STRIDE_J: %d\n", B_STRIDE_J);
	printf("C_STRIDE_J: %d\n", C_STRIDE_J);
	printf("A_STRIDE_I: %d\n", A_STRIDE_I);
	printf("C_STRIDE_I: %d\n", C_STRIDE_I);

	p = mem_unif.cpu_addr;
	unif_add_unsigned(P / 16         , &p);
	unif_add_unsigned(Q              , &p);
	unif_add_unsigned(R / NREG       , &p);
	unif_add_float   (ALPHA          , &p);
	unif_add_float   (BETA           , &p);
	unif_add_unsigned(A_BASE.gpu_addr, &p);
	unif_add_unsigned(B_BASE.gpu_addr, &p);
	unif_add_unsigned(C_BASE.gpu_addr, &p);
	unif_add_unsigned(A_STRIDE_K     , &p);
	unif_add_unsigned(B_STRIDE_K     , &p);
	unif_add_unsigned(B_STRIDE_J     , &p);
	unif_add_unsigned(C_STRIDE_J     , &p);
	unif_add_unsigned(A_STRIDE_I     , &p);
	unif_add_unsigned(C_STRIDE_I     , &p);

	gettimeofday(&start, NULL);
	launch_qpu_job_mailbox(1, 1, 10e3, mem_unif.gpu_addr, mem_code.gpu_addr);
	gettimeofday(&end, NULL);
	printf("GPU: %g [s], %g [flop/s]\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6, (2 * P * Q * R + 3 * P * R) / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6));

	gettimeofday(&start, NULL);
	mf_sgemm(A_BASE.cpu_addr, B_BASE.cpu_addr, C_ref.cpu_addr, P, Q, R, ALPHA, BETA);
	gettimeofday(&end, NULL);
	printf("CPU: %g [s], %g [flop/s]\n", (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6, (2 * P * Q * R + 3 * P * R) / ((end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6));

	printf("Minimum absolute error: %g\n", mf_minimum_absolute_error(C_ref.cpu_addr, C_BASE.cpu_addr, P, R));
	printf("Maximum absolute error: %g\n", mf_maximum_absolute_error(C_ref.cpu_addr, C_BASE.cpu_addr, P, R));
	printf("Minimum relative error: %g\n", mf_minimum_relative_error(C_ref.cpu_addr, C_BASE.cpu_addr, P, R));
	printf("Maximum relative error: %g\n", mf_maximum_relative_error(C_ref.cpu_addr, C_BASE.cpu_addr, P, R));

cleanup:
	vc4vec_mem_free(&C_BASE);
	vc4vec_mem_free(&B_BASE);
	vc4vec_mem_free(&A_BASE);
	vc4vec_mem_free(&mem_code);
	vc4vec_mem_free(&mem_unif);
	vc4vec_finalize();
	exit(exitcode);
}
