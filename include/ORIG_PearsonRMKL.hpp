/*
 * PearsonRMKL.hpp
 *
 *  Created on: Mar 22, 2016
 *  Author: Liu, Yongchao
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 *  URL: www.liuyc.org
 */

#ifndef PEARSONR_MKL_HPP_
#define PEARSONR_MKL_HPP_
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include <vector>
#include <list>
#include <typeinfo>
using namespace std;
#pragma once

#ifdef WITH_MPI
#include <mpi.h>
#endif	/*with mpi*/

/*Intel MKL*/
#include <mkl.h>

#ifdef WITH_PHI
#include <immintrin.h>
/*for device memroy allocation*/
#define PR_MKL_MIC_REUSE alloc_if(0) free_if(0)
#define PR_MKL_MIC_ALLOC alloc_if(1) free_if(0)
#define PR_MKL_MIC_FREE  length(0) alloc_if(0) free_if(1)
#define PR_MKL_MIC_ALLOC_FREE alloc_if(1) free_if(1)
#endif	/*with phi*/

/*template functions for general matrix-matrix multiplication*/
template<typename FloatType>
__attribute__((target(mic))) void mygemm(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const FloatType alpha,
				const FloatType *A, const int lda, const FloatType *B, const int ldb, const FloatType beta, FloatType *C, const int ldc);

template<>
__attribute__((target(mic))) void mygemm<float>(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const float alpha,
        const float *A, const int lda, const float *B, const int ldb, const float beta, float *C, const int ldc)
{
	cblas_sgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}
template<> 
__attribute__((target(mic))) void mygemm<double>(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K, const double alpha,
        const double *A, const int lda, const double *B, const int ldb, const double beta, double *C, const int ldc)
{
	cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

/*template class*/
template<typename FloatType>
class PearsonRMKL {
public:
	PearsonRMKL(int numVectors, int vectorSize, int numCPUThreads,
			int numMICThreads, int micIndex, int rank, int numProcs);
	~PearsonRMKL();

	inline FloatType* getVectors() {
		return _vectors;
	}
	inline int getNumVectors() {
		return _numVectors;
	}
	inline int getVectorSize() {
		return _vectorSize;
	}
	inline int getVectorSizeAligned() {
		return _vectorSizeAligned;
	}

	/*generate random data*/
	void generateRandomData(const int seed = 11);

	/*single threaded implementation*/
	void runSingleThreaded();

	/*multiple threads with optimized*/
	void runMultiThreaded();

#ifdef WITH_PHI
	/*exon phi*/
	void runSingleXeonPhi();
#endif	/*with phi*/

	/*MPI*/
#ifdef WITH_MPI
	void runMPICPU();
#ifdef WITH_PHI
	void runMPIXeonPhi();
#endif	/*with phi*/
#endif	/*with mpi*/

private:
	FloatType* _vectors; /*vector data. Stored consecutively and each vector contains an aligned number of elements*/
	int _numVectors; /*number of vectors in the data*/
	int _vectorSize; /*effective vector size, i.e. the real number of elements per vector*/
	int _vectorSizeAligned; /*aligned to vector size to 16 so that the address is aligned to 64 byte boundary*/
	//int _maxPartitionSize; /*maximum partition size for asynchronous mode*/
	int _numCPUThreads; /*the number of CPU threads*/
	int _numMICThreads; /*the number of MIC threads*/
	int _micIndex;	/*index of MIC used by this processs*/
	int _rank;			/*process rank*/
	int _numProcs;	/*number of MPI processes*/
	FloatType* _pearsonCorr; /*pearson correlation matrix*/

#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void _runSingleXeonPhiCore(
			FloatType* __restrict__ vectors, FloatType* __restrict__ prValues, const ssize_t indexRangeStart, const ssize_t indexRangeClose);

public:

#ifdef WITH_PHI
__attribute__((target(mic)))
#endif
	inline double getSysTime() {
		double dtime;
		struct timeval tv;

		gettimeofday(&tv, NULL);

		dtime = (double) tv.tv_sec;
		dtime += (double) (tv.tv_usec) / 1000000.0;

		return dtime;
	}

	/*for intel*/
#ifdef WITH_PHI
__attribute__((target(mic)))
#endif
	inline void* mm_malloc(ssize_t size, ssize_t alignment)
	{
		return _mm_malloc(size, alignment);
	}
#ifdef WITH_PHI
__attribute__((target(mic)))
#endif
	inline void mm_free(void* buffer)
	{
		_mm_free(buffer);
	}
};

template<typename FloatType>
PearsonRMKL<FloatType>::PearsonRMKL(int numVectors, int vectorSize, int numCPUThreads,
		int numMICThreads, int micIndex, int rank, int numProcs) {
	int alignment = 64 / sizeof(FloatType);	/*align to 64 byte boundary*/
	_numVectors = numVectors;
	_vectorSize = vectorSize;
	_vectorSizeAligned = (_vectorSize + alignment - 1) / alignment * alignment;
	_numCPUThreads = numCPUThreads;
	_numMICThreads = numMICThreads;
	_micIndex = micIndex; /*index of mic device*/
	_rank = rank;
	_numProcs = numProcs;

	/*allocate space*/
	_pearsonCorr = NULL;

	/*align each vector*/
  	
_vectors = (FloatType*) mm_malloc(
			(ssize_t)_numVectors * _vectorSizeAligned * sizeof(FloatType), 64);
	if (!_vectors) {
		fprintf(stderr, "Memory allocation failed at line %d in file %s\n",
		__LINE__, __FILE__);
		exit(-1);
	}
}
 
template<typename FloatType>
PearsonRMKL<FloatType>::~PearsonRMKL() {
	if (_vectors) {
		mm_free(_vectors);
	}
	if (_pearsonCorr) {
		mm_free(_pearsonCorr);
	}
}

template<typename FloatType>
void PearsonRMKL<FloatType>::generateRandomData(const int seed) {
	srand48(11);
	for (int i = 0; i < _numVectors; ++i) {
		FloatType* __restrict__ dst = _vectors + i * _vectorSizeAligned;
		for (int j = 0; j < _vectorSize; ++j) {
			dst[j] = drand48();
		}
	}
}
template<typename FloatType>
void PearsonRMKL<FloatType>::runSingleThreaded() {
	double stime, etime;
#ifdef VERBOSE
	fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

	/*output matrix*/
	_pearsonCorr = (FloatType*) mm_malloc(
			(ssize_t) _numVectors * _numVectors * sizeof(FloatType), 64);
	if (!_pearsonCorr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(-1);
	}

	/*record the system time*/
	stime = getSysTime();

	/*allocate vectors for mean and variance*/
	double t1 = getSysTime();
	FloatType x, meanX, varX, prod;
	FloatType* __restrict__ vecX;
	FloatType* __restrict__ vecY;
	const FloatType avg = 1.0 / (FloatType)_vectorSize;
	vecX = _vectors;
	for (int i = 0; i < _numVectors; ++i, vecX += _vectorSizeAligned) {
		/*get the vector data*/
		meanX = 0;
#pragma vector aligned
#pragma simd reduction(+:meanX)
		for (int j = 0; j < _vectorSize; ++j) {
			meanX += vecX[j] * avg;
		}

		/*compute the variance*/
		varX = 0;
#pragma vector aligned
#pragma simd reduction(+:varX)
		for (int j = 0; j < _vectorSize; ++j) {
			x = vecX[j] - meanX;
			varX += x * x;
		}
		varX = 1 / sqrt(varX);

		/*normalize the data*/
#pragma vector aligned
#pragma simd
		for (int j = 0; j < _vectorSize; ++j) {
			x = vecX[j] - meanX;
			vecX[j] = x * varX;
		}
	}
	double t2 = getSysTime();
	fprintf(stderr, "time for transformation: %f seconds\n", t2 - t1);

	/*invoke the sequential Intel MKL*/
	omp_set_num_threads(1);	/*use a single thread*/
	mygemm<FloatType>(CblasRowMajor, CblasNoTrans, CblasTrans, _numVectors, _numVectors, _vectorSize, 1.0, _vectors, _vectorSizeAligned, _vectors, _vectorSizeAligned, 0, _pearsonCorr, _numVectors);

	/*recored the system time*/
	etime = getSysTime();
	fprintf(stderr, "Overall time: %f seconds\n",
			etime - stime);

#if 0
  for(int i = 0; i < _numVectors; ++i){
    for(int j = i; j < _numVectors; ++j){
      printf("%f\n", _pearsonCorr[i *_numVectors + j]);
    }
  }
#endif
}

template<typename FloatType>
void PearsonRMKL<FloatType>::runMultiThreaded() {
	double stime, etime;
	ssize_t totalNumPairs = 0;
	const FloatType avg = 1.0 / (FloatType)_vectorSize;

#ifdef VERBOSE
  fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

	/*record system time*/
	stime = getSysTime();

	/*allocate space*/
	_pearsonCorr = (FloatType*) mm_malloc(
			(ssize_t) _numVectors * _numVectors * sizeof(FloatType), 64);
	if (!_pearsonCorr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(-1);
	}

	/*enter the core computation*/
	if (_numCPUThreads < 1) {
		_numCPUThreads = omp_get_num_procs();
	}
	omp_set_num_threads(_numCPUThreads);
#pragma omp parallel reduction(+:totalNumPairs)
	{
		ssize_t chunkSize;
		int row, col;
		int loRowRange, hiRowRange;
		int loColRange, hiColRange;
		int startColPerRow, endColPerRow;
		int rowStart, rowEnd, colStart, colEnd;
		ssize_t numPairsProcessed = 0;
		FloatType meanX, varX, x, prod;
		FloatType* __restrict__ vecX;
		FloatType* __restrict__ vecY;
		const int tid = omp_get_thread_num();
		const int nthreads = omp_get_num_threads();

		/*compute mean and variance*/
		chunkSize = (_numVectors + nthreads - 1) / nthreads;
		loRowRange = tid * chunkSize;
		hiRowRange = min(_numVectors, (tid + 1) * (int) chunkSize) - 1;

		vecX = _vectors + loRowRange * _vectorSizeAligned;
		for (row = loRowRange; row <= hiRowRange; ++row, vecX +=
				_vectorSizeAligned) {

			/*compute the mean*/
			meanX = 0;
#pragma vector aligned
#pragma simd reduction(+:meanX)
			for (int j = 0; j < _vectorSize; ++j) {
				meanX += vecX[j] * avg;
			}

			/*compute the variance*/
			varX = 0;
#pragma vector aligned
#pragma simd reduction(+:varX)
			for (int j = 0; j < _vectorSize; ++j) {
				x = vecX[j] - meanX;
				varX += x * x;
			}
			varX = 1 / sqrt(varX);

			/*normalize the data*/
#pragma vector aligned
#pragma simd
			for (int j = 0; j < _vectorSize; ++j) {
				x = vecX[j] - meanX;
				vecX[j] = x * varX;
			}
		}
	} /*#pragma omp paralle*/

	/*invoke the Intel MKL kernel: multithreads*/
	mygemm<FloatType>(CblasRowMajor, CblasNoTrans, CblasTrans, _numVectors, _numVectors, _vectorSize, 1, _vectors, _vectorSizeAligned, _vectors, _vectorSizeAligned, 0, _pearsonCorr, _numVectors);

	/*recored the system time*/
	etime = getSysTime();
	fprintf(stderr, "Overall time (%ld pairs): %f seconds\n", totalNumPairs,
			etime - stime);

#if 0
  for(int i = 0; i < _numVectors; ++i){
    for(int j = i; j < _numVectors; ++j){
      printf("%f\n", _pearsonCorr[i *_numVectors + j]);
    }
  }
#endif

}

#ifdef WITH_PHI
template<typename FloatType>
void PearsonRMKL<FloatType>::runSingleXeonPhi() {
	double stime, etime;

#ifdef VERBOSE
  fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

	/*set the number of threads on Xeon Phi*/
#pragma offload target(mic:_micIndex) inout(_numMICThreads)
	{
		/*set the number of threads*/
		if (_numMICThreads < 1) {
			_numMICThreads = omp_get_num_procs();
		}
		omp_set_num_threads (_numMICThreads);
	}
	fprintf(stderr, "number of threads: %d\n", _numMICThreads);

	/*allocate output buffer and align to the tile size*/
	_pearsonCorr = (FloatType*)mm_malloc((ssize_t)_numVectors * _numVectors * sizeof(FloatType), 64);
	if(!_pearsonCorr) {
		fprintf(stderr, "Memory allocation failed\n");
		exit(-1);
	}

	/*set the variables*/
  int signalVar, tileRowStart, tileRowEnd;
  int tileColStart, tileColEnd;
  ssize_t tileOffset, numElems, maxNumTilesPerPass;
  FloatType value;
	FloatType* __restrict__ vectors = _vectors;
	FloatType* __restrict__ prValues = _pearsonCorr;

	fprintf(stderr, "Start transfering data\n");
#pragma offload_transfer target(mic: _micIndex) \
		in(vectors: length(_numVectors * _vectorSizeAligned) PR_MKL_MIC_ALLOC) \
	  nocopy(prValues: length(_numVectors * _numVectors) PR_MKL_MIC_ALLOC)

	fprintf(stderr, "Start computing\n");
	/*record system time*/
	stime = getSysTime();

	/*compute mean and variance*/
	double t1 = getSysTime();
#pragma offload target(mic:_micIndex) nocopy(vectors: PR_MKL_MIC_REUSE)
	{
#pragma omp parallel
		{
			int row, col;
			int chunkSize, loRowRange, hiRowRange;
			FloatType x, y, r, meanX, varX;
			FloatType* __restrict__ vecX;
			const int tid = omp_get_thread_num();
			const int nthreads = omp_get_num_threads();
			const FloatType avg = 1.0 / (FloatType)_vectorSize;

			/*compute mean and variance*/
			chunkSize = (_numVectors + nthreads - 1) / nthreads;
			loRowRange = tid * chunkSize;
			hiRowRange = min(_numVectors, (tid + 1) * chunkSize);

			vecX = vectors + loRowRange * _vectorSizeAligned;
			for (row = loRowRange; row < hiRowRange; ++row, vecX +=
					_vectorSizeAligned) {

				/*compute the mean*/
				meanX = 0;
#pragma vector aligned
#pragma simd reduction(+:meanX)
				for (int j = 0; j < _vectorSize; ++j) {
					meanX += vecX[j] * avg;
				}

				/*compute the variance*/
				varX = 0;
#pragma vector aligned
#pragma simd reduction(+:varX)
				for(int j = 0; j < _vectorSize; ++j) {
					x = vecX[j] - meanX;
					varX += x * x;
				}
				varX = 1 / sqrt(varX);

				/*normalize the data*/
#pragma vector aligned
#pragma simd
				for(int j = 0; j < _vectorSize; ++j) {
					x = vecX[j] - meanX;
					vecX[j] = x * varX;
				}
			}
		}
	}
	double t2 = getSysTime();
	fprintf(stderr, "time for transformation: %f seconds\n", t2 - t1);

	/*invoke the GEMM core*/
#pragma offload target(mic:_micIndex) \
	nocopy(vectors: PR_MKL_MIC_REUSE) \
	out(prValues: length(_numVectors * _numVectors) PR_MKL_MIC_REUSE)
	{
		mygemm<FloatType>(CblasRowMajor, CblasNoTrans, CblasTrans, _numVectors, _numVectors, _vectorSize, 1.0, vectors, _vectorSizeAligned, vectors, _vectorSizeAligned, 0, prValues, _numVectors);
	}

	/*release memory on the Xeon Phi*/
#pragma offload_transfer target(mic: _micIndex) \
		nocopy(vectors : PR_MKL_MIC_FREE) \
		nocopy(prValues: PR_MKL_MIC_FREE)

	/*recored the system time*/
	etime = getSysTime();
	fprintf(stderr, "Overall time: %f seconds\n", etime - stime);

#if 0
  for(int i = 0; i < _numVectors; ++i){
    for(int j = i; j < _numVectors; ++j){
      printf("%f\n", _pearsonCorr[i *_numVectors + j]);
    }
  }
#endif
}
#endif	/*with phi*/

#ifdef WITH_MPI
template<typename FloatType>
void PearsonRMKL<FloatType>::runMPICPU() {
	fprintf(stderr, "function %s is not supported by Intel MKL\n", __FUNCTION__);
}

#ifdef WITH_PHI
template<typename FloatType>
void PearsonRMKL<FloatType>::runMPIXeonPhi() {
	fprintf(stderr, "function %s is not supported by Intel MKL\n", __FUNCTION__);
}
#endif	/*with phi*/
#endif	/*with mpi*/

#endif /* PEARSONR_MKL_HPP_ */
