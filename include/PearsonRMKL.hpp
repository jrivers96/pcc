#ifndef PEARSONR_MKL_HPP_
#define PEARSONR_MKL_HPP_
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <omp.h>
#include <vector>
#include <queue>
#include <list>
#include <typeinfo>
//conversion of float to string with precision
#include <iomanip> // setprecision
#include <sstream> // stringstream
//for creating the counts map
#include <map>
//create output file
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>

using namespace std;
//using namespace mmap_allocator_namespace;
#pragma once

#ifdef WITH_MPI
#include <mpi.h>
#endif	/*with mpi*/

/*Intel MKL*/
#include <mkl.h>
#include "mkl_vml.h"

/*Intel IPP*/
#include <ipp.h>

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

/*
namespace mmap_allocator_namespace
{
        template <typename T>
        class mmap_allocator: public std::allocator<T>
        {
                public:
                typedef size_t size_type;
                typedef T* pointer;
                typedef const T* const_pointer;

                template<typename _Tp1>
                struct rebind
                {
                        typedef mmap_allocator<_Tp1> other;
                };

                pointer allocate(size_type n, const void *hint=0)
                {
                        fprintf(stderr, "Alloc %d bytes.\n", n*sizeof(T));
                        //return std::allocator<T>::allocate(n, hint);
                        return  mm_malloc((ssize_t)* n *sizeof(T), 64);
                }

                void deallocate(pointer p, size_type n)
                {
                        fprintf(stderr, "Dealloc %d bytes (%p).\n", n * sizeof(T), p);
                        //return std::allocator<T>::deallocate(p, n);
                        return mm_free(p);
                }
                mmap_allocator() throw(): std::allocator<T>() { fprintf(stderr, "Hello allocator!\n"); }
                mmap_allocator(const mmap_allocator &a) throw(): std::allocator<T>(a) { }
                template <class U>                    
                mmap_allocator(const mmap_allocator<U> &a) throw(): std::allocator<T>(a) { }
                ~mmap_allocator() throw() { }
        };
}
*/

template<typename FloatType>
void PearsonRMKL<FloatType>::runMultiThreaded() {
	double stime, etime;
	ssize_t totalNumPairs = 0;
	const double avg = 1.0 / (double)_vectorSize;

#ifdef VERBOSE
  fprintf(stderr, "execute function %s\n", __FUNCTION__);
#endif

  /*record system time*/
  stime = getSysTime();

  /*allocate space*/
  /*_pearsonCorr = (FloatType*) mm_malloc(
      (ssize_t) _numVectors * _numVectors * sizeof(FloatType), 64);
  if (!_pearsonCorr) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }
 */

 int desiredBatch = 250;
 ssize_t numBatches = ceil((double)_numVectors / (double)desiredBatch);
 fprintf(stderr, "numVectors: %u \n", _numVectors);
 fprintf(stderr, "desiredBatch: %u \n", desiredBatch);
 fprintf(stderr, "numBatched: %u \n", numBatches);
 
 ssize_t *endVec = (ssize_t*) mm_malloc(
      (ssize_t) numBatches * sizeof(ssize_t), 64);
  if (!endVec) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }
 ssize_t *startVec = (ssize_t*) mm_malloc(
      (ssize_t) numBatches * sizeof(ssize_t), 64);
  if (!startVec) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }
 startVec[0]=0; 
 endVec[0]= min(_numVectors-1, desiredBatch-1);
 for(int xx = 1; xx < numBatches; ++xx) {
   startVec[xx] = startVec[xx-1] +  (desiredBatch); 
   if(startVec[xx] + desiredBatch >= _numVectors){
     endVec[xx]=(ssize_t)_numVectors-1;
    }else{
     endVec[xx]=(ssize_t)(startVec[xx] +(desiredBatch-1));
    }
  }
  /*allocate space*/
size_t batchSize = (size_t)_numVectors * (size_t)desiredBatch;
 _pearsonCorr = (FloatType*) mm_malloc(
      (ssize_t) batchSize * sizeof(FloatType), 64);
  if (!_pearsonCorr) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

size_t dataSize = (_numVectors * _vectorSizeAligned); 
 FloatType *countMat = (FloatType*) mm_malloc(
      (ssize_t) dataSize * sizeof(FloatType), 64);
  if (!countMat) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

/*FloatType *columnMeans = (FloatType*) mm_malloc(
      (ssize_t) _numVectors * sizeof(FloatType), 64);
  if (!columnMeans) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }
*/

FloatType *totCounts = (FloatType*) mm_malloc(
      (ssize_t) batchSize * sizeof(FloatType), 64);
  if (!totCounts) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

FloatType *tempX = (FloatType*) mm_malloc(
      (ssize_t) batchSize * sizeof(FloatType), 64);
  if (!tempX) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

FloatType *tempY = (FloatType*) mm_malloc(
      (ssize_t) batchSize * sizeof(FloatType), 64);
  if (!tempY) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

FloatType *meanX = (FloatType*) mm_malloc(
      (ssize_t) batchSize * sizeof(FloatType), 64);
  if (!meanX) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

FloatType *meanY = (FloatType*) mm_malloc(
      (ssize_t) batchSize * sizeof(FloatType), 64);
  if (!meanY) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

FloatType *meanXY = (FloatType*) mm_malloc(
      (ssize_t) batchSize * sizeof(FloatType), 64);
  if (!meanXY) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }


 FloatType *squared = (FloatType*) mm_malloc(
      (ssize_t) (dataSize) * sizeof(FloatType), 64);
  if (!squared) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

 FloatType *power = (FloatType*) mm_malloc(
      (ssize_t) (dataSize) * sizeof(FloatType), 64);
  if (!power) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

#pragma vector aligned
#pragma simd
for(size_t ii = 0; ii<dataSize; ii++){
   power[ii] = 2.0;
}

#pragma vector aligned
#pragma simd
for(size_t ii = 0; ii<dataSize; ii++){
   countMat[ii] = (_vectors[ii]==0.0) ? 0.0:1.0;
}
/*
#pragma vector aligned
#pragma simd
//for(size_t xx = 0;xx<_vectorSizeAligned;xx++){
for(size_t xx = 0; xx<_numVectors; xx++){
   colCount[xx] = 0.0;	
   for(size_t ii = 0; ii<_vectorSizeAligned; ii++){
   	colCount[xx] += countMat[ii+ xx*_vectorSizeAligned];
}
}
*/
size_t *idxSave = (size_t*) mm_malloc(
      (ssize_t) (batchSize) * sizeof(size_t), 64);
  if (!idxSave) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }
size_t numNeighbors = 100;
size_t neighborSize = numNeighbors*desiredBatch;
FloatType *neighborVal = (FloatType*) mm_malloc(
      (ssize_t) (neighborSize) * sizeof(FloatType), 64);
  if (!neighborVal) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

size_t *neighborIdx = (size_t*) mm_malloc(
      (ssize_t) (neighborSize) * sizeof(size_t), 64);
  if (!neighborIdx) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

#ifdef DETAILDEBUG
for(size_t ii=0; ii < dataSize; ++ii)
{
  fprintf(stderr,"%f,", countMat[ii]);
}
fprintf(stderr,"\n" );
#endif

size_t vSize = (_vectorSize+1) * 2001;

/*enter the core computation*/
    if (_numCPUThreads < 1) {
                _numCPUThreads = omp_get_num_procs();
        }
        omp_set_num_threads(_numCPUThreads);

 size_t *vv = (size_t*) mm_malloc(
      (ssize_t) (16 * vSize) * sizeof(size_t), 64);
  if (!vv) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }
for(size_t ii=0;ii<(vSize*16);ii++)
{
 vv[ii]=0;
}
size_t *vvOut = (size_t*) mm_malloc(
      (ssize_t) (vSize) * sizeof(size_t), 64);
  if (!vvOut) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(-1);
  }

for(size_t ii=0;ii<(vSize);ii++)
{
 vvOut[ii]=0;
}

fprintf(stderr,"num_threads:%d\n", _numCPUThreads );

/*invoke the Intel MKL kernel: multithreads*/
//mygemm<FloatType>(CblasRowMajor, CblasNoTrans, CblasTrans, _numVectors, _numVectors, _vectorSize, 1, _vectors, _vectorSizeAligned, _vectors, _vectorSizeAligned, 0, _pearsonCorr, _numVectors);

double ptime, pltime;
ptime = getSysTime();

///create the squared matrix for the entire data set to be used in normalization
if(sizeof(FloatType) == 4){
vsPow(dataSize, (float *)_vectors, (float *)power, (float *) squared);
}else{
vdPow(dataSize, (double*)_vectors, (double*)power, (double *) squared);
}
mm_free(power);

pltime = getSysTime();
fprintf(stderr, "vdPow: %f seconds\n", pltime - ptime);

FloatType* __restrict__ vecX;
FloatType* __restrict__ vecXSquared;
FloatType* __restrict__ vecCountMat;
FloatType* __restrict__ pccLarge;

double ltime, letime;
double mtime, metime;
double itime, ietime;
stringstream stream;

for(int xx = 0; xx < numBatches; ++xx)
 {
  ltime = getSysTime();
  size_t mSize = endVec[xx] - startVec[xx] + 1;

  fprintf(stderr, "mSize: %u \n", mSize);
  fprintf(stderr, "vectorSize: %u \n", _vectorSize);
  fprintf(stderr, "_vectorSizeAligned: %u \n", _vectorSizeAligned);
  fprintf(stderr, "_startVec: %u \n", startVec[xx]);
  fprintf(stderr, "_endVec: %u \n", endVec[xx]);

  vecX        = _vectors + startVec[xx] * _vectorSizeAligned;
  vecXSquared = squared  + startVec[xx] * _vectorSizeAligned;
  vecCountMat = countMat + startVec[xx] * _vectorSizeAligned;

  //numerator
  mygemm<FloatType>(CblasRowMajor, CblasNoTrans, CblasTrans, mSize, _numVectors, _vectorSize, 1, vecX, _vectorSizeAligned, _vectors, _vectorSizeAligned, 0, _pearsonCorr, _numVectors);

  mygemm<FloatType>(CblasRowMajor, CblasNoTrans, CblasTrans, mSize, _numVectors, _vectorSize, 1, vecX, _vectorSizeAligned, countMat, _vectorSizeAligned, 0, tempX, _numVectors);
 
  mygemm<FloatType>(CblasRowMajor, CblasNoTrans, CblasTrans, mSize, _numVectors, _vectorSize, 1, vecCountMat, _vectorSizeAligned, _vectors, _vectorSizeAligned, 0, tempY, _numVectors);

  //generate the counts
  mygemm<FloatType>(CblasRowMajor, CblasNoTrans, CblasTrans, mSize, _numVectors, _vectorSize, 1, vecCountMat, _vectorSizeAligned, countMat, _vectorSizeAligned, 0, totCounts, _numVectors);
  
  size_t flatSize =  mSize*_numVectors;
 
  fprintf(stderr, "begin element by element:%u \n", flatSize);
 
 if(sizeof(FloatType) == 4){
   vsMul(flatSize, (float *)tempX,(float *)tempX, (float*)meanX);
  }else{
   vdMul(flatSize, (double *)tempX,(double *)tempX, (double*)meanX);
  }

 if(sizeof(FloatType) == 4){
   vsMul(flatSize, (float *)tempY,(float *)tempY, (float*)meanY);
  }else{
   vdMul(flatSize, (double *)tempY,(double *)tempY, (double*)meanY);
  }

 if(sizeof(FloatType) == 4){
   vsMul(flatSize, (float *)tempX,(float *)tempY, (float*)meanXY);
  }else{
   vdMul(flatSize, (double *)tempX,(double *)tempY, (double*)meanXY);
  }
 
 if(sizeof(FloatType) == 4){
   vsMul(flatSize, (float *)totCounts,(float *)_pearsonCorr, (float *)_pearsonCorr);
  }else{
   vdMul(flatSize, (double *)totCounts,(double *)_pearsonCorr, (double*)_pearsonCorr);
  }

 if(sizeof(FloatType) == 4){
   vsSub(flatSize, (float *) _pearsonCorr, (float *)meanXY,(float *)_pearsonCorr);
  }else{
   vdSub(flatSize, (double *)_pearsonCorr, (double *)meanXY,(double *)_pearsonCorr);
  }

  fprintf(stderr, "numerator \n");

  //left denominator
  mygemm<FloatType>(CblasRowMajor, CblasNoTrans, CblasTrans, mSize, _numVectors, _vectorSize, 1, vecXSquared, _vectorSizeAligned, countMat, _vectorSizeAligned, 0, tempX, _numVectors);

 if(sizeof(FloatType) == 4){
   vsMul(flatSize, (float *)tempX, (float *)totCounts,(float *)tempX);
  }else{
   vdMul(flatSize, (double *)tempX, (double *)totCounts,(double *)tempX);
  }

 if(sizeof(FloatType) == 4){
   vsSub(flatSize, (float *)tempX, (float *)meanX,(float *)tempX);
  }else{
   vdSub(flatSize, (double *)tempX, (double *)meanX,(double *)tempX);
  }


  fprintf(stderr, "left den \n");

  //right denominator
  mygemm<FloatType>(CblasRowMajor, CblasNoTrans, CblasTrans, mSize, _numVectors, _vectorSize, 1, vecCountMat, _vectorSizeAligned, squared, _vectorSizeAligned, 0, tempY, _numVectors);
  fprintf(stderr, "right den \n");

 if(sizeof(FloatType) == 4){
   vsMul(flatSize, (float *)tempY, (float *)totCounts,(float *)tempY);
  }else{
   vdMul(flatSize, (double *)tempY, (double *)totCounts,(double *)tempY);
  }

 if(sizeof(FloatType) == 4){
   vsSub(flatSize, (float *)tempY, (float *)meanY,(float *)tempY);
  }else{
   vdSub(flatSize, (double *)tempY, (double *)meanY,(double *)tempY);
  }

  size_t N;
  long ix;
  //combine left and right denominator

//#define DETAILDEBUG
#ifdef DETAILDEBUG
#endif

if(sizeof(FloatType) == 4){
vsSqrt(flatSize, (float *)tempX,(float *)tempX);
}else{
vdSqrt(flatSize, (double *)tempX,(double *)tempX);
}

if(sizeof(FloatType) == 4){
vsSqrt(flatSize, (float *)tempY,(float *)tempY);
}else{
vdSqrt(flatSize, (double *)tempY,(double *)tempY);
}

pccLarge = _pearsonCorr;

for(size_t ii=0; ii < 2; ++ii)
{
  fprintf(stderr,"tempX:%f,", tempX[ii]);
  fprintf(stderr,"tempY:%f,", tempY[ii]);

}

fprintf(stderr, "begin counts \n");
#pragma omp parallel for shared(flatSize, tempX, tempY, pccLarge) default(none)
  for(size_t j = 0; j < flatSize; ++j){
       tempY[j] = pccLarge[j]/((tempX[j])*(tempY[j]));
       tempY[j] *=1000.0;  
}

for(size_t ii=0; ii < 2; ++ii)
{
  fprintf(stderr,"Full PCC:%f,", tempY[ii]);
}


#ifdef DETAILDEBUG
fprintf(stderr,"Counts:\n" );
fprintf(stderr,"%f,", tempX[40]);
fprintf(stderr,"%f,", tempX[80]);
//fprintf(stderr,"%f",  tempX[1853047522]);
fprintf(stderr,"\n" );

fprintf(stderr,"Original pearson:\n" );
fprintf(stderr,"%f,", _pearsonCorr[40]);
fprintf(stderr,"%f,", _pearsonCorr[80]);
//fprintf(stderr,"%f,", _pearsonCorr[1853047522]);

for(size_t ii=0; ii < flatSize; ++ii)
{
fprintf(stderr,"%f,", _pearsonCorr[ii]);
}
  fprintf(stderr,"\n" );
  fprintf(stderr, "counts \n");
  //vsSqrt((MKL_INT)flatSize,(float *)tempY,(float *)tempY)(
  fprintf(stderr, "NORMALIZE FACTOR: %f\n", tempY[40]);
  fprintf(stderr, "NORMALIZE FACTOR: %f\n", tempY[80]);
  //fprintf(stderr, "NORMALIZE FACTOR: %f\n", tempY[1853047522]);
  fprintf(stderr, "AFTER NORMALIZE: %f\n", _pearsonCorr[40]/tempY[40]);
  fprintf(stderr, "AFTER NORMALIZE: %f\n", _pearsonCorr[80]/tempY[80]);
  //fprintf(stderr, "AFTER NORMALIZE: %f\n", _pearsonCorr[1853047522]/tempY[1853047522]);
#endif
  fprintf(stderr, "Begin normalize\n");
  letime = getSysTime();
  fprintf(stderr, "Batch time: %f seconds\n", letime - ltime);
/*
for(size_t ii=0; ii < _numVectors; ++ii)
{
fprintf(stderr,"%d:%f,", ii, _pearsonCorr[ii]);
}

for(size_t ii=0; ii < _numVectors; ++ii)
{
fprintf(stderr,"%d:%f,", ii, tempY[ii]);
}
*/
 //The values below range from [0 2000] * 400 + 400 = max(800400)
 pccLarge = _pearsonCorr;
 const size_t vecSize = _vectorSize;

/* 
  #pragma vector aligned
  #pragma simd
  for(size_t j = 0; j < flatSize; ++j){
       pccLarge[j] =  (pccLarge[j]/tempY[j])*1000.0 + 1000.0;
  }

 #pragma vector aligned
 #pragma simd
  for(size_t j = 0; j < flatSize; ++j){
       idxSave[j] = _pearsonCorr[j] * vecSize + tempX[j];
 }
*/

/*int *idxx = &idxSave[0];
#pragma omp parallel for shared(flatSize, tempY, idxSave, idxx) default(none)
#pragma vector aligned
#pragma simd
  for(size_t j = 0; j < flatSize; ++j){
       idxx[j] = tempY[j];
  }
*/

#define NEIGHBORS 
#ifdef NEIGHBORS

ltime = getSysTime();

ofstream myfile;
myfile.open("neighbors.csv", std::ofstream::out |  std::ofstream::app);
fprintf(stderr, "Begin neighbor sort\n");

size_t pneighbor = min((size_t)numNeighbors,(size_t)_numVectors);
size_t numVectors = _numVectors;

#pragma omp parallel for shared( mSize, tempY,totCounts, pneighbor, numVectors, neighborIdx, neighborVal,stderr) default(none)
for(size_t j = 0; j < mSize;j++)
{
  size_t begin  = j*numVectors;
  size_t beginN = j*pneighbor;
  //size_t end = j*_numVectors + _numVectors - 1;
  std::priority_queue<std::pair<int, int>> q; 
 
  for (size_t ii = 0; ii < numVectors; ++ii) {
    q.push(std::pair<int, int>((int)tempY[begin + ii], ii));
  }
  size_t cc = 0;
  for(size_t ii = 0; ii < pneighbor; ii++){
   size_t nnid =  q.top().second;
   size_t countmatch = totCounts[begin + nnid]; 
   while((countmatch < 5) && (cc < numVectors) )
   {
    q.pop();
    nnid =  q.top().second;
    countmatch = totCounts[begin + nnid]; 
    cc+=1;
    if(cc>numVectors)
    {
     fprintf(stderr, "cc:%u\n",cc);
    }
   }
   neighborVal[beginN+ii]  =(FloatType)q.top().first/((FloatType)1000.0);
   neighborIdx[beginN+ii]  =  q.top().second;
    q.pop(); 
 }
 q = priority_queue<std::pair<int,int>>();
}

for(size_t j = 0; j < mSize;j++)
{
  
  size_t beginO = j*numVectors;
  size_t begin = j*pneighbor;
  for(size_t nn = 0; nn < pneighbor; nn++)
   {
     FloatType printpcc = neighborVal[begin + nn];
     size_t neighborid   = neighborIdx[begin + nn];
     size_t printcount   = totCounts[beginO + neighborid];
     if(printpcc > 1.01 || printpcc < -1.01)
	continue;
     myfile << std::setprecision(6) << (startVec[xx]+j) << " " << (float)printpcc << " " << neighborid << " " << printcount << "\n";
   }
}

letime = getSysTime();
fprintf(stderr, "Exit neighbor sort: %f seconds\n", letime - ltime);
myfile.close(); 
#endif

mtime = getSysTime();

//#pragma omp parallel for shared(flatSize, tempX, tempY, idxSave, vecSize) default(none)
//#pragma vector aligned
//#pragma simd
 
#pragma omp parallel for shared(flatSize, totCounts, tempY, idxSave, vecSize) default(none)
 for(size_t j = 0; j < flatSize; ++j){
       idxSave[j] =  (tempY[j] +(FloatType)1000.0)*vecSize + totCounts[j];
  }
 metime = getSysTime();
 fprintf(stderr, "Create index time: %f seconds\n", metime - mtime);

 /*for(int xx = 0; xx < flatSize; xx++)
 {
 if(tempY[xx] > vSize)
        fprintf(stderr, "tempY[%d]:%f \n", xx, tempY[xx] );
 }
 */
 mtime = getSysTime();
 fprintf(stderr, "VSize: %lu \n", vSize );
 //#pragma omp parallel for reduction(+:vv) 
size_t numtt = 16;//omp_get_num_threads(); 
#pragma omp parallel for shared(flatSize, vSize, stderr,numtt, idxSave,vv) default(none)
for(size_t j = 0; j < flatSize; ++j){
   size_t ixx =  idxSave[j]; 
   const size_t tid = omp_get_thread_num(); 
if((ixx >= vSize) || (isinf(ixx)!= 0) || (isnan(ixx)!=0) ){
   /*fprintf(stderr, "[j:%d]ixx: %lu \n", j, ixx );
   fprintf(stderr, "pearsonCorr: %f \n", _pearsonCorr[j]);
   fprintf(stderr, "tempX: %f \n",tempX[j]);
   fprintf(stderr, "tempY: %f \n",tempY[j]);
   */
   continue;
 }
  size_t idxx = tid*(vSize) + ixx; 
if(idxx >=(numtt*vSize))
{
fprintf(stderr, "num threads: %d tid: %lu \n", numtt, tid );
//fprintf(stderr, "vv error index: %lu \n", idxx );
continue;
}
vv[idxx]+=1;
}

 metime = getSysTime();
 fprintf(stderr, "Map time: %f seconds\n", metime - mtime);
 }
/*recored the system time*/
etime = getSysTime();
fprintf(stderr, "Overall time (%ld pairs): %f seconds\n", totalNumPairs,
	etime - stime);

int totalThreads = _numCPUThreads;

fprintf(stderr, "Start thread reduction step totalThreads:%d \n", totalThreads);
for(size_t ii = 0; ii < vSize; ii++)
{
vvOut[ii]=0;
for(size_t jj = 0; jj < totalThreads; jj++)
{
  //vvOut[ii] = vvOut[ii] + vv[jj*(vSize-1) + ii];
  vvOut[ii] = vvOut[ii] + vv[jj*(vSize) + ii];

}
}

fprintf(stderr, "Completed thread reduction step \n");
#if 1
ofstream myfile;
myfile.open ("countTable.csv");
int nPrint;
double pcc;
int count;
//2000 are pearson values from -1 to 1 in three digits. 
for(size_t ii=0; ii < vSize; ii++)
{
   nPrint = (ii % (_vectorSize+1));
   pcc    = ((((double)ii -(double)_vectorSize-1)/((double)_vectorSize-1)) - 1000.0);
   double printpcc =double(floor(pcc)/1000.0);
   if(printpcc>1.0)
	printpcc=1.0;
   
   if(printpcc < -1.0)
     printpcc = -1.0;
  count  = vvOut[ii];
  
   if(count == 0)
    continue; 
   myfile << std::setprecision(6) << nPrint << " " << setw(3) << printpcc << " " << std::setprecision(6) << count << "\n";
}
myfile.close();
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
	//				meanX += vecX[j] * 1.0/columnCount[j];
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
					//vecX[j] = x * varX;
					vecX[j] = x;
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
