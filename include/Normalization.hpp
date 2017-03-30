/*
 *  Normalization.hpp
 * 
 *  Created on: Nov 30, 2016
 *  Author: Liu, Yongchao
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 *  URL: www.liuyc.org
 */

#ifndef NORMALIZATION_HPP_
#define NORMALIZATION_HPP_

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
#include <limits>
using namespace std;
#pragma once

/*Min-max scaling normalization approach, which is typically
 * done via the following equation
 * X_{norm} = \frac{X-X_{min}}{X_{max}-X_{min}}
 */
template <typename FloatType, typename IntType=int>
class MinMaxScaling
{
public:
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void operator()(FloatType* __restrict__ mat, const IntType width, const IntType widthAligned, const IntType height, const bool globalNorm);

#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void operator()(const FloatType* __restrict__ inMatrix, FloatType* __restrict__ outMatrix, const IntType width, const IntType widthAligned, const IntType height, const bool globalNorm);
};

template<typename FloatType, typename IntType>
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
void MinMaxScaling<FloatType, IntType>::operator()(FloatType* __restrict__ matrix, const IntType width, const IntType widthAligned, const IntType height, const bool globalNorm)
{
	FloatType v, v2, minV, maxV, *mat;

	if(globalNorm == true){/*global normalization across the whole matrix*/
		minV = numeric_limits<FloatType>.max();
		maxV = numeric_limits<FloatType>.min();

		/*get the global maximum and minimum values*/
		mat = matrix;
		for(IntType row = 0; row < height; ++row){
			for(IntType col = 0; col < height; ++col){
				v = mat[col];
				minV = min(minV, v);
				maxV = max(maxV, v);
			}
			/*move to the next row*/
			mat += widthAligned;
		}

		/*compute the normalized values*/
		v = maxV - minV;
		if(v == 0){
			v = numeric_limits<FloatType>.min();
		}
		v2 = minV / v;
		v = 1.0 / v;
		mat = matrix;
		for(IntType row = 0; row < height; ++row){
#pragma simd
#pragma vector aligned
			for(IntType col = 0; col < width; ++col){
				mat[col] = mat[col] * v - v2;
			}
			/*move the next row*/
			mat += widthAligned;
		}
	}else{/*local normalization within each vector*/

		minV = numeric_limits<FloatType>.max();
		maxV = numeric_limits<FloatType>.min();

		mat = matrix;
		for(IntType row = 0; row < height; ++row){
			/*compute the local minimum and maximum values*/
			for(IntType col = 0; col < width; ++col){
				v = mat[col];
				minV = min(minV, v);
				maxV = max(maxV, v);
			}

			/*compute the normalized value*/
			v = maxV - minV;
			if(v == 0){
				v = numeric_limits<FloatType>.min();
			}
			v2 = minV / v;
			v = 1.0 / v;
#pragma vector aligned
#pragma simd
			for(IntType col = 0; col < width; ++col){
				mat[col] = mat[col] * v - v2;
			}

			/*move to the next row*/
			mat += widthAligned;
		}
	}
}

template<typename FloatType, typename IntType>
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
void MinMaxScaling<FloatType, IntType>::operator()(const FloatType* __restrict__ inMatrix, FloatType* __restrict__ outMatrix, const IntType width, const IntType widthAligned, const IntType height, const bool globalNorm)
{
	FloatType v, v2, minV, maxV, *imat, *omat;
	if(globalNorm == true){
		/*compute global minimum and maximum values*/
		imat = inMatrix;
		minV = numeric_limits<FloatType>.max();
		maxV = numeric_limits<FloatType>.min();
		for(IntType row = 0; row < height; ++row){
			for(IntType col = 0; col < width; ++col){
        v = imat[col];
        minV = min(minV, v);
        maxV = max(maxV, v);
			}

			/*move to the next row*/
			imat += widthAligned;
		}

		/*compute the normalized value*/
		v = maxV - minV;
		if(v == 0){
			v = numeric_limits<FloatType>.min();
		}
		v2 = minV / v;
		v = 1.0 / v;
		imat = inMatrix;
		omat = outMatrix;
    for(IntType row = 0; row < height; ++row){
#pragma simd
#pragma vector aligned
      for(IntType col = 0; col < width; ++col){
				omat[col] = imat[col] * v - v2;
      }

      /*move to the next row*/
      imat += widthAligned;
			omat += widthAligned;
    }
	}else{/*local normalization*/
 	 for(IntType row = 0; row < height; ++row){

   	 	/*compute the minimum and maximum values*/
			imat = inMatrix;
			minV = numeric_limits<FloatType>.max();
			maxV = numeric_limits<FloatType>.min();
    	for(IntType col = 0; col < width; ++col){
      	v = imat[col];
      	minV = min(minV, v);
      	maxV = max(maxV, v);
    	}

    	/*compute the normalized value*/
    	v = maxV - minV;
   	 	if(v == 0){
     		v = numeric_limits<FloatType>.min();
    	}
			v2 = minV / v;
    	v = 1.0 / v;
#pragma simd
#pragma vector aligned
    	for(IntType col = 0; col < width; ++col){
      	omat[col] = imat[col] * v - v2;
    	}
  	}
	}
}

/*Z-score normalization approach is computed as follows
 *
 *	X_{norm} = \frac{X-\mu}{\sigma}, where \mu is the mean of the population
 *	and \sigma is the standard deviation of the population.
 */
template<typename FloatType, typename IntType = int>
class Zscore
{
public:
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void operator()(FloatType* __restrict__ matrix, const IntType width, const IntType widthAligned, const IntType height, const bool globalNorm);

#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
	void operator()(const FloatType* __restrict__ inMatrix, FloatType* __restrict__ outMatrix, const IntType width, const IntType widthAligned, const IntType height, const bool globalNorm);
};

template<typename FloatType, typename IntType>
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
void Zscore<FloatType, IntType>::operator()(FloatType* __restrict__ matrix, const IntType width, const IntType widthAligned, const IntType height, const bool globalNorm)
{
	FloatType v, mean, var, *mat;
	if(globalNorm == true){
    /*compute the mean*/
    mat = matrix;
		mean = 0;
    for(IntType row = 0; row < height; ++row){
			v = 0;
#pragma vector aligned
#pragma simd reduction(+:v)
      for(IntType col = 0; col < width; ++col){
      	v += mat[col];
      }
			mean += v / width;

			/*move to the next row*/
			mat += widthAligned;
		}
		mean /= height;

  	/*compute the standard deviation*/
   	var = 0;
		mat = matrix;
		for(IntType row = 0; row < height; ++row){
#pragma vector aligned
#pragma simd reduction(+:var)
      for(IntType col = 0; col < width; ++col){
        v = mat[col] - mean;
        var += v * v;
      }

			/*move to the next row*/
			mat += widthAligned;
		}
  	var = sqrt(var / (width * height > 1 ? width * height - 1 : 1));

  	/*compute the standard score*/
   	var = 1.0 / var;
  	v = mean * var;
		mat = matrix;
		for(IntType row = 0; row < height; ++row){
#pragma vector aligned
#pragma simd
      for(IntType col = 0; col < width; ++col){
        mat[col] = mat[col] * var - v;
      }

      /*move to the next row*/
      mat += widthAligned;
    }
	}else{
		mat = matrix;
		for(IntType row = 0; row < height; ++row){
			/*compute the mean*/
			mean = 0;
#pragma vector aligned
#pragma simd reduction(+:mean)
			for(IntType col = 0; col < width; ++col){
				mean += mat[col];
			}
			mean /= width;

			/*compute the standard deviation*/
			var = 0;
#pragma vector aligned
#pragma simd reduction(+:var)
			for(IntType col = 0; col < width; ++col){
				v = mat[col] - mean;
				var += v * v;
			}
			var = sqrt(var / (width > 1 ? width - 1 : 1));

			/*compute the standard score*/
			var = 1.0 / var;
			v = mean * var;
#pragma vector aligned
#pragma simd
			for(IntType col = 0; col < width; ++col){
				mat[col] = mat[col] * var - v;
			}

			/*move to the next row*/
			mat += widthAligned;
		}
	}
}

template<typename FloatType, typename IntType>
#ifdef WITH_PHI
  __attribute__((target(mic)))
#endif
void Zscore<FloatType, IntType>::operator()(const FloatType* __restrict__ inMatrix, FloatType* __restrict__ outMatrix, const IntType width, const IntType widthAligned, const IntType height, const bool globalNorm)
{
	FloatType v, mean, var, *imat, *omat;
	if(globalNorm == true){
    /*compute the mean*/
		mean = 0;
		imat = inMatrix;
    for(IntType row = 0; row < height; ++row){
			v = 0;
#pragma vector aligned
#pragma simd reduction(+:v)
      for(IntType col = 0; col < width; ++col){
      	v += imat[col];
      }
			mean += v / width;

			/*move to the next row*/
			imat += widthAligned;
		}
		mean /= height;

  	/*compute the standard deviation*/
   	var = 0;
		imat = inMatrix;
		for(IntType row = 0; row < height; ++row){
#pragma vector aligned
#pragma simd reduction(+:var)
      for(IntType col = 0; col < width; ++col){
        v = imat[col] - mean;
        var += v * v;
      }

			/*move to the next row*/
			imat += widthAligned;
		}
  	var = sqrt(var / (width * height > 1 ? width * height - 1 : 1));

  	/*compute the standard score*/
   	var = 1.0 / var;
  	v = mean * var;
		imat = inMatrix;
		omat = outMatrix;
		for(IntType row = 0; row < height; ++row){
#pragma vector aligned
#pragma simd
      for(IntType col = 0; col < width; ++col){
        omat[col] = imat[col] * var - v;
      }

      /*move to the next row*/
     	imat += widthAligned;
     	omat += widthAligned;
    }
	}else{
		imat = inMatrix;
		omat = outMatrix;
		for(IntType row = 0; row < height; ++row){
			/*compute the mean*/
			mean = 0;
#pragma vector aligned
#pragma simd reduction(+:mean)
			for(IntType col = 0; col < width; ++col){
				mean += imat[col];
			}
			mean /= width;

			/*compute the standard deviation*/
			var = 0;
#pragma vector aligned
#pragma simd reduction(+:var)
			for(IntType col = 0; col < width; ++col){
				v = imat[col] - mean;
				var += v * v;
			}
			var = sqrt(var / (width > 1 ? width - 1 : 1));

			/*compute the standard score*/
			var = 1.0 / var;
			v = mean * var;
#pragma vector aligned
#pragma simd
			for(IntType col = 0; col < width; ++col){
				omat[col] = imat[col] * var - v;
			}

			/*move to the next row*/
			imat += widthAligned;
			omat += widthAligned;
		}
	}
}
#endif

