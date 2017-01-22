#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

#include "Shared.h"


__device__ Limb MultiplyHighHelper(Limb a, Limb b)
{
	//return static_cast<Limb>((static_cast<DoubleLimb>(a) * b) >> LIMB_BITS);
	return __umulhi(a, b);
}


__device__ Limb AddFullHelper(Limb &dest, Limb a, Limb b, Limb c)
{
	DoubleLimb result = a;
	result += b;
	result += c;
	dest = static_cast<Limb>(result);
	return static_cast<Limb>(result >> 32);
}


__device__ Limb SubFullHelper(Limb &dest, Limb a, Limb b, Limb c)
{
	DoubleLimb result = a;
	result -= b;
	result -= c;
	dest = static_cast<Limb>(result);
	return static_cast<Limb>(result >> 32) & 1;
}


template <unsigned Limbs>
__device__ int BigCompareN(const Limb left[Limbs], const Limb right[Limbs])
{
	Limb borrow = 0;
	Limb notEqual = 0;

	for (unsigned x = 0; x < Limbs; ++x)
	{
		DoubleLimb current = left[x];
		current -= borrow;
		current -= right[x];
		notEqual |= static_cast<Limb>(current);

		borrow = static_cast<Limb>(current >> 32) & 1;
	}

	if (borrow)
	{
		return -1;
	}
	else if (notEqual)
	{
		return 1;
	}
	else
	{
		return 0;
	}
}


template <unsigned Limbs>
__device__ Limb BigSubtractN(Limb dest[Limbs], const Limb left[Limbs], const Limb right[Limbs], Limb borrow)
{
	for (unsigned x = 0; x < Limbs; ++x)
	{
		DoubleLimb current = left[x];
		current -= borrow;
		current -= right[x];
		dest[x] = static_cast<Limb>(current);

		borrow = static_cast<Limb>(current >> 32) & 1;
	}

	return borrow;
}


template <unsigned LeftLimbs, unsigned RightLimbs>
__device__ void BigMultiplyMN(Limb dest[LeftLimbs + RightLimbs], const Limb left[LeftLimbs], const Limb right[RightLimbs])
{
	static_assert(LeftLimbs > 0, "invalid LeftLimbs");
	static_assert(RightLimbs > 0, "invalid RightLimbs");

	DoubleLimb result[LeftLimbs + RightLimbs];
	std::memset(result, 0, sizeof(result));

	for (unsigned r = 0; r < RightLimbs; ++r)
	{
		Limb multiplier = right[r];

		for (unsigned l = 0; l < LeftLimbs; ++l)
		{
			result[r + l] += left[l] * multiplier;
		}

		for (unsigned l = 0; l < LeftLimbs; ++l)
		{
			result[r + l + 1] += MultiplyHighHelper(left[l], multiplier);
		}
	}

	DoubleLimb accumulator = 0;
	for (unsigned x = 0; x < LeftLimbs + RightLimbs; ++x)
	{
		accumulator += result[x];
		dest[x] = static_cast<Limb>(accumulator);
		accumulator >>= 32;
	}
}


template <unsigned Limbs>
__device__ void BarrettReduce(Limb remainder[Limbs * 3 + 1 + 1], const Limb dividend[Limbs * 2], const Limb modulus[Limbs], const Limb inverse[Limbs + 1])
{
	BigMultiplyMN<Limbs * 2, Limbs + 1>(&remainder[1], dividend, inverse);

	BigMultiplyMN<Limbs + 1, Limbs>(&remainder[0], &remainder[Limbs * 2 + 1], modulus);

	BigSubtractN<Limbs + 1>(&remainder[0], &dividend[0], &remainder[0], 0);

	if ((BigCompareN<Limbs>(&remainder[0], &modulus[0]) >= 0) || (remainder[Limbs] > 0))
	{
		BigSubtractN<Limbs>(&remainder[0], &remainder[0], &modulus[0], 0);
	}
}

/*template <unsigned Limbs>
__device__ void BarrettReduce(Limb remainder[Limbs], const Limb dividend[Limbs * 2], const Limb modulus[Limbs], const Limb inverse[Limbs + 1])
{
	Limb result[Limbs * 3 + 1 + 1];
	BigMultiplyMN<Limbs * 2, Limbs + 1>(&result[1], dividend, inverse);

	BigMultiplyMN<Limbs + 1, Limbs>(&result[0], &result[Limbs * 2 + 1], modulus);

	BigSubtractN<Limbs + 1>(&result[0], &dividend[0], &result[0], 0);

	if ((BigCompareN<Limbs>(&result[0], &modulus[0]) >= 0) || (result[Limbs] > 0))
	{
		BigSubtractN<Limbs>(&remainder[0], &result[0], &modulus[0], 0);
	}
	else
	{
		for (unsigned x = 0; x < Limbs; ++x)
		{
			remainder[x] = result[x];
		}
	}
}*/
//#ifdef NUM_REPEATS
//#error hiss
//#endif

// The root function of the GPU side of the code.
__global__ void MultiplyStuffKernel(TransferBlob *dest, const TransferBlob *src, const MathParameters *math)
{
	// Copy the global variables to kernel-local variables.
	__shared__ Limb sharedModulus[LIMB_COUNT];
	__shared__ Limb sharedInverse[LIMB_COUNT + 1];

	if (threadIdx.x == 0)
	{
		static_assert(sizeof(sharedModulus) == sizeof(math->m_modulus), "modulus size mismatch");
		std::memcpy(sharedModulus, math->m_modulus, sizeof(sharedModulus));
		static_assert(sizeof(sharedInverse) == sizeof(math->m_inverse), "modulus size mismatch");
		std::memcpy(sharedInverse, math->m_inverse, sizeof(sharedInverse));
	}
	__syncthreads();

	// Copy this thread's input to local variables.
	Limb localSrc[LIMB_COUNT * 3 + 1 + 1];
	Limb squared[LIMB_COUNT * 2];

	for (unsigned x = 0; x < LIMB_COUNT; ++x)
	{
		localSrc[x] = src->m_iterations[threadIdx.x].m_limbs[x];
	}

	// Do the math.
	for (unsigned x = 0; x < NUM_REPEATS; ++x)
	{
		BigMultiplyMN<LIMB_COUNT, LIMB_COUNT>(squared, localSrc, localSrc);
		BarrettReduce<LIMB_COUNT>(localSrc, squared, sharedModulus, sharedInverse);
		__syncthreads();
	}

	// Write this thread's results.
	for (unsigned x = 0; x < LIMB_COUNT; ++x)
	{
		dest->m_iterations[threadIdx.x].m_limbs[x] = localSrc[x];
	}
}


// Execute the math operation.
cudaError_t GPUExecuteOperation(TransferBlob &dest, const TransferBlob &src, const MathParameters &mathParams)
{
	cudaError_t cudaStatus;

	// Allocate memory for data transfers.
	void *d_src;
	cudaStatus = cudaMalloc(&d_src, sizeof(src));
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	void *d_dest;
	cudaStatus = cudaMalloc(&d_dest, sizeof(dest));
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_src);
		return cudaStatus;
	}

	void *d_mathParams;
	cudaStatus = cudaMalloc(&d_mathParams, sizeof(mathParams));
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_src);
		cudaFree(d_dest);
		return cudaStatus;
	}

	// Copy parameters over.
	cudaStatus = cudaMemcpy(d_src, &src, sizeof(src), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_src);
		cudaFree(d_dest);
		cudaFree(d_mathParams);
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_mathParams, &mathParams, sizeof(mathParams), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_src);
		cudaFree(d_dest);
		cudaFree(d_mathParams);
		return cudaStatus;
	}

	// Execute operation.
	MultiplyStuffKernel<<<1, NUM_THREADS>>>(static_cast<TransferBlob *>(d_dest), static_cast<TransferBlob *>(d_src), static_cast<MathParameters *>(d_mathParams));

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_src);
		cudaFree(d_dest);
		cudaFree(d_mathParams);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_src);
		cudaFree(d_dest);
		cudaFree(d_mathParams);
		return cudaStatus;
	}

	// Copy result to host.
	cudaStatus = cudaMemcpy(&dest, d_dest, sizeof(dest), cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_dest);
	cudaFree(d_mathParams);

	return cudaSuccess;
}
