#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "Shared.h"
#include "../Moduli/Pattern.h"


// The GPU code.
__global__ void MultiplyStuffKernel(
	Limb *dest,
	const Limb *__restrict__ src,
	unsigned char *__restrict__ resultFlags)
{
	// Start of code.

	// Offset the two pointers depending on which block this is.
	dest += blockIdx.x * BLOCK_LIMB_COUNT;
	src += blockIdx.x * BLOCK_LIMB_COUNT;

	#define DECLARE_OUTPUT_ROUND(num) Limb var_t##num = 0

	#define DECLARE_OUTPUT_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
		DECLARE_OUTPUT_ROUND(r1); DECLARE_OUTPUT_ROUND(r2); DECLARE_OUTPUT_ROUND(r3); \
		DECLARE_OUTPUT_ROUND(r4); DECLARE_OUTPUT_ROUND(r5); DECLARE_OUTPUT_ROUND(r6); \
		DECLARE_OUTPUT_ROUND(r7); DECLARE_OUTPUT_ROUND(r8); DECLARE_OUTPUT_ROUND(r9);

	DECLARE_OUTPUT_ROUND(00);
	DECLARE_OUTPUT_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
	DECLARE_OUTPUT_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
	DECLARE_OUTPUT_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
	DECLARE_OUTPUT_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
	DECLARE_OUTPUT_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
	DECLARE_OUTPUT_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
	DECLARE_OUTPUT_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);
	DECLARE_OUTPUT_ROUND(64);
	DECLARE_OUTPUT_ROUND(65);

	Limb var_multiplicand = 0;
	Limb var_m = 0;
	Limb var_carry = 0;
	Limb var_temp = 0;


	// Reduce subroutine.  Multiplies by R^-1 mod n.
	auto reduce = [&]()
	{
		// Reduce rounds.
		asm(
			// Get multiplicand used for reduction.
			"mul.lo.u32 %0, %1, " MODULUS_INVERSE_LOW ";\n\t"
			// The first round of reduction discards the result.
			"mad.lo.cc.u32 %2, %0, " MODULUS_WORD_00 ", %1;\n\t"
			"madc.hi.u32 %2, %0, " MODULUS_WORD_00 ", 0;\n\t" // note no % on 0
			:	"+r"(var_m), "+r"(var_t00), "+r"(var_carry));

		#define REDUCE_ROUND(prev, curr) \
			asm( \
				"mad.lo.cc.u32 %0, %2, " MODULUS_WORD_##curr ", %3;\n\t" \
				"madc.hi.u32 %3, %2, " MODULUS_WORD_##curr ", 0;\n\t" /* note no % on 0 */ \
				"add.cc.u32 %0, %0, %1;\n\t" \
				"addc.u32 %3, %3, 0;\n\t" /* note no % on 0*/ \
				:	"+r"(var_t##prev), "+r"(var_t##curr), "+r"(var_m), "+r"(var_carry))

		#define REDUCE_ROUND_9(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9) \
			REDUCE_ROUND(r0, r1); REDUCE_ROUND(r1, r2); REDUCE_ROUND(r2, r3); \
			REDUCE_ROUND(r3, r4); REDUCE_ROUND(r4, r5); REDUCE_ROUND(r5, r6); \
			REDUCE_ROUND(r6, r7); REDUCE_ROUND(r7, r8); REDUCE_ROUND(r8, r9)

		REDUCE_ROUND_9(00, 01, 02, 03, 04, 05, 06, 07, 08, 09);
		REDUCE_ROUND_9(09, 10, 11, 12, 13, 14, 15, 16, 17, 18);
		REDUCE_ROUND_9(18, 19, 20, 21, 22, 23, 24, 25, 26, 27);
		REDUCE_ROUND_9(27, 28, 29, 30, 31, 32, 33, 34, 35, 36);
		REDUCE_ROUND_9(36, 37, 38, 39, 40, 41, 42, 43, 44, 45);
		REDUCE_ROUND_9(45, 46, 47, 48, 49, 50, 51, 52, 53, 54);
		REDUCE_ROUND_9(54, 55, 56, 57, 58, 59, 60, 61, 62, 63);

		asm(
			"add.cc.u32 %0, %1, %3;\n\t"
			"addc.u32 %1, %2, 0;\n\t" // note no % on 0
			:	"+r"(var_t63), "+r"(var_t64), "+r"(var_t65), "+r"(var_carry));
	};


	// Subroutine to write results.
	auto writeResults = [&](Limb *d, unsigned span)
	{
		#define STORE_ROUND(num) d[(1##num - 100) * span] = var_t##num;

		#define STORE_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
			STORE_ROUND(r1); STORE_ROUND(r2); STORE_ROUND(r3); \
			STORE_ROUND(r4); STORE_ROUND(r5); STORE_ROUND(r6); \
			STORE_ROUND(r7); STORE_ROUND(r8); STORE_ROUND(r9);

		STORE_ROUND(00);
		STORE_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
		STORE_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
		STORE_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
		STORE_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
		STORE_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
		STORE_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
		STORE_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);
	};


	// First round moved out of main loop.
	// First round is special, because registers not set yet.
	// Multiplier for this round.
	var_multiplicand = src[(0 * NUM_THREADS) + threadIdx.x];

	asm(
		"mul.lo.u32 %0, %1, " MULTIPLIER_WORD_00 ";\n\t"
		"mul.hi.u32 %2, %1, " MULTIPLIER_WORD_00 ";\n\t"
		:	"+r"(var_t00), "+r"(var_multiplicand), "+r"(var_carry));

	#define MULTIPLY_FIRST_ROUND(num) \
		asm( \
			"mad.lo.cc.u32 %0, %1, " MULTIPLIER_WORD_##num ", %2;\n\t" \
			"madc.hi.u32 %2, %1, " MULTIPLIER_WORD_##num ", 0;\n\t" \
			:	"+r"(var_t##num), "+r"(var_multiplicand), "+r"(var_carry))

	#define MULTIPLY_FIRST_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
		MULTIPLY_FIRST_ROUND(r1); MULTIPLY_FIRST_ROUND(r2); MULTIPLY_FIRST_ROUND(r3); \
		MULTIPLY_FIRST_ROUND(r4); MULTIPLY_FIRST_ROUND(r5); MULTIPLY_FIRST_ROUND(r6); \
		MULTIPLY_FIRST_ROUND(r7); MULTIPLY_FIRST_ROUND(r8); MULTIPLY_FIRST_ROUND(r9);

	MULTIPLY_FIRST_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
	MULTIPLY_FIRST_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
	MULTIPLY_FIRST_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
	MULTIPLY_FIRST_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
	MULTIPLY_FIRST_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
	MULTIPLY_FIRST_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
	MULTIPLY_FIRST_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);

	var_t64 = var_carry;
	var_t65 = 0;

	// Reduce - multiply by R^-1 mod n.
	reduce();


	// Main multiplication-reduction loop.
	for (unsigned i = 1; i < LIMB_COUNT; ++i)
	{
		// Multiplier for this round.
		var_multiplicand = src[(i * NUM_THREADS) + threadIdx.x];

		// Every round after the first is this way.
		asm(
			"mad.lo.cc.u32 %0, %1, " MULTIPLIER_WORD_00 ", %0;\n\t"
			"madc.hi.u32 %2, %1, " MULTIPLIER_WORD_00 ", 0;\n\t"  // note no % on this 0
			:	"+r"(var_t00), "+r"(var_multiplicand), "+r"(var_carry));

		#define MULTIPLY_ROUND(num) \
			asm( \
				"mad.lo.cc.u32 %0, %1, " MULTIPLIER_WORD_##num ", %0;\n\t" \
				"madc.hi.u32 %2, %1, " MULTIPLIER_WORD_##num ", 0;\n\t" /* note no % on 0 */ \
				"add.cc.u32 %0, %0, %3;\n\t" \
				"addc.u32 %3, %2, 0;\n\t" /* note no % on 0 */ \
				:	"+r"(var_t##num), "+r"(var_multiplicand), "+r"(var_temp), "+r"(var_carry))

		#define MULTIPLY_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
			MULTIPLY_ROUND(r1); MULTIPLY_ROUND(r2); MULTIPLY_ROUND(r3); \
			MULTIPLY_ROUND(r4); MULTIPLY_ROUND(r5); MULTIPLY_ROUND(r6); \
			MULTIPLY_ROUND(r7); MULTIPLY_ROUND(r8); MULTIPLY_ROUND(r9)

		MULTIPLY_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
		MULTIPLY_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
		MULTIPLY_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
		MULTIPLY_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
		MULTIPLY_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
		MULTIPLY_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
		MULTIPLY_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);

		asm(
			"add.cc.u32 %0, %0, %1;\n\t"
			"addc.u32 %2, %2, 0;\n\t" // note no % on 0
			:	"+r"(var_t64), "+r"(var_carry), "+r"(var_t65));

		// Reduce - multiply by R^-1 mod n.
		reduce();
	}


	// Function to subtract the modulus.
	auto subtract = [&]()
	{
		asm(
			"sub.cc.u32 %1, %1, " MODULUS_WORD_00 ";\n\t"
			"addc.u32 %2, 0, 0;\n\t"  // extract carry flag to carry variable
			:	"+r"(var_temp), "+r"(var_t00), "+r"(var_carry));

		#define SUBTRACT_ROUND(regnum, wordnum) \
			"subc.cc.u32 %" #regnum ", %" #regnum ", " MODULUS_WORD_##wordnum ";\n\t"

		#define SUBTRACT_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
			asm( \
				"add.cc.u32 %0, %10, 0xFFFFFFFFU;\n\t" /* put carry variable into carry flag */ \
				SUBTRACT_ROUND(1, r1) SUBTRACT_ROUND(2, r2) SUBTRACT_ROUND(3, r3) \
				SUBTRACT_ROUND(4, r4) SUBTRACT_ROUND(5, r5) SUBTRACT_ROUND(6, r6) \
				SUBTRACT_ROUND(7, r7) SUBTRACT_ROUND(8, r8) SUBTRACT_ROUND(9, r9) \
				"addc.u32 %10, 0, 0;\n\t"  /* extract carry flag to carry variable */ \
				:	"+r"(var_temp), \
					"+r"(var_t##r1), "+r"(var_t##r2), "+r"(var_t##r3), \
					"+r"(var_t##r4), "+r"(var_t##r5), "+r"(var_t##r6), \
					"+r"(var_t##r7), "+r"(var_t##r8), "+r"(var_t##r9), \
					"+r"(var_carry))

		SUBTRACT_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
		SUBTRACT_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
		SUBTRACT_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
		SUBTRACT_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
		SUBTRACT_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
		SUBTRACT_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
		SUBTRACT_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);
	};


	// Function to subtract the modulus if the result is greater than or equal
	// to the modulus.
	auto compareReduce1 = [&]()
	{
		// Compare against the modulus.
		asm(
			"sub.cc.u32 %0, %1, " MODULUS_WORD_00 ";\n\t"
			"addc.u32 %2, 0, 0;\n\t"  // extract carry flag to carry variable
			:	"+r"(var_temp), "+r"(var_t00), "+r"(var_carry));

		#define COMPARE_ROUND(regnum, wordnum) \
			"subc.cc.u32 %0, %" #regnum ", " MODULUS_WORD_##wordnum ";\n\t"

		#define COMPARE_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
			asm( \
				"add.cc.u32 %0, %10, 0xFFFFFFFFU;\n\t" /* put carry variable into carry flag */ \
				COMPARE_ROUND(1, r1) COMPARE_ROUND(2, r2) COMPARE_ROUND(3, r3) \
				COMPARE_ROUND(4, r4) COMPARE_ROUND(5, r5) COMPARE_ROUND(6, r6) \
				COMPARE_ROUND(7, r7) COMPARE_ROUND(8, r8) COMPARE_ROUND(9, r9) \
				"addc.u32 %10, 0, 0;\n\t"  /* extract carry flag to carry variable */ \
				:	"+r"(var_temp), \
					"+r"(var_t##r1), "+r"(var_t##r2), "+r"(var_t##r3), \
					"+r"(var_t##r4), "+r"(var_t##r5), "+r"(var_t##r6), \
					"+r"(var_t##r7), "+r"(var_t##r8), "+r"(var_t##r9), \
					"+r"(var_carry))

		COMPARE_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
		COMPARE_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
		COMPARE_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
		COMPARE_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
		COMPARE_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
		COMPARE_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
		COMPARE_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);

		// If necessary, subtract.
		// carry is 1 if above or equal, 0 if not (6502 semantics, not x86).
		if ((var_carry != 0) || (var_t64 != 0))
		{
			subtract();
		}
	};


	// Function to negate modulo the modulus.
	auto negate = [&]()
	{
		asm(
			"sub.cc.u32 %1, " MODULUS_WORD_00 ", %1;\n\t"
			"addc.u32 %2, 0, 0;\n\t"  // extract carry flag to carry variable
			:	"+r"(var_temp), "+r"(var_t00), "+r"(var_carry));

		#define NEGATE_ROUND(regnum, wordnum) \
			"subc.cc.u32 %" #regnum ", " MODULUS_WORD_##wordnum ", %" #regnum ";\n\t"

		#define NEGATE_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
			asm( \
				"add.cc.u32 %0, %10, 0xFFFFFFFFU;\n\t" /* put carry variable into carry flag */ \
				NEGATE_ROUND(1, r1) NEGATE_ROUND(2, r2) NEGATE_ROUND(3, r3) \
				NEGATE_ROUND(4, r4) NEGATE_ROUND(5, r5) NEGATE_ROUND(6, r6) \
				NEGATE_ROUND(7, r7) NEGATE_ROUND(8, r8) NEGATE_ROUND(9, r9) \
				"addc.u32 %10, 0, 0;\n\t"  /* extract carry flag to carry variable */ \
				:	"+r"(var_temp), \
					"+r"(var_t##r1), "+r"(var_t##r2), "+r"(var_t##r3), \
					"+r"(var_t##r4), "+r"(var_t##r5), "+r"(var_t##r6), \
					"+r"(var_t##r7), "+r"(var_t##r8), "+r"(var_t##r9), \
					"+r"(var_carry))

		NEGATE_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
		NEGATE_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
		NEGATE_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
		NEGATE_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
		NEGATE_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
		NEGATE_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
		NEGATE_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);
	};


	// Do one iteration of the subtraction.
	compareReduce1();

	// Write results, which is a*b*R mod n.
	writeResults(&dest[(0 * TOTAL_LIMB_COUNT) + threadIdx.x], NUM_THREADS);

	// Multiply by R^-1 again so that the actual answer is visible.
	var_t64 = 0;
	var_t65 = 0;
	for (unsigned i = 0; i < LIMB_COUNT; ++i)
	{
		reduce();
	}
	compareReduce1();

	// Write the non-Montgomery results.
	writeResults(&dest[(1 * TOTAL_LIMB_COUNT) + threadIdx.x], NUM_THREADS);


	// Check whether the positive version is a match.
	Limb checkBuffer[LIMB_COUNT];
	writeResults(checkBuffer, 1);

	auto getByteFromCheckBuffer = [&checkBuffer](unsigned index) -> unsigned char
	{
		index = ((LIMB_COUNT * sizeof(Limb)) - 1) - index;
		return static_cast<unsigned char>(checkBuffer[index / sizeof(Limb)] >> ((index % sizeof(Limb)) * 8));
	};

	bool matchPositive = IsWhatWeWant(getByteFromCheckBuffer);

	// Check whether the negative version is a match.
	negate();
	writeResults(checkBuffer, 1);

	bool matchNegative = IsWhatWeWant(getByteFromCheckBuffer);


	// Flag whether we found a result.
	int combinedResult = __syncthreads_or(matchPositive || matchNegative);

	if (threadIdx.x == 0)
	{
		resultFlags[blockIdx.x] = combinedResult ? 1 : 0;
	}

#undef DECLARE_OUTPUT_ROUND
#undef DECLARE_OUTPUT_ROUND_9
#undef MULTIPLY_FIRST_ROUND
#undef MULTIPLY_FIRST_ROUND_9
#undef MULTIPLY_ROUND
#undef MULTIPLY_ROUND_9
#undef REDUCE_ROUND
#undef REDUCE_ROUND_9
#undef COMPARE_ROUND
#undef COMPARE_ROUND_9
#undef SUBTRACT_ROUND
#undef SUBTRACT_ROUND_9
#undef NEGATE_ROUND
#undef NEGATE_ROUND_9
#undef STORE_ROUND
#undef STORE_ROUND_9
}


// Constructor to just null out the pointers.
GPUState::GPUState()
{
	d_buffers[0] = nullptr;
	d_buffers[1] = nullptr;
	d_resultFlags = nullptr;
	h_resultFlags = nullptr;
}


// Destructor to free resources.
GPUState::~GPUState()
{
	if (d_buffers[0])
	{
		cudaFree(d_buffers[0]);
		d_buffers[0] = nullptr;
	}

	if (d_buffers[1])
	{
		cudaFree(d_buffers[1]);
		d_buffers[1] = nullptr;
	}

	if (d_resultFlags)
	{
		cudaFree(d_resultFlags);
		d_resultFlags = nullptr;
	}

	delete[] h_resultFlags;
	h_resultFlags = nullptr;

	cudaDeviceReset();
}


// Main initialization function.
cudaError_t GPUState::Initialize(int device)
{
	cudaError_t status;

	// Choose which GPU to run on, change this on a multi-GPU system.
	status = cudaSetDevice(device);
	if (status != cudaSuccess)
	{
		return status;
	}

	// Allocate the two buffers.
	status = cudaMalloc(&d_buffers[0], sizeof(Limb) * (TOTAL_LIMB_COUNT * 2));
	if (status != cudaSuccess)
	{
		d_buffers[0] = nullptr;
		return status;
	}

	status = cudaMalloc(&d_buffers[1], sizeof(Limb) * (TOTAL_LIMB_COUNT * 2));
	if (status != cudaSuccess)
	{
		d_buffers[1] = nullptr;
		return status;
	}

	// Allocate the result flags.
	status = cudaMalloc(&d_resultFlags, sizeof(unsigned char) * NUM_BLOCKS);
	if (status != cudaSuccess)
	{
		d_resultFlags = nullptr;
		return status;
	}

	// Allocate the result flags on the host side.
	h_resultFlags = new unsigned char[NUM_BLOCKS];

	return cudaSuccess;
}


// Reseeds the state for a new round.
cudaError_t GPUState::Reseed(unsigned currentSrc, const Limb seed[TOTAL_LIMB_COUNT])
{
	// Copy to the specified buffer.
	return cudaMemcpy(d_buffers[currentSrc], seed, sizeof(Limb[TOTAL_LIMB_COUNT]), cudaMemcpyHostToDevice);
}


// Execute the math operation.
cudaError_t GPUState::Execute(unsigned currentSrc, Limb output[TOTAL_LIMB_COUNT], bool &matchFound)
{
	cudaError_t status;

	// Debug information.
	cudaFuncAttributes attributes;
	status = cudaFuncGetAttributes(&attributes, reinterpret_cast<void *>(MultiplyStuffKernel));
	if (status != cudaSuccess)
	{
		return status;
	}

	static bool s_printed = true; //false;
	if (!s_printed)
	{
	#define OUTPUT_DEBUG(field, formatter) std::printf(#field " = " formatter "\n", field)
		OUTPUT_DEBUG(attributes.sharedSizeBytes, "%zu");
		OUTPUT_DEBUG(attributes.constSizeBytes, "%zu");
		OUTPUT_DEBUG(attributes.localSizeBytes, "%zu");
		OUTPUT_DEBUG(attributes.maxThreadsPerBlock, "%d");
		OUTPUT_DEBUG(attributes.numRegs, "%d");
		OUTPUT_DEBUG(attributes.ptxVersion, "%d");
		OUTPUT_DEBUG(attributes.binaryVersion, "%d");
		OUTPUT_DEBUG(attributes.cacheModeCA, "%d");
		s_printed = true;
	}

	// Execute operation.
	MultiplyStuffKernel<<<NUM_BLOCKS, NUM_THREADS>>>(
		static_cast<Limb *>(d_buffers[currentSrc ^ 1]),
		static_cast<Limb *>(d_buffers[currentSrc]),
		static_cast<unsigned char *>(d_resultFlags));

	// Check for any errors launching the kernel
	status = cudaGetLastError();
	if (status != cudaSuccess)
	{
		return status;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	status = cudaDeviceSynchronize();
	if (status != cudaSuccess)
	{
		return status;
	}

	// Copy result flags to host.
	status = cudaMemcpy(h_resultFlags, d_resultFlags, sizeof(unsigned char) * NUM_BLOCKS, cudaMemcpyDeviceToHost);
	if (status != cudaSuccess)
	{
		return status;
	}

	// Collapse the result flags down to a single value.
	DoubleLimb combinedResult = 0;

	static_assert((sizeof(unsigned char) * NUM_BLOCKS) % sizeof(DoubleLimb) == 0,
		"NUM_BLOCKS not a multiple of DoubleLimb's size");

	for (size_t x = 0; x < (sizeof(unsigned char) * NUM_BLOCKS) / sizeof(DoubleLimb); ++x)
	{
		DoubleLimb temp;
		std::memcpy(&temp, &h_resultFlags[x * sizeof(DoubleLimb)], sizeof(DoubleLimb));
		combinedResult |= temp;
	}

	matchFound = (combinedResult != 0);

	// If a match was found, we want the full data.
	if (matchFound)
	{
		// Copy result to host.
		status = cudaMemcpy(output, static_cast<char *>(d_buffers[currentSrc ^ 1]) + sizeof(Limb[TOTAL_LIMB_COUNT]),
			sizeof(Limb[TOTAL_LIMB_COUNT]), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess)
		{
			return status;
		}
	}

	return cudaSuccess;
}
