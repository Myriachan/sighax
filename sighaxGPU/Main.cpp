#include "cuda_runtime.h"

#include <cstdio>
#include <cinttypes>
#include <cstring>
#include <ctime>
#include <type_traits>

#include "Shared.h"
#include "Pattern.h"

#ifdef _WIN32
	#define _WIN32_WINNT 0x0601
	#define NOGDI
	#define NOUSER
	#include <Windows.h>
	// Use MPIR in Windows.
	#include <mpir.h>
#else
	// For priority functions.
	#include <sys/resource.h>
	#include <sys/time.h>
	#include <unistd.h>
	// Use GMP elsewhere.
	#include <gmp.h>
#endif


#ifdef _WIN32
	// Because NTSecAPI.h has an incorrect prototype (missing __stdcall).
	extern "C" __declspec(dllimport) BOOLEAN NTAPI SystemFunction036(PVOID, ULONG);
#endif


// Number of rounds.
unsigned optNumRounds = DEFAULT_NUM_ROUNDS;


void ReadRandom(void *data, size_t size)
{
#ifdef _WIN32
	// You might need to link advapi32.lib (-ladvapi32 in GCC).
	if (size > (std::numeric_limits<unsigned long>::max)())
	{
		std::printf("ReadRandom size too large: %llu\n", static_cast<unsigned long long>(size));
		std::exit(1);
	}

	if (!SystemFunction036(data, static_cast<unsigned long>(size)))
	{
		std::printf("RtlGenRandom failed\n");
		std::exit(1);
	}
#else
	FILE *file = std::fopen("/dev/urandom", "rb");
	if (!file)
	{
		std::printf("Could not open /dev/urandom\n");
		std::exit(1);
	}

	if (std::fread(data, 1, size, file) != size)
	{
		std::printf("Reading from /dev/urandom failed\n");
		std::exit(1);
	}

	std::fclose(file);
#endif
}


// Exports an mpz_t as an array of mp_limb_t for mpn_* use.
template <size_t S>
void ToLimbArray(mp_limb_t(&limbs)[S], mpz_t number)
{
	if (mpz_sizeinbase(number, 2) > S * std::numeric_limits<mp_limb_t>::digits)
	{
		std::abort();
	}

	size_t count;
	mpz_export(limbs, &count, -1, sizeof(mp_limb_t), 0, 0, number);
	std::memset(&limbs[count], 0, (S - count) * sizeof(mp_limb_t));
}

// Imports a buffer from a limb array to an mpz_t.
template <size_t S>
void FromLimbArray(mpz_t number, const mp_limb_t(&limbs)[S])
{
	mpz_import(number, S, -1, sizeof(limbs[0]), 0, 0, limbs);
}


template <unsigned S>
union Number
{
	LimbArray<S> m_limbs;
	mp_limb_t m_gmp[(S + (GMP_LIMB_BITS / LIMB_BITS) - 1) / (GMP_LIMB_BITS / LIMB_BITS)];
	unsigned char m_bytes[LIMB_COUNT * sizeof(Limb)];

	static void Dummy()
	{
		static_assert(sizeof(m_limbs) == sizeof(m_gmp), "Unsupported limb size");
	}

	void Print(std::FILE *output = stdout)
	{
		for (unsigned x = 0; x < countof(m_limbs); ++x)
		{
			std::fprintf(output, LIMB_PRINTF_FORMAT, m_limbs[countof(m_limbs) - 1 - x]);
		}
	}
};


class MPZNumber
{
public:
	MPZNumber()
	{
		mpz_init2(m_gmp, MODULUS_BITS * 2);
	}

	~MPZNumber()
	{
		mpz_clear(m_gmp);
	}

	operator std::remove_extent_t<mpz_t> *() { return m_gmp; }

private:
	mpz_t m_gmp;
};


// Byte swap helper.
Limb ByteSwapLimb(Limb limb)
{
#ifdef _MSC_VER
	static_assert((std::numeric_limits<Limb>::max)() == (std::numeric_limits<unsigned long>::max)(),
		"Unexpected limb size");
	return static_cast<Limb>(_byteswap_ulong(static_cast<unsigned long>(limb)));
#elif defined(__clang__) || defined(__GNUC__)
	static_assert((std::numeric_limits<Limb>::max)() == (std::numeric_limits<std::uint32_t>::max)(),
		"Unexpected limb size");
	return static_cast<Limb>(__builtin_bswap32(static_cast<std::uint32_t>(limb)));
#else
	#error Need implementation
#endif
}


// Quick and dirty random number generator from Wikipedia.
// The state must be seeded so that it is not everywhere zero.
uint64_t xorshift_s[16];
int xorshift_p = 0;

uint64_t xorshift1024star(void)
{
	const uint64_t s0 = xorshift_s[xorshift_p];
	uint64_t s1 = xorshift_s[xorshift_p = (xorshift_p + 1) & 15];
	s1 ^= s1 << 31; // a
	xorshift_s[xorshift_p] = s1 ^ s0 ^ (s1 >> 11) ^ (s0 >> 30); // b, c
	return xorshift_s[xorshift_p] * UINT64_C(1181783497276652981);
}


// Generates a random base, then calculates it to the 65537th power and multiplies
// by R to put it in Montgomery form.
void MakeRandomRequest(Number<LIMB_COUNT> &base, Number<LIMB_COUNT> &initialValue, mpz_t gmpModulus, mpz_t gmpR)
{
	// Generate random crap.
	static_assert(std::numeric_limits<mp_limb_t>::digits == std::numeric_limits<std::uint64_t>::digits,
		"unexpected type of mp_limb_t");

	for (mp_limb_t &limb : base.m_gmp)
	{
		limb = xorshift1024star();
	}

	// Convert to GMP form.
	MPZNumber gmp;
	FromLimbArray(gmp, base.m_gmp);

	// Mod it by the modulus to ensure that it is in range.
	mpz_mod(gmp, gmp, gmpModulus);

	// Re-export it to "base".
	ToLimbArray(base.m_gmp, gmp);

	// Take it to the 65537th power.
	mpz_powm_ui(gmp, gmp, s_publicExponent, gmpModulus);

	// Convert to Montgomery form by multiplying by R.
	mpz_mul(gmp, gmp, gmpR);
	mpz_mod(gmp, gmp, gmpModulus);

	// Export as the data passed to the GPU.
	ToLimbArray(initialValue.m_gmp, gmp);
}


// Takes an interleaved number and reverts it to interleaved format.
void DeinterleaveNumber(Limb *dest, const Limb *src, unsigned block, unsigned thread)
{
	// The interleaved format has each block in order, then within a block,
	// the first limbs for each thread, then the second limbs for each thread...
	// This is the interleaved form.
	unsigned base = (block * BLOCK_LIMB_COUNT) + thread;

	for (unsigned i = 0; i < LIMB_COUNT; ++i)
	{
		dest[i] = src[base + (i * NUM_THREADS)];
	}
}


// Wrapper class for reading bytes from a limb array.
struct GetByteFromArrayWrapper
{
	GetByteFromArrayWrapper(const mp_limb_t *limbs)
	:	m_limbs(limbs)
	{
	}

	unsigned char operator()(unsigned index)
	{
		// unsigned char is allowed to alias in C/C++ rules.
		static_assert((KEY_SIZE & (KEY_SIZE - 1)) == 0, "KEY_SIZE must be a power of 2");
		return (reinterpret_cast<const unsigned char *>(m_limbs))[index ^ (KEY_SIZE - 1)];
	}

	const mp_limb_t *m_limbs;
};


// Checks a potential match.
bool CheckForMatch(const Limb *buffer, const Number<LIMB_COUNT> &modulus, unsigned block, unsigned thread, bool negative)
{
	// Deinterleave the input.
	Number<LIMB_COUNT> deinterleaved;
	DeinterleaveNumber(deinterleaved.m_limbs, buffer, block, thread);

	// Subtract from the modulus if negative is true.
	if (negative)
	{
		Limb borrow = 0;
		for (unsigned i = 0; i < LIMB_COUNT; ++i)
		{
			DoubleLimb difference = static_cast<DoubleLimb>(modulus.m_limbs[i]) - deinterleaved.m_limbs[i] - borrow;
			deinterleaved.m_limbs[i] = static_cast<Limb>(difference);
			borrow = static_cast<Limb>(difference >> ((LIMB_BITS * 2) - 1));
		}
	}

	// Return whether the pattern matches.
	return IsWhatWeWant(GetByteFromArrayWrapper(deinterleaved.m_gmp));
}


// Search for matches.
bool SearchForMatches(unsigned &matchedBlock, unsigned &matchedThread, bool &matchedNegative,
	const Limb *buffer, const Number<LIMB_COUNT> &modulus)
{
	// The three constants to check for.
	static const Limb check1 = 0x00020000;
	Limb check2 = modulus.m_limbs[LIMB_COUNT - 1] - check1;
	Limb check3 = check2 - 0x10000;
	check2 &= 0xFFFF0000;
	check3 &= 0xFFFF0000;

	// Search each block.
	for (unsigned block = 0; block < NUM_BLOCKS; ++block)
	{
		// Get pointer to plaintext for this block.
		const Limb *blockPlaintext = buffer + (block * BLOCK_LIMB_COUNT);

		// Get pointer to the last limb of the plaintext for each thread.
		const Limb *blockLastLimb = blockPlaintext + ((LIMB_COUNT - 1) * NUM_THREADS);

		// Search this block.
		for (unsigned thread = 0; thread < NUM_THREADS; ++thread)
		{
			Limb high = blockLastLimb[thread] & 0xFFFF0000;

			if ((high == check1) || (high == check2) || (high == check3))
			{
				bool negative = high != check1;
				if (CheckForMatch(buffer, modulus, block, thread, negative))
				{
					matchedBlock = block;
					matchedThread = thread;
					matchedNegative = negative;
					return true;
				}
			}
		}
	}

	return false;
}


// Verify and report match.  Shows errors if no match.
bool VerifyAndReportMatch(const Number<LIMB_COUNT> &base, mpz_t gmpModulus, mpz_t gmpRoot,
	unsigned seed, unsigned round, unsigned block, unsigned thread, bool negative)
{
	// Load base as a GMP number.
	MPZNumber gmpBase;
	FromLimbArray(gmpBase, base.m_gmp);

	// Take the base to the power of the round number plus 1.
	// This is the number of times we multiplied by the multiplier.
	MPZNumber gmpPower;
	mpz_powm_ui(gmpPower, gmpRoot, round + 1, gmpModulus);

	// Multiply by the original base.
	MPZNumber gmpSignature;
	mpz_mul(gmpSignature, gmpBase, gmpPower);
	mpz_mod(gmpSignature, gmpSignature, gmpModulus);

	// Negate if the negative is what we found.
	if (negative)
	{
		mpz_sub(gmpSignature, gmpModulus, gmpSignature);
	}

	// Take it to the 65537th power to simulate a signature check.
	MPZNumber gmpResult;
	mpz_powm_ui(gmpResult, gmpSignature, s_publicExponent, gmpModulus);

	// Convert the result and signature to arrays.
	Number<LIMB_COUNT> signature;
	ToLimbArray(signature.m_gmp, gmpSignature);
	Number<LIMB_COUNT> result;
	ToLimbArray(result.m_gmp, gmpResult);

	if (!IsWhatWeWant(GetByteFromArrayWrapper(result.m_gmp)))
	{
		std::printf("Reconstruction error on seed %u round %u block %u thread %u mode %s!\n",
			seed, round, block, thread, negative ? "negative" : "positive");
		std::fflush(stdout);
		return false;
	}
	else
	{
		std::printf("Match found at seed %u round %u block %u thread %u mode %s!\n",
			seed, round, block, thread, negative ? "negative" : "positive");

		// The multiply by 2 is because we check both positive and negative.
		unsigned long long checked = 0;
		checked += seed;
		checked *= optNumRounds;
		checked += round;
		checked *= NUM_BLOCKS;
		checked += block;
		checked *= NUM_THREADS;
		checked += thread;
		checked *= 2;
		checked += negative ? 1 : 0;
		std::printf("%llu multiplies checked before match found.\n", checked);

		std::printf("Result:\n");
		result.Print();
		std::puts("");

		std::printf("Signature:\n");
		signature.Print();
		std::puts("");

		std::fflush(stdout);
		return true;
	}
}


// Main program.
int main(int, char **argv)
{
	// Command-line parameters.
	int optDevice = 0;
	for (++argv; *argv; ++argv)
	{
		if (std::strncmp(*argv, "--gpu=", 6) == 0)
		{
			optDevice = static_cast<int>(std::strtol(*argv + 6, nullptr, 0));
		}
		else if (std::strncmp(*argv, "--rounds=", 9) == 0)
		{
			optNumRounds = static_cast<int>(std::strtoul(*argv + 9, nullptr, 0));
		}
	}

	// Initialize random generator.
	ReadRandom(xorshift_s, sizeof(xorshift_s));

	// Initialize constants.
	MPZNumber gmpModulus;
	mpz_set_str(gmpModulus, s_modulus, 16);

	MPZNumber gmpR;
	mpz_setbit(gmpR, MODULUS_BITS);

	MPZNumber gmpInverse;
	mpz_invert(gmpInverse, gmpModulus, gmpR);
	mpz_sub(gmpInverse, gmpR, gmpInverse);

	MPZNumber gmpRModModulus;
	mpz_mod(gmpRModModulus, gmpR, gmpModulus);

	MPZNumber gmpRInverse;
	mpz_invert(gmpRInverse, gmpRModModulus, gmpModulus);

	MPZNumber gmpMultiplier;
	mpz_set_str(gmpMultiplier, s_multiplier, 16);

	MPZNumber gmpMultiplierMontgomery;
	mpz_set_str(gmpMultiplierMontgomery, s_multiplierMontgomery, 16);

	MPZNumber gmpMultiplierRoot;
	mpz_set_str(gmpMultiplierRoot, s_root, 16);

	// Get modulus as a raw number.
	Number<LIMB_COUNT> modulus;
	ToLimbArray(modulus.m_gmp, gmpModulus);

	// Initialize the GPU and allocate GPU memory.
	GPUState gpu;
	cudaError_t cudaStatus = gpu.Initialize(optDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::fprintf(stderr, "GPUState::Initialize failed (%d)!\n", static_cast<int>(cudaStatus));
		return 1;
	}

	// Allocate our list of bases.
	Number<LIMB_COUNT> *bases = new Number<LIMB_COUNT>[NUM_BLOCKS * NUM_THREADS];

	// Allocate our communication buffer.
	Limb *buffer = new Limb[TOTAL_LIMB_COUNT];

	// Total operations completed overall.
	unsigned long long programTotal = 0;
	
	std::clock_t programStart = std::clock();

	// Main search loop.  Seed won't overflow in our timeframe.
	for (unsigned seed = 0; ; ++seed)
	{
		std::printf("Seeding seed %u...\n", seed);
		std::fflush(stdout);

		// Generate new random numbers to use.
		for (unsigned block = 0; block < NUM_BLOCKS; ++block)
		{
			Limb *blockBase = buffer + (block * BLOCK_LIMB_COUNT);
			for (unsigned thread = 0; thread < NUM_THREADS; ++thread)
			{
				Number<LIMB_COUNT> initialValue;
				MakeRandomRequest(bases[(block * NUM_THREADS) + thread], initialValue, gmpModulus, gmpR);

				// Interleave initial value into buffer.
				Limb *threadBase = blockBase + thread;
				for (unsigned limb = 0; limb < LIMB_COUNT; ++limb)
				{
					threadBase[limb * NUM_THREADS] = initialValue.m_limbs[limb];
				}
			}
		}

		// Re-seed the round.
		gpu.Reseed(0, buffer);

		std::printf("Searching seed %u for %u rounds...\n", seed, optNumRounds);
		std::fflush(stdout);

		std::clock_t roundStart = std::clock();
		unsigned long long roundTotal = 0;

		for (unsigned round = 0; round < optNumRounds; ++round)
		{
			if ((round % 100) == 0)
			{
				std::printf("Executing seed %u round %u/%u...\n", seed, round, optNumRounds);
				std::fflush(stdout);
			}

			bool matchFound;

			cudaStatus = gpu.Execute(round & 1, buffer, matchFound);
			if (cudaStatus != cudaSuccess)
			{
				std::printf("GPUExecuteOperation failed: %d\n", static_cast<int>(cudaStatus));
				goto done;
			}

			// Multiplied by two because we count both positive and negative.
			roundTotal += NUM_BLOCKS * NUM_THREADS * 2;
			programTotal += NUM_BLOCKS * NUM_THREADS * 2;

			// Check for matches if gpu.Execute says that there are any.
			if (matchFound)
			{
				unsigned block;
				unsigned thread;
				bool negative;
				if (SearchForMatches(block, thread, negative, buffer, modulus))
				{
					VerifyAndReportMatch(bases[(block * NUM_THREADS) + thread], gmpModulus,
						gmpMultiplierRoot, seed, round, block, thread, negative);
				}
			}
		}

		std::clock_t roundEnd = std::clock();

		std::printf("Finished seed %u: %llu operations in %g seconds (%llu/second)\n",
			seed,
			roundTotal,
			static_cast<double>(roundEnd - roundStart) / CLOCKS_PER_SEC,
			static_cast<unsigned long long>(roundTotal / (static_cast<double>(roundEnd - roundStart) / CLOCKS_PER_SEC)));
		std::fflush(stdout);
	}

done:
	std::clock_t programEnd = std::clock();

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "cudaDeviceReset failed (%d)!\n", static_cast<int>(cudaStatus));
		return 1;
	}

	std::printf("Finished program: %llu operations in %g seconds (%llu/second)\n",
		programTotal,
		static_cast<double>(programEnd - programStart) / CLOCKS_PER_SEC,
		static_cast<unsigned long long>(programTotal / (static_cast<double>(programEnd - programStart) / CLOCKS_PER_SEC)));
	std::fflush(stdout);

	delete[] buffer;
	delete[] bases;
	return 0;
}
