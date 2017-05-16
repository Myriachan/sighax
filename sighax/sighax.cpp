// SigHax brute-forcer by Myria.
// BSD-licensed.
//
// Credits:
// * Myria: Main author.
// * SciresM: Insight into possible valid signatures (IsWhatWeWant).
// * plutoo: Hint on doing negation to get more bang for the mpz_mul buck.
// * Normmatt: Ported to Windows.
// * derrek: For finding the SigHax flaw in boot9 in the first place. <3

// PROFILE_MODE makes this program run for timing purposes, rather than
// actually searching.
//#define PROFILE_MODE

// The modulus to use.
#include "../Moduli/FIRM-NAND-retail.h"

#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>

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

using std::size_t;
using std::uint8_t;
using std::uint32_t;
using std::uint64_t;

enum : unsigned
{
	KEY_BITS = 2048,
	KEY_SIZE = KEY_BITS / CHAR_BIT,
	KEY_LIMB_SIZE = KEY_BITS / std::numeric_limits<mp_limb_t>::digits,
};

static_assert(KEY_BITS % std::numeric_limits<mp_limb_t>::digits == 0, "bad mp_limb_t size");

#define __device__
#define __host__
#include "../Moduli/Pattern.h"

// Useful types.
typedef mp_limb_t Single[KEY_LIMB_SIZE];
typedef mp_limb_t Double[2 * KEY_LIMB_SIZE];
typedef mp_limb_t Quad[4 * KEY_LIMB_SIZE];


template <typename T, size_t S>
constexpr size_t countof(const T (&)[S])
{
	return S;
}


#ifdef _WIN32
	// Because NTSecAPI.h has an incorrect prototype (missing __stdcall).
	extern "C" __declspec(dllimport) BOOLEAN NTAPI SystemFunction036(PVOID, ULONG);
#endif


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


void ToArray(uint8_t (&buffer)[KEY_SIZE], mpz_t number)
{
	size_t bits = mpz_sizeinbase(number, 2);
	if (bits > KEY_SIZE * CHAR_BIT)
	{
		std::abort();
	}

	size_t count;
	mpz_export(buffer, &count, -1, sizeof(uint64_t), -1, 0, number);
	std::memset(&buffer[count * sizeof(uint64_t)], 0, KEY_SIZE - (count * sizeof(uint64_t)));
}


void DumpNumber(mpz_t number)
{
	uint8_t buffer[KEY_SIZE];
	ToArray(buffer, number);

	size_t x = KEY_SIZE;
	do
	{
		std::printf("%02X", static_cast<unsigned>(buffer[x - 1]));
	}
	while ((--x) > 0);
	std::puts("");
}


// Exports a mpz_t as an array of mp_limb_t for mpn_* use.
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


// Wrapper class for reading bytes from a limb array.
struct GetByteFromArrayWrapper
{
	GetByteFromArrayWrapper(const mp_limb_t *limbs)
	:	m_limbs(limbs)
	{
	}

	unsigned char operator()(unsigned index) const
	{
		// unsigned char is allowed to alias in C/C++ rules.
		static_assert((KEY_SIZE & (KEY_SIZE - 1)) == 0, "KEY_SIZE must be a power of 2");
		return (reinterpret_cast<const unsigned char *>(m_limbs))[index ^ (KEY_SIZE - 1)];
	}

	const mp_limb_t *m_limbs;
};


void BruteForce(mpz_t mpzModulus, mpz_t mpzRoot, mpz_t mpzBlock, unsigned long long numIterations)
{
	enum
	{
		ATTEMPTS_PER_RANDOM = 1048576,
		PADDING_SHIFT_BITS = KEY_BITS - 15,
		LIMB_BITS = std::numeric_limits<mp_limb_t>::digits,
	};

	mpz_t mpzRandomBase;
	mpz_init(mpzRandomBase);
	mpz_t mpzRandomPowered;
	mpz_init(mpzRandomPowered);

	mpz_t mpzFoundBase;
	mpz_init(mpzFoundBase);
	mpz_t mpzFoundPowered;
	mpz_init(mpzFoundPowered);

	// Convert the modulus and root to mpn form.
	Single modulus;
	ToLimbArray(modulus, mpzModulus);

	Single root;
	ToLimbArray(root, mpzRoot);

	Single block;
	ToLimbArray(block, mpzBlock);

#ifdef SIGNATURE_IS_PKCS1
	// This is the PKCS #1 "01 FF FF FF ..." padding plus one.
	Single paddingBase;
	std::memset(paddingBase, 0, sizeof(paddingBase));
	paddingBase[PADDING_SHIFT_BITS / LIMB_BITS] = (0u + mp_limb_t(1)) << (PADDING_SHIFT_BITS % LIMB_BITS);

	// paddingBase minus the block, giving the multiplier to subtract.
	Single paddingNegLong;
	mpn_sub_n(paddingNegLong, paddingBase, block, countof(block));

	mp_limb_t paddingNegShort[7];
	std::memcpy(paddingNegShort, paddingNegLong, sizeof(paddingNegShort));
#endif // SIGNATURE_IS_PKCS1

	// Main loop.
	Single randomBase;
	Double current[2];

	for (unsigned long long iteration = 0; iteration < numIterations; ++iteration)
	{
		unsigned index = static_cast<unsigned>(iteration % 2);

		Double &prev = current[index ^ 1];
		Double &next = current[index];

		// Every so often, we need to reset the random number, otherwise it'll be hard to calculate
		// the signature later.  By resetting every ATTEMPTS_PER_RANDOM times, we limit the time
		// spent calculating the signature after finding an answer to ATTEMPTS_PER_RANDOM-1 multiplies.
		if (iteration % ATTEMPTS_PER_RANDOM == 0)
		{
			// Choose a random number.  Technically, we want 1 < r < n-1, but the three
			// degenerate numbers are so incredibly unlikely that we don't need to check for them.
			// (If they do hit, all that'll happen is that we'll never find a match this round.)
			ReadRandom(randomBase, sizeof(randomBase));

			mpz_import(mpzRandomBase, countof(randomBase), -1, sizeof(*randomBase), 0, 0, randomBase);
			mpz_mod(mpzRandomBase, mpzRandomBase, mpzModulus);
			ToLimbArray(randomBase, mpzRandomBase);

			// Calculate the 65537th power of this number.
			mpz_powm_ui(mpzRandomPowered, mpzRandomBase, s_publicExponent, mpzModulus);
			ToLimbArray(current[index], mpzRandomPowered);
		}
		else
		{
			// Multiply by the decrypted signature block.
		#ifdef SIGNATURE_IS_PKCS1

			// When s_signature is a valid PKCS #1 signature modulo s_modulus, we take advantage
			// of a simple math fact combined with the PKCS #1 signature format:
			//
			// 0x1FFFFFFFFFFF..00<408 other bits> * x
			//    =
			// (0x200000000000..00 * x) - ((0x200000000000..00 - 0x1FFFFFFFFFFF..00<408 other bits>) * x)
			//
			// This makes the multiplication a 2048x408 multiplication, a 2033-bit left shift,
			// and a 4096-bit subtract.  That is faster than either a 2048x2048 multiply or a
			// 2048-bit squaring, which are the alternatives.

			// Multiply by the low 408 bits of the signature block.
			mpn_mul(next, prev, KEY_LIMB_SIZE, paddingNegShort, countof(paddingNegShort));

			// Multiplying by paddingBase is a left shift.
			prev[(PADDING_SHIFT_BITS / LIMB_BITS) + KEY_LIMB_SIZE] = mpn_lshift(&prev[PADDING_SHIFT_BITS / LIMB_BITS], prev, KEY_LIMB_SIZE, PADDING_SHIFT_BITS % LIMB_BITS);
			std::memset(prev, 0, (PADDING_SHIFT_BITS / LIMB_BITS) * sizeof(prev[0]));

			// Subtract off the result of the 2048x408-bit multiply.
			mpn_sub(next, prev, 2 * KEY_LIMB_SIZE, next, KEY_LIMB_SIZE + countof(paddingNegShort));

		#else  // SIGNATURE_IS_PKCS1

			// When s_signature is not a valid PKCS #1 signature modulo s_modulus, we can't do
			// the above trick and instead must do an ordinary multiply.
			mpn_mul_n(next, prev, block, KEY_LIMB_SIZE);

		#endif

			// Reduce the product modulo the key modulus.  This is now the dominating force
			// in performance, since it's significantly more expensive than the multiplication
			// trick above.
			Double dummyQuotient;
			mpn_tdiv_qr(dummyQuotient, next, 0, next, 2 * KEY_LIMB_SIZE, modulus, KEY_LIMB_SIZE);
		}

		// Check for matches.
		// Note that we can check the negative for a second result as well, because
		// (-x)^65537 == -(x^65537) for any x (because 65537 is odd).
		// Someone in a math IRC channel pointed out that this is equivalent to multiplication
		// by another number whose 65537th root we know: namely, -1, whose root is itself.
		Double *match = nullptr;
		bool negative = false;

		// TODO: A very slightly faster implementation would be to check whether the highest
		// limb could possibly be 0x0002____ when negated, avoiding subtraction.  This can
		// only happen when the highest limb is 0xDECD____ or 0xDECC____.
		if (IsWhatWeWant(GetByteFromArrayWrapper(next)))
		{
			match = &next;
			negative = false;
		}
		else
		{
			// Calculate the negative mod modulus.
			mpn_sub_n(prev, modulus, next, KEY_LIMB_SIZE);
			if (IsWhatWeWant(GetByteFromArrayWrapper(prev)))
			{
				match = &prev;
				negative = true;
			}
		}

		// Is this what we want?
		if (match)
		{
			// Calculate randomBase*(root^iteration).
			mpz_set(mpzFoundBase, mpzRandomBase);

			unsigned long long squareCount = iteration % ATTEMPTS_PER_RANDOM;
			for (unsigned long long i = 0; i < squareCount; ++i)
			{
				mpz_mul(mpzFoundBase, mpzFoundBase, mpzRoot);
				mpz_mod(mpzFoundBase, mpzFoundBase, mpzModulus);
			}

			// If it's the negative case, negate the base.
			if (negative)
			{
				mpz_sub(mpzFoundBase, mpzModulus, mpzFoundBase);
			}

			// Calculate foundBase^65537.
			mpz_powm_ui(mpzFoundPowered, mpzFoundBase, s_publicExponent, mpzModulus);

			// Export these numbers.
			Single foundBase;
			ToLimbArray(foundBase, mpzFoundBase);
			Single foundPowered;
			ToLimbArray(foundPowered, mpzFoundPowered);

			// Verify that foundPowered matches the number we found.
			if (std::memcmp(foundPowered, *match, sizeof(Single)) != 0)
			{
				std::printf("Exponential build error\n");
				std::fflush(stdout);
				std::exit(1);
			}

			std::printf("Match found!!  (mode=%s)\n", (!negative) ? "positive" : "negative");
			std::printf("iteration = %llu\n", iteration);
			std::printf("Result:\n");
			DumpNumber(mpzFoundPowered);
			std::printf("Signature:\n");
			DumpNumber(mpzFoundBase);
			std::fflush(stdout);
		}

		iteration = iteration;
	}

	mpz_clears(mpzRandomBase, mpzRandomPowered, mpzFoundBase, mpzFoundPowered, nullptr);
}


void Profile(void(*function)(mpz_t, mpz_t, mpz_t, unsigned long long), mpz_t modulus, mpz_t root, mpz_t block)
{
	enum : unsigned long long { NUM_ITERATIONS = 10000000 };

	std::clock_t start = std::clock();

	function(modulus, root, block, NUM_ITERATIONS);

	std::clock_t end = std::clock();

	// Multiplying NUM_ITERATIONS by 2 because the brute forcer is checking
	// positive and negative numbers each iteration.
	std::printf("%llu operations in %g seconds (%llu/second)\n",
		static_cast<unsigned long long>(NUM_ITERATIONS * 2),
		static_cast<double>(end - start) / CLOCKS_PER_SEC,
		static_cast<unsigned long long>((NUM_ITERATIONS * 2) / (static_cast<double>(end - start) / CLOCKS_PER_SEC)));
}


// Lowers this program's priority to idle, so that it won't interfere with normal use
// of the computer.
void BackgroundMode()
{
#ifdef _WIN32
	SetPriorityClass(GetCurrentProcess(), IDLE_PRIORITY_CLASS | PROCESS_MODE_BACKGROUND_BEGIN);
#else
	setpriority(PRIO_PROCESS, getpid(), 19);
#endif
}


int main()
{
	BackgroundMode();

	mpz_t modulus;
	mpz_init_set_str(modulus, s_modulus, 16);

	mpz_t root;
	mpz_init_set_str(root, s_root, 16);

	mpz_t block;
	mpz_init(block);

	mpz_powm_ui(block, root, s_publicExponent, modulus);

	//DumpNumber(sample);

#ifdef PROFILE_MODE
	Profile(BruteForce, modulus, root, block);
#else
	BruteForce(modulus, root, block, -1LL);
#endif

	mpz_clears(modulus, root, block, nullptr);
	return 0;
}
