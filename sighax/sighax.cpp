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

// SIGNATURE_IS_PKCS1 means that s_signature, when taken to the 65537th power
// mod the modulus, is a PKCS #1 encoded block.  That is, s_signature is a
// valid signature matching s_modulus.  If you disable this flag, s_signature
// is any number you want; it doesn't need to match the modulus.  Setting this
// flag when it isn't true will cause an "exponential build error" when a
// match is found.
#define SIGNATURE_IS_PKCS1

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

// Useful types.
typedef mp_limb_t Single[KEY_LIMB_SIZE];
typedef mp_limb_t Double[2 * KEY_LIMB_SIZE];
typedef mp_limb_t Quad[4 * KEY_LIMB_SIZE];


// big-endian
const char s_modulusText[] =
	"DECFB6FC3D33E955FDAC90E88817B003A16B9AAB72707932A2A08CBB336FB076"
	"962EC4E92ED88F92C02D4D410FDE451B253CBE376B458221E64DB1238182B681"
	"62B730F4604BC7F7F0170CB575887793526370F00BC6734341EEE4F071ECC8C1"
	"32C4DCA9991D31B8A47EDD19040F02A81AAFB3489A29295E4984E09411D17EAB"
	"B2C0447EA11B5E9D0D1AF9029A2E53032D48967C2CA6D7ACF1ED2B18BB01CB13"
	"B9ACA6EE5500377C696162890154779F075D26343AA949A5AFF25E0651B71CE0"
	"DEDA5C0B9F98C215FDBAD8A99900ABA48E4A169D662AE85664B2B6C093AF4D38"
	"A0165CE4BD62C2466BC95A594A7258FDB2CC36873085E8A1045BE0179BD0EC9B";

const unsigned long s_publicExponent = 65537;

// sample signed message
const char s_signature[] =
	"C26EAACEDD4FF31CD970262B2A6BE06D5CEC1115528CAA6F00BADA3A6B9A886B"
	"5E35DE4FB7E9E4356C4B06B310CCA15AED2B7B433DAB681B0366CC3C769F6D35"
	"79E6B816A8F01BE9C58C1A61A5AB817E2C2FC55C8C70F584D8D485E75584D71A"
	"0EA1A6092751DBE6BCBBE3C119A4CBA5E383E740813129AA4E9CB49DD396BB7F"
	"97F332FAA24F0A4BCBC362E34D4F09F1395B565CC6153D37F057A0496886E66E"
	"965BE08A1030EA038BC45DDF6D4F527F3ED41E2545C0E4772EA6A3F97DD2A0C7"
	"0D340769E8AF211CD1EEB504A96C70B4DE40AD146BF63F509FD56A55358211CC"
	"27A96914769E50864FF4EEA245A5FFA95265D5733EDB0D33D9D1602F5F3CC8E6";


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


// Testing flag.
volatile bool g_meow = false;

// Determine whether the given buffer is what we want.
bool IsWhatWeWant(const mp_limb_t *limbs)
{
	// Test code - used when profiling so that we never find anything.
#ifdef PROFILE_MODE
	if (!g_meow) return false;
#endif

	// For our own sanity, we use big-endian indexing here.
	// It's much easier to conceptualize that way.
	auto getByte = [&limbs](std::size_t index) -> const unsigned char &
	{
		// unsigned char is allowed to alias in C/C++ rules.
		static_assert((KEY_SIZE & (KEY_SIZE - 1)) == 0, "KEY_SIZE must be a power of 2");
		return (reinterpret_cast<const unsigned char *>(limbs))[index ^ (KEY_SIZE - 1)];
	};

	// A match must begin with 00 02.
	if ((getByte(0x00) != 0x00) || (getByte(0x01) != 0x02))
	{
		return false;
	}

	// Count how many nonzero bytes are after the 0x02.
	unsigned zeroIndex;
	for (zeroIndex = 0x02; zeroIndex < KEY_SIZE; ++zeroIndex)
	{
		if (getByte(zeroIndex) == 0x00)
		{
			break;
		}
	}

	if (zeroIndex >= KEY_SIZE)
	{
		return false;
	}

	// TODO: Rest of implementation.

	return true;
}


void BruteForce(mpz_t mpzModulus, mpz_t mpzSignature, mpz_t mpzBlock, unsigned long long numIterations)
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

	// Convert the modulus and signature to mpn form.
	Single modulus;
	ToLimbArray(modulus, mpzModulus);

	Single signature;
	ToLimbArray(signature, mpzSignature);

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
		if (IsWhatWeWant(next))
		{
			match = &next;
			negative = false;
		}
		else
		{
			// Calculate the negative mod modulus.
			mpn_sub_n(prev, modulus, next, KEY_LIMB_SIZE);
			if (IsWhatWeWant(prev))
			{
				match = &prev;
				negative = true;
			}
		}

		// Is this what we want?
		if (match)
		{
			// Calculate randomBase*(signature^iteration).
			mpz_set(mpzFoundBase, mpzRandomBase);

			unsigned long long squareCount = iteration % ATTEMPTS_PER_RANDOM;
			for (unsigned long long i = 0; i < squareCount; ++i)
			{
				mpz_mul(mpzFoundBase, mpzFoundBase, mpzSignature);
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
			std::printf("Buffer:\n");
			DumpNumber(mpzFoundPowered);
			std::printf("Signature:\n");
			DumpNumber(mpzFoundBase);
			std::fflush(stdout);
		}

		iteration = iteration;
	}

	mpz_clears(mpzRandomBase, mpzRandomPowered, mpzFoundBase, mpzFoundPowered, nullptr);
}


void Profile(void(*function)(mpz_t, mpz_t, mpz_t, unsigned long long), mpz_t modulus, mpz_t signature, mpz_t block)
{
	enum : unsigned long long { NUM_ITERATIONS = 10000000 };

	std::clock_t start = std::clock();

	function(modulus, signature, block, NUM_ITERATIONS);

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
	mpz_init_set_str(modulus, s_modulusText, 16);

	mpz_t signature;
	mpz_init_set_str(signature, s_signature, 16);

	mpz_t block;
	mpz_init(block);

	mpz_powm_ui(block, signature, s_publicExponent, modulus);

	//DumpNumber(sample);

#ifdef PROFILE_MODE
	Profile(BruteForce, modulus, signature, block);
#else
	BruteForce(modulus, signature, block, 10000000);
#endif

	mpz_clears(modulus, signature, block, nullptr);
	return 0;
}
