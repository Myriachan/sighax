#include "cuda_runtime.h"

#include <cstdio>
#include <cinttypes>
#include <cstring>
#include <ctime>
#include <type_traits>
#include "Shared.h"

#ifdef _DEBUG
	#define ENABLE_VERIFY_MODE
#endif
//#define PROFILE_MODE

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

// A silly string we're using as the 65537th root.
const char s_root[] =
    "4D454F574D454F574D454F574D454F574D454F574D454F574D454F574D454F57"
	"4D454F574D454F574D454F574D454F574D454F574D454F574D454F574D454F57"
	"4D454F574D454F574D454F574D454F574D454F574D454F574D454F574D454F57"
	"4D454F574D454F574D454F574D454F574D454F574D454F574D454F574D454F57"
	"4D454F574D454F574D454F574D454F574D454F574D454F574D454F574D454F57"
	"4D454F574D454F574D454F574D454F574D454F574D454F574D454F574D454F57"
	"4D454F574D454F574D454F574D454F574D454F574D454F574D454F574D454F57"
	"4D454F574D454F574D454F574D454F574D454F574D454F574D454F574D454F57";

// s_root to the 65537th power.
const char s_multiplier[] =
	"BAEAB32E2B70904E0C9483DE5728A8C52F14F1B0A655A361B46266C4429B7430"
	"7B919A8D447AFAD867068AE6C39B9DE75FA3562137B2BF14567670461AEB98ED"
	"9A1842429AA82F3BAD46B7F817C692F0810990E8A1A4F72C6DF997DBFB36E327"
	"C8A626BF7CE7A73948C41C572DFE5F28EF964FFA1BB5E14AB28B61688A779E11"
	"D63FA56FBC3CA18B36B5EA57FC264EB23671C2928498CFBA0B9380036EBB9BB4"
	"80E8A1BACEB7B34878FDA227B2E22449237CAE4687C7606A58A539C3C0580201"
	"53FC00B4FE1C3913DF6EA63293F8DB013E848EBBA43A00B4B7E4B80319B02C6F"
	"99A8D672BC76035F76513A67F534525509E9D0DA8782B153DA26ECF2829F9ED5";

// s_multiplier in Montgomery form (i.e., times R).
const char s_multiplierMontgomery[] =
	"43E701976FE966F3CEF43461744D79A9CEB1823F86448BDB3486B994AC1B1DD3"
	"FC175DC248E344A705B0ADB25C6D352A38DCAE67AC1B4D12BAB19C9C02ADD7F1"
	"2EDA7839696B969AF0298AB0C1D2473343C85CA9E87C6774DEFB34FC474FE9E0"
	"08D8B04F2FD7E863AB129DBF9583AA499A18D6F868EE4D1EBE8D824C7752F30D"
	"6A4028536B37B4DABD9673337EA0B4900C70076C95A383E1D3BDEF92F5D73ABB"
	"CFE89D5FAFD05422268291ADEC8F0FFBD3B4BF5A2DB55CBF385EBC48395BF3CC"
	"995AB257F9A20DB7E9F4FFC9FB66B5CAC2E095FBB48A741E29EBC942468A8D87"
	"6824C5EA17296E7C71E2EEC1EFBED8EE22B71AECFA2C6D291020BA70B5F30EF7";


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

/*	// Count how many nonzero bytes are after the 0x02.
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

	// TODO: Rest of implementation.*/

	return true;
}


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
	return IsWhatWeWant(deinterleaved.m_gmp);
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
bool VerifyAndReportMatch(const Number<LIMB_COUNT> &base, mpz_t gmpModulus, mpz_t gmpRoot, unsigned long long round, bool negative)
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

	if (!IsWhatWeWant(result.m_gmp))
	{
		std::printf("Reconstruction error on round %llu\n", round);
		std::fflush(stdout);
		return false;
	}
	else
	{
		std::printf("Match found!\n");

		std::printf("Signature: ");
		signature.Print();
		std::puts("");

		std::printf("Signature: ");
		result.Print();
		std::puts("");

		std::fflush(stdout);
		return true;
	}
}


// Main program.
#ifdef _WIN32
extern "C" int __cdecl wmain()
#else
int main()
#endif
{
	// Initialize random generator.
	ReadRandom(xorshift_s, sizeof(xorshift_s));

	// Initialize constants.
	MPZNumber gmpModulus;
	mpz_set_str(gmpModulus, s_modulusText, 16);

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
	cudaError_t cudaStatus = gpu.Initialize();
	if (cudaStatus != cudaSuccess)
	{
		std::fprintf(stderr, "GPUState::Initialize failed (%d)!\n", static_cast<int>(cudaStatus));
		return 1;
	}

	// Allocate our list of bases.
	Number<LIMB_COUNT> *bases = new Number<LIMB_COUNT>[NUM_BLOCKS * NUM_THREADS];

	// Allocate our communication buffer.
	Limb *buffer = new Limb[TOTAL_LIMB_COUNT];

	for (;;)
	{
		std::printf("Seeding...\n");
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

		std::printf("Searching...\n");
		std::fflush(stdout);

		std::clock_t meowStart = std::clock();
		unsigned long long total = 0;
		unsigned long long check = 0;

		for (unsigned round = 0; round < 1000; ++round)
		{
			//std::printf("Executing round %u...\n", round);
			//std::fflush(stdout);

			cudaStatus = gpu.Execute(round & 1, buffer);
			if (cudaStatus != cudaSuccess)
			{
				std::printf("GPUExecuteOperation failed: %d\n", static_cast<int>(cudaStatus));
				goto done;
			}

			// Check for matches.
			unsigned block;
			unsigned thread;
			bool negative;
			if (SearchForMatches(block, thread, negative, buffer, modulus))
			{
				if (VerifyAndReportMatch(bases[(block * NUM_THREADS) + thread], gmpModulus,
						gmpMultiplierRoot, round, negative))
				{
					goto done;
				}
				else
				{
					std::abort();
				}
			}

			total += NUM_BLOCKS * NUM_THREADS;
			check += buffer[12345];
		}

		std::clock_t meowEnd = std::clock();

		std::printf("check: %016llX\n", check);
		std::printf("%llu operations in %g seconds (%llu/second)\n",
			total,
			static_cast<double>(meowEnd - meowStart) / CLOCKS_PER_SEC,
			static_cast<unsigned long long>(total / (static_cast<double>(meowEnd - meowStart) / CLOCKS_PER_SEC)));

		break;
	}

done:
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "cudaDeviceReset failed (%d)!\n", static_cast<int>(cudaStatus));
		return 1;
	}

	delete[] buffer;
	delete[] bases;
	return 0;
}
