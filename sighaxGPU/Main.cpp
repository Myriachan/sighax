#include "cuda_runtime.h"

#include <cstdio>
#include <cinttypes>
#include <cstring>
#include <ctime>
#include <type_traits>
#include "Shared.h"

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


// The 32-bit parts of s_modulus, least-significant first.
#define MODULUS_WORD_00 "0x9BD0EC9BU"
#define MODULUS_WORD_01 "0x045BE017U"
#define MODULUS_WORD_02 "0x3085E8A1U"
#define MODULUS_WORD_03 "0xB2CC3687U"
#define MODULUS_WORD_04 "0x4A7258FDU"
#define MODULUS_WORD_05 "0x6BC95A59U"
#define MODULUS_WORD_06 "0xBD62C246U"
#define MODULUS_WORD_07 "0xA0165CE4U"
#define MODULUS_WORD_08 "0x93AF4D38U"
#define MODULUS_WORD_09 "0x64B2B6C0U"
#define MODULUS_WORD_10 "0x662AE856U"
#define MODULUS_WORD_11 "0x8E4A169DU"
#define MODULUS_WORD_12 "0x9900ABA4U"
#define MODULUS_WORD_13 "0xFDBAD8A9U"
#define MODULUS_WORD_14 "0x9F98C215U"
#define MODULUS_WORD_15 "0xDEDA5C0BU"
#define MODULUS_WORD_16 "0x51B71CE0U"
#define MODULUS_WORD_17 "0xAFF25E06U"
#define MODULUS_WORD_18 "0x3AA949A5U"
#define MODULUS_WORD_19 "0x075D2634U"
#define MODULUS_WORD_20 "0x0154779FU"
#define MODULUS_WORD_21 "0x69616289U"
#define MODULUS_WORD_22 "0x5500377CU"
#define MODULUS_WORD_23 "0xB9ACA6EEU"
#define MODULUS_WORD_24 "0xBB01CB13U"
#define MODULUS_WORD_25 "0xF1ED2B18U"
#define MODULUS_WORD_26 "0x2CA6D7ACU"
#define MODULUS_WORD_27 "0x2D48967CU"
#define MODULUS_WORD_28 "0x9A2E5303U"
#define MODULUS_WORD_29 "0x0D1AF902U"
#define MODULUS_WORD_30 "0xA11B5E9DU"
#define MODULUS_WORD_31 "0xB2C0447EU"
#define MODULUS_WORD_32 "0x11D17EABU"
#define MODULUS_WORD_33 "0x4984E094U"
#define MODULUS_WORD_34 "0x9A29295EU"
#define MODULUS_WORD_35 "0x1AAFB348U"
#define MODULUS_WORD_36 "0x040F02A8U"
#define MODULUS_WORD_37 "0xA47EDD19U"
#define MODULUS_WORD_38 "0x991D31B8U"
#define MODULUS_WORD_39 "0x32C4DCA9U"
#define MODULUS_WORD_40 "0x71ECC8C1U"
#define MODULUS_WORD_41 "0x41EEE4F0U"
#define MODULUS_WORD_42 "0x0BC67343U"
#define MODULUS_WORD_43 "0x526370F0U"
#define MODULUS_WORD_44 "0x75887793U"
#define MODULUS_WORD_45 "0xF0170CB5U"
#define MODULUS_WORD_46 "0x604BC7F7U"
#define MODULUS_WORD_47 "0x62B730F4U"
#define MODULUS_WORD_48 "0x8182B681U"
#define MODULUS_WORD_49 "0xE64DB123U"
#define MODULUS_WORD_50 "0x6B458221U"
#define MODULUS_WORD_51 "0x253CBE37U"
#define MODULUS_WORD_52 "0x0FDE451BU"
#define MODULUS_WORD_53 "0xC02D4D41U"
#define MODULUS_WORD_54 "0x2ED88F92U"
#define MODULUS_WORD_55 "0x962EC4E9U"
#define MODULUS_WORD_56 "0x336FB076U"
#define MODULUS_WORD_57 "0xA2A08CBBU"
#define MODULUS_WORD_58 "0x72707932U"
#define MODULUS_WORD_59 "0xA16B9AABU"
#define MODULUS_WORD_60 "0x8817B003U"
#define MODULUS_WORD_61 "0xFDAC90E8U"
#define MODULUS_WORD_62 "0x3D33E955U"
#define MODULUS_WORD_63 "0xDECFB6FCU"

// The low 32 bits of R - (modulus^-1) mod R.
#define MODULUS_INVERSE_LOW "0x85E8E66DU"

// The 32-bit parts of MULTIPLIER_MONTGOMERY, least-significant first.
#define MULTIPLIER_WORD_00 "0xB5F30EF7U"
#define MULTIPLIER_WORD_01 "0x1020BA70U"
#define MULTIPLIER_WORD_02 "0xFA2C6D29U"
#define MULTIPLIER_WORD_03 "0x22B71AECU"
#define MULTIPLIER_WORD_04 "0xEFBED8EEU"
#define MULTIPLIER_WORD_05 "0x71E2EEC1U"
#define MULTIPLIER_WORD_06 "0x17296E7CU"
#define MULTIPLIER_WORD_07 "0x6824C5EAU"
#define MULTIPLIER_WORD_08 "0x468A8D87U"
#define MULTIPLIER_WORD_09 "0x29EBC942U"
#define MULTIPLIER_WORD_10 "0xB48A741EU"
#define MULTIPLIER_WORD_11 "0xC2E095FBU"
#define MULTIPLIER_WORD_12 "0xFB66B5CAU"
#define MULTIPLIER_WORD_13 "0xE9F4FFC9U"
#define MULTIPLIER_WORD_14 "0xF9A20DB7U"
#define MULTIPLIER_WORD_15 "0x995AB257U"
#define MULTIPLIER_WORD_16 "0x395BF3CCU"
#define MULTIPLIER_WORD_17 "0x385EBC48U"
#define MULTIPLIER_WORD_18 "0x2DB55CBFU"
#define MULTIPLIER_WORD_19 "0xD3B4BF5AU"
#define MULTIPLIER_WORD_20 "0xEC8F0FFBU"
#define MULTIPLIER_WORD_21 "0x268291ADU"
#define MULTIPLIER_WORD_22 "0xAFD05422U"
#define MULTIPLIER_WORD_23 "0xCFE89D5FU"
#define MULTIPLIER_WORD_24 "0xF5D73ABBU"
#define MULTIPLIER_WORD_25 "0xD3BDEF92U"
#define MULTIPLIER_WORD_26 "0x95A383E1U"
#define MULTIPLIER_WORD_27 "0x0C70076CU"
#define MULTIPLIER_WORD_28 "0x7EA0B490U"
#define MULTIPLIER_WORD_29 "0xBD967333U"
#define MULTIPLIER_WORD_30 "0x6B37B4DAU"
#define MULTIPLIER_WORD_31 "0x6A402853U"
#define MULTIPLIER_WORD_32 "0x7752F30DU"
#define MULTIPLIER_WORD_33 "0xBE8D824CU"
#define MULTIPLIER_WORD_34 "0x68EE4D1EU"
#define MULTIPLIER_WORD_35 "0x9A18D6F8U"
#define MULTIPLIER_WORD_36 "0x9583AA49U"
#define MULTIPLIER_WORD_37 "0xAB129DBFU"
#define MULTIPLIER_WORD_38 "0x2FD7E863U"
#define MULTIPLIER_WORD_39 "0x08D8B04FU"
#define MULTIPLIER_WORD_40 "0x474FE9E0U"
#define MULTIPLIER_WORD_41 "0xDEFB34FCU"
#define MULTIPLIER_WORD_42 "0xE87C6774U"
#define MULTIPLIER_WORD_43 "0x43C85CA9U"
#define MULTIPLIER_WORD_44 "0xC1D24733U"
#define MULTIPLIER_WORD_45 "0xF0298AB0U"
#define MULTIPLIER_WORD_46 "0x696B969AU"
#define MULTIPLIER_WORD_47 "0x2EDA7839U"
#define MULTIPLIER_WORD_48 "0x02ADD7F1U"
#define MULTIPLIER_WORD_49 "0xBAB19C9CU"
#define MULTIPLIER_WORD_50 "0xAC1B4D12U"
#define MULTIPLIER_WORD_51 "0x38DCAE67U"
#define MULTIPLIER_WORD_52 "0x5C6D352AU"
#define MULTIPLIER_WORD_53 "0x05B0ADB2U"
#define MULTIPLIER_WORD_54 "0x48E344A7U"
#define MULTIPLIER_WORD_55 "0xFC175DC2U"
#define MULTIPLIER_WORD_56 "0xAC1B1DD3U"
#define MULTIPLIER_WORD_57 "0x3486B994U"
#define MULTIPLIER_WORD_58 "0x86448BDBU"
#define MULTIPLIER_WORD_59 "0xCEB1823FU"
#define MULTIPLIER_WORD_60 "0x744D79A9U"
#define MULTIPLIER_WORD_61 "0xCEF43461U"
#define MULTIPLIER_WORD_62 "0x6FE966F3U"
#define MULTIPLIER_WORD_63 "0x43E70197U"


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

	static void Dummy()
	{
		static_assert(sizeof(m_limbs) == sizeof(m_gmp), "Unsupported limb size");
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


int wmain()
{
	// Initialize random generator.
	ReadRandom(xorshift_s, sizeof(xorshift_s));

	// Initialize constants.
	MPZNumber gmpModulus;
	mpz_set_str(gmpModulus, s_modulusText, 16);

	// Initialize constants.
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

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		std::fprintf(stderr, "cudaSetDevice failed (%d)!  Do you have a CUDA-capable GPU installed?\n",
			static_cast<int>(cudaStatus));
		return 1;
	}

	// Allocate our list of bases.
	Number<LIMB_COUNT> *bases = new Number<LIMB_COUNT>[NUM_BLOCKS * NUM_THREADS];

	// Allocate our communication buffers.
	Limb *buffer[2];
	buffer[0] = new Limb[TOTAL_LIMB_COUNT * 2];
	buffer[1] = new Limb[TOTAL_LIMB_COUNT * 2];

	for (;;)
	{
		std::printf("Seeding...\n");
		std::fflush(stdout);

		// Generate new random numbers to use.
		for (unsigned block = 0; block < NUM_BLOCKS; ++block)
		{
			Limb *blockBase = buffer[0] + (block * BLOCK_LIMB_COUNT);
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

		bases = bases;

		std::printf("Searching...\n");
		std::fflush(stdout);

		std::clock_t meowStart = std::clock();
		unsigned long long total = 0;
		unsigned long long check = 0;

		for (unsigned round = 0; round < 1000; ++round)
		{
			//std::printf("Executing round %u...\n", round);
			//std::fflush(stdout);

			unsigned currentBuffer = round & 1;
			unsigned nextBuffer = currentBuffer ^ 1;

			GPUExecuteOperation(buffer[nextBuffer], buffer[currentBuffer]);

			total += NUM_BLOCKS * NUM_THREADS;
			check += buffer[currentBuffer][12345];
		}

		std::clock_t meowEnd = std::clock();

		std::printf("check: %016llX\n", check);
		std::printf("%llu operations in %g seconds (%llu/second)\n",
			total,
			static_cast<double>(meowEnd - meowStart) / CLOCKS_PER_SEC,
			static_cast<unsigned long long>(total / (static_cast<double>(meowEnd - meowStart) / CLOCKS_PER_SEC)));

		break;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "cudaDeviceReset failed (%d)!\n", static_cast<int>(cudaStatus));
		return 1;
	}

	delete[] buffer[0];
	delete[] buffer[1];
	delete[] bases;
	return 0;
}
