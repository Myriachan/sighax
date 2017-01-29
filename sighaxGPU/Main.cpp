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
};


class MPZNumber
{
public:
	MPZNumber()
	{
		mpz_init2(m_gmp, MODULUS_BITS);
	}

	~MPZNumber()
	{
		mpz_clear(m_gmp);
	}

	operator std::remove_extent_t<mpz_t> *() { return m_gmp; }

private:
	mpz_t m_gmp;
};


int wmain()
{
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

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		std::fprintf(stderr, "cudaSetDevice failed (%d)!  Do you have a CUDA-capable GPU installed?\n",
			static_cast<int>(cudaStatus));
		return 1;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "cudaDeviceReset failed (%d)!\n", static_cast<int>(cudaStatus));
		return 1;
	}

/*	std::clock_t meowend = std::clock();
	std::clock_t meowelapsed = meowend - meowstart;
	double meowelapsedD = static_cast<double>(meowelapsed) / CLOCKS_PER_SEC;
	std::printf("Timing: %g sec, %llu ops/sec\n", meowelapsedD, static_cast<unsigned long long>(meowcount / meowelapsedD));*/

	return 0;
}
