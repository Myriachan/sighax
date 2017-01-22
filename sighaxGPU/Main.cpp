#include "cuda_runtime.h"

#include <cstdio>
#include <cinttypes>
#include <cstring>
#include <ctime>
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


// Exports a mpz_t as an array of mp_limb_t for mpn_* use.
template <size_t S>
void ToLimbArray(Limb (&limbs)[S], mpz_t number)
{
	if (mpz_sizeinbase(number, 2) > S * std::numeric_limits<Limb>::digits)
	{
		std::abort();
	}

	size_t count;
	mpz_export(limbs, &count, -1, sizeof(Limb), 0, 0, number);
	std::memset(&limbs[count], 0, (S - count) * sizeof(Limb));
}


// Size-checking limb copy.
template <size_t S>
void CopyLimbArray(Limb (&dest)[S], const Limb (&src)[S])
{
	std::memcpy(dest, src, sizeof(dest));
}


template <unsigned S>
union Number
{
	LimbArray<S> m_limbs;
	mp_limb_t m_gmp[(S + (GMP_LIMB_BITS / LIMB_BITS) - 1) / (GMP_LIMB_BITS / LIMB_BITS)];
};


// Makes a new random TransferBlob.
void NewTransferBlob(TransferBlob &blob, mpz_t gmpModulus)
{
	mpz_t gmpNumber;
	mpz_init2(gmpNumber, TOTAL_BITS);

	for (OneIteration &current : blob.m_iterations)
	{
		ReadRandom(current.m_limbs, sizeof(current.m_limbs));

		mpz_import(gmpNumber, countof(current.m_limbs), -1, sizeof(current.m_limbs[0]), 0, 0, current.m_limbs);
		mpz_mod(gmpNumber, gmpNumber, gmpModulus);

		ToLimbArray(current.m_limbs, gmpNumber);
	}

	mpz_clear(gmpNumber);
}


// Verifies that a square operation occurred correctly.
bool VerifyModSquare(const Limb (&root)[LIMB_COUNT], const Limb (&square)[LIMB_COUNT], mpz_t gmpModulus)
{
	mpz_t gmpNumber;
	mpz_init2(gmpNumber, TOTAL_BITS * 2);

	mpz_import(gmpNumber, countof(root), -1, sizeof(root[0]), 0, 0, root);

	for (unsigned x = 0; x < NUM_REPEATS; ++x)
	{
		mpz_mul(gmpNumber, gmpNumber, gmpNumber);
		mpz_mod(gmpNumber, gmpNumber, gmpModulus);
	}

	Limb check[LIMB_COUNT];
	ToLimbArray(check, gmpNumber);

	bool result = std::memcmp(square, check, sizeof(square)) == 0;

	mpz_clear(gmpNumber);
	return result;
}


volatile unsigned g_errorOuterIteration;
volatile unsigned g_errorInnerIteration;
volatile unsigned g_errorThread;
TransferBlob *volatile g_errorSrc;
TransferBlob *volatile g_errorDest;


int wmain()
{
	// Initialize constants.
	mpz_t gmpModulus;
	mpz_init_set_str(gmpModulus, s_modulusText, 16);

	mpz_t gmpInverse;
	mpz_init(gmpInverse);
	mpz_setbit(gmpInverse, 4096);
	mpz_tdiv_q(gmpInverse, gmpInverse, gmpModulus);

	Number<LIMB_COUNT> modulus;
	ToLimbArray(modulus.m_limbs, gmpModulus);

	Number<LIMB_COUNT + 1> inverse;
	ToLimbArray(inverse.m_limbs, gmpInverse);

	// Copy constant parameters.
	MathParameters mathParams;
	CopyLimbArray(mathParams.m_modulus, modulus.m_limbs);
	CopyLimbArray(mathParams.m_inverse, inverse.m_limbs);

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		std::fprintf(stderr, "cudaSetDevice failed (%d)!  Do you have a CUDA-capable GPU installed?\n",
			static_cast<int>(cudaStatus));
		return 1;
	}

	unsigned long long meowcount = 0;
	std::clock_t meowstart = std::clock();

	// Generate NUM_THREADS random numbers.
	TransferBlob blobs[2];

	for (unsigned outerIteration = 0; outerIteration < 100; ++outerIteration)
	{
		NewTransferBlob(blobs[0], gmpModulus);

		for (unsigned innerIteration = 0; innerIteration < 100; ++innerIteration)
		{
			unsigned current = innerIteration % 2;

			TransferBlob &src = blobs[current];
			TransferBlob &dest = blobs[current ^ 1];

			cudaStatus = GPUExecuteOperation(dest, src, mathParams);
			if (cudaStatus != cudaSuccess)
			{
				std::fprintf(stderr, "GPUExecuteOperation failed (%d)!\n", static_cast<int>(cudaStatus));
				return 1;
			}

			meowcount += NUM_THREADS * NUM_REPEATS;

		#if defined(_DEBUG)
			for (unsigned x = 0; x < countof(src.m_iterations); ++x)
			{
				if (!VerifyModSquare(src.m_iterations[x].m_limbs, dest.m_iterations[x].m_limbs, gmpModulus))
				{
					g_errorOuterIteration = outerIteration;
					g_errorInnerIteration = innerIteration;
					g_errorThread = x;
					g_errorSrc = &src;
					g_errorDest = &dest;
					_ReadWriteBarrier();
					__debugbreak();
				}
			}
		#endif
		}
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		std::fprintf(stderr, "cudaDeviceReset failed (%d)!\n", static_cast<int>(cudaStatus));
		return 1;
	}

	std::clock_t meowend = std::clock();
	std::clock_t meowelapsed = meowend - meowstart;
	double meowelapsedD = static_cast<double>(meowelapsed) / CLOCKS_PER_SEC;
	std::printf("Timing: %g sec, %llu ops/sec\n", meowelapsedD, static_cast<unsigned long long>(meowcount / meowelapsedD));

	return 0;
}
