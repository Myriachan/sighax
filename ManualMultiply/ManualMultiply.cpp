#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits>
#include <type_traits>

#ifdef _WIN32
	#define _WIN32_WINNT 0x0601
	#define NOGDI
	#define NOUSER
	#include <Windows.h>
	// Use MPIR in Windows.
	#include <mpir.h>
	#include <intrin.h>
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


typedef std::uint32_t Limb;
typedef std::uint64_t DoubleLimb;

enum { MODULUS_BITS = 2048 };
enum { LIMB_BITS = std::numeric_limits<Limb>::digits };
enum { LIMB_COUNT = MODULUS_BITS / LIMB_BITS };

template <unsigned Bits>
struct BitsToLimbs
{
	static_assert(Bits % LIMB_BITS == 0, "Bits must be a multiple of LIMB_BITS");
	enum { LIMBS = Bits / LIMB_BITS };
};

template <unsigned Limbs>
using LimbArray = Limb[Limbs];


template <typename T, size_t S>
constexpr size_t countof(const T(&)[S])
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


// Simulation.  GPU will use __umulhi or assembly language for this.
inline Limb MultiplyHighHelper(Limb a, Limb b)
{
	return static_cast<Limb>((static_cast<DoubleLimb>(a) * b) >> LIMB_BITS);
}


template <unsigned LimbCount>
void ConditionalSubtract(Limb left[LimbCount], const Limb right[LimbCount], Limb mask)
{
	Limb borrow = 0;

	for (unsigned x = 0; x < LimbCount; ++x)
	{
		DoubleLimb current = left[x];
		current -= borrow;
		current -= (right[x] & mask);
		left[x] = static_cast<Limb>(current);

		borrow = static_cast<Limb>(current >> 32) & 1;
	}
}


template <unsigned Limbs>
int GetIsGreaterOrEqualMask(const Limb left[Limbs], const Limb right[Limbs])
{
	Limb borrow = 0;

	for (unsigned x = 0; x < Limbs; ++x)
	{
		DoubleLimb current = left[x];
		current -= borrow;
		current -= right[x];

		borrow = static_cast<Limb>(current >> 32) & 1;
	}

	return borrow - 1;
}


// From Kaiyong Zhao's Algorithm 3.
template <unsigned LimbCount, bool Meow>
void MontgomeryModularMultiply(Limb output[LimbCount + 2], const Limb a[LimbCount], const Limb b[LimbCount], const Limb modulus[LimbCount], Limb modulusInverseR0)
{
	for (unsigned i = 0; i < LimbCount; ++i)
	{
		// Multiplication loop.
		if (i == 0 && Meow)
		{
			// In the first round, we need to initialize the output.
			// The document says that "t" (output) needs to be initialized to
			// zero, but we can just set it in this loop instead of adding.
			// This conditional branch will be synched across threads.
			Limb carry = 0;
			for (unsigned j = 0; j < LimbCount; ++j)
			{
				Limb low = a[j] * b[i];
				Limb high = MultiplyHighHelper(a[j], b[i]);
				DoubleLimb product = (static_cast<DoubleLimb>(high) << 32) + low;
				product += carry;
				output[j] = static_cast<Limb>(product);
				carry = static_cast<Limb>(product >> 32);
			}
			output[LimbCount] = carry;
			output[LimbCount + 1] = 0;
		}
		else
		{
			Limb carry = 0;
			for (unsigned j = 0; j < LimbCount; ++j)
			{
				Limb low = a[j] * b[i];
				Limb high = MultiplyHighHelper(a[j], b[i]);
				DoubleLimb product = (static_cast<DoubleLimb>(high) << 32) + low;
				product += carry;
				product += output[j];
				output[j] = static_cast<Limb>(product);
				carry = static_cast<Limb>(product >> 32);
			}
			DoubleLimb sum = output[LimbCount];
			sum += carry;
			output[LimbCount] = static_cast<Limb>(sum);
			output[LimbCount + 1] = static_cast<Limb>(sum >> 32);
		}

		// Reduction loop.
		// NOTE: This comes from the optimized version in the document
		// discussed below the algorithm itself.
		Limb carry = 0;
		Limb m = 1u * output[0] * modulusInverseR0;

		// First limb, which gets discarded in the end.
		{
			Limb low = m * modulus[0];
			Limb high = MultiplyHighHelper(m, modulus[0]);
			DoubleLimb product = (static_cast<DoubleLimb>(high) << 32) + low;
			product += output[0];
			carry = static_cast<Limb>(product >> 32);
		}

		// The rest of the limbs.
		for (unsigned j = 1; j < LimbCount; ++j)
		{
			Limb low = m * modulus[j];
			Limb high = MultiplyHighHelper(m, modulus[j]);
			DoubleLimb product = (static_cast<DoubleLimb>(high) << 32) + low;
			product += carry;
			product += output[j];
			output[j - 1] = static_cast<Limb>(product);
			carry = static_cast<Limb>(product >> 32);
		}

		DoubleLimb sum = output[LimbCount];
		sum += carry;
		output[LimbCount - 1] = static_cast<Limb>(sum);
		output[LimbCount] = output[LimbCount + 1] + static_cast<Limb>(sum >> 32);
	}

	// The result might be larger than the modulus.  If so, subtract the modulus.
	Limb subtractMask = GetIsGreaterOrEqualMask<LimbCount>(output, modulus);

	// output[LimbCount] could also be 1, which obviously makes it larger.
	subtractMask |= 0u - output[LimbCount];

	ConditionalSubtract<LimbCount>(output, modulus, subtractMask);
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


int main()
{
	// Initialize constants.
	MPZNumber gmpModulus;
	mpz_set_str(gmpModulus, s_modulusText, 16);

	MPZNumber gmpSignature;
	mpz_set_str(gmpSignature, s_signature, 16);

	MPZNumber gmpBlock;
	mpz_powm_ui(gmpBlock, gmpSignature, s_publicExponent, gmpModulus);

	MPZNumber gmpR;
	mpz_setbit(gmpR, MODULUS_BITS);

	MPZNumber gmpInverse;
	mpz_invert(gmpInverse, gmpModulus, gmpR);
	mpz_sub(gmpInverse, gmpR, gmpInverse);

	MPZNumber gmpRModModulus;
	mpz_mod(gmpRModModulus, gmpR, gmpModulus);

	MPZNumber gmpRInverse;
	mpz_invert(gmpRInverse, gmpRModModulus, gmpModulus);

	Number<LIMB_COUNT> modulus;
	ToLimbArray(modulus.m_gmp, gmpModulus);

	Number<LIMB_COUNT> inverse;
	ToLimbArray(inverse.m_gmp, gmpInverse);

	Limb modulusInverseR0 = inverse.m_limbs[0];

	for (unsigned x = 0; x < 100000; ++x)
	{
		// Generate random numbers a and b.
		MPZNumber gmpA;
		Number<LIMB_COUNT> a;
		ReadRandom(a.m_limbs, sizeof(a.m_limbs));
		FromLimbArray(gmpA, a.m_gmp);
		mpz_mod(gmpA, gmpA, gmpModulus);
		ToLimbArray(a.m_gmp, gmpA);

		MPZNumber gmpB;
		Number<LIMB_COUNT> b;
		ReadRandom(b.m_limbs, sizeof(b.m_limbs));
		FromLimbArray(gmpB, b.m_gmp);
		mpz_mod(gmpB, gmpB, gmpModulus);
		ToLimbArray(b.m_gmp, gmpB);

		MPZNumber gmpCheck;
		Number<LIMB_COUNT> check;
		mpz_mul(gmpCheck, gmpA, gmpB);
		mpz_mod(gmpCheck, gmpCheck, gmpModulus);
		mpz_mul(gmpCheck, gmpCheck, gmpRInverse);
		mpz_mod(gmpCheck, gmpCheck, gmpModulus);
		ToLimbArray(check.m_gmp, gmpCheck);

		Limb output[LIMB_COUNT + 2];
		std::memset(output, 0, sizeof(output));
		MontgomeryModularMultiply<LIMB_COUNT, false>(output, a.m_limbs, b.m_limbs, modulus.m_limbs, modulusInverseR0);
		std::memset(output, 0xCC, sizeof(output));
		MontgomeryModularMultiply<LIMB_COUNT, true>(output, a.m_limbs, b.m_limbs, modulus.m_limbs, modulusInverseR0);

		if (std::memcmp(output, check.m_limbs, sizeof(check.m_limbs)) != 0)
		{
			__debugbreak();
		}
	}

	return 0;
}
