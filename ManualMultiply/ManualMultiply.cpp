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

enum { LIMB_BITS = std::numeric_limits<Limb>::digits };
enum { LIMB_COUNT = 2048 / LIMB_BITS };

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


Limb MultiplyHighHelper(Limb a, Limb b)
{
	return static_cast<Limb>((static_cast<DoubleLimb>(a) * b) >> LIMB_BITS);
}


Limb AddFullHelper(Limb &dest, Limb a, Limb b, Limb c)
{
	DoubleLimb result = a;
	result += b;
	result += c;
	dest = static_cast<Limb>(result);
	return static_cast<Limb>(result >> 32);
}


Limb SubFullHelper(Limb &dest, Limb a, Limb b, Limb c)
{
	DoubleLimb result = a;
	result -= b;
	result -= c;
	dest = static_cast<Limb>(result);
	return static_cast<Limb>(result >> 32) & 1;
}


template <unsigned Limbs>
int BigCompareN(const Limb left[Limbs], const Limb right[Limbs])
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
Limb BigSubtractN(Limb dest[Limbs], const Limb left[Limbs], const Limb right[Limbs], Limb borrow)
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
void BigMultiplyMN(Limb dest[LeftLimbs + RightLimbs], const Limb left[LeftLimbs], const Limb right[RightLimbs])
{
	static_assert(LeftLimbs > 0, "invalid LeftLimbs");
	static_assert(RightLimbs > 0, "invalid RightLimbs");

	DoubleLimb result[LeftLimbs + RightLimbs] = {};

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
void BarrettReduce(Limb remainder[Limbs], const Limb dividend[Limbs * 2], const Limb modulus[Limbs], const Limb inverse[Limbs + 1])
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

template <unsigned S>
union Number
{
	LimbArray<S> m_limbs;
	mp_limb_t m_gmp[(S + (GMP_LIMB_BITS / LIMB_BITS) - 1) / (GMP_LIMB_BITS / LIMB_BITS)];
};


int main()
{
	// Initialize constants.
	mpz_t gmpModulus;
	mpz_init_set_str(gmpModulus, s_modulusText, 16);

	mpz_t gmpSignature;
	mpz_init_set_str(gmpSignature, s_signature, 16);

	mpz_t gmpBlock;
	mpz_init(gmpBlock);
	mpz_powm_ui(gmpBlock, gmpSignature, s_publicExponent, gmpModulus);

	mpz_t gmpInverse;
	mpz_init(gmpInverse);
	mpz_setbit(gmpInverse, 4096);
	mpz_tdiv_q(gmpInverse, gmpInverse, gmpModulus);

	Number<LIMB_COUNT> modulus;
	ToLimbArray(modulus.m_gmp, gmpModulus);

	Number<LIMB_COUNT> signature;
	ToLimbArray(signature.m_gmp, gmpSignature);

	Number<LIMB_COUNT> block;
	ToLimbArray(block.m_gmp, gmpBlock);

	Number<LIMB_COUNT + 1> inverse;
	ToLimbArray(inverse.m_gmp, gmpInverse);

	// Main code
	for (unsigned x = 0; x < 100000; ++x)
	{
		Number<2048 / LIMB_BITS> multiplicand;
		Number<4096 / LIMB_BITS> mySquare;
		Number<4096 / LIMB_BITS> gmpSquare;
		Number<2048 / LIMB_BITS> myResult;
		Number<2048 / LIMB_BITS> gmpResult;
		Number<4096 / LIMB_BITS> dummyQuotient;

		//std::memset(multiplicand.m_limbs, 0xFF, sizeof(multiplicand.m_limbs));
		if (x & 1)
			mpn_random(multiplicand.m_gmp, countof(multiplicand.m_gmp));
		else
			mpn_random2(multiplicand.m_gmp, countof(multiplicand.m_gmp));

		BigMultiplyMN<2048 / LIMB_BITS, 2048 / LIMB_BITS>(mySquare.m_limbs, multiplicand.m_limbs, multiplicand.m_limbs);
		BarrettReduce<2048 / LIMB_BITS>(myResult.m_limbs, mySquare.m_limbs, modulus.m_limbs, inverse.m_limbs);

		mpn_sqr(gmpSquare.m_gmp, multiplicand.m_gmp, countof(multiplicand.m_gmp));
		mpn_tdiv_qr(dummyQuotient.m_gmp, gmpResult.m_gmp, 0, gmpSquare.m_gmp, countof(gmpSquare.m_gmp), modulus.m_gmp, countof(modulus.m_gmp));

		if (std::memcmp(mySquare.m_gmp, gmpSquare.m_gmp, sizeof(mySquare.m_gmp)))
		{
			__debugbreak();
		}
		if (std::memcmp(myResult.m_gmp, gmpResult.m_gmp, sizeof(myResult.m_gmp)))
		{
			__debugbreak();
		}
	}

	mpz_clears(gmpModulus, gmpSignature, gmpBlock, gmpInverse, nullptr);

	return 0;
}
