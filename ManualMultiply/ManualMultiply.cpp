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
#define MODULUS_WORD_00 0x9BD0EC9BU
#define MODULUS_WORD_01 0x045BE017U
#define MODULUS_WORD_02 0x3085E8A1U
#define MODULUS_WORD_03 0xB2CC3687U
#define MODULUS_WORD_04 0x4A7258FDU
#define MODULUS_WORD_05 0x6BC95A59U
#define MODULUS_WORD_06 0xBD62C246U
#define MODULUS_WORD_07 0xA0165CE4U
#define MODULUS_WORD_08 0x93AF4D38U
#define MODULUS_WORD_09 0x64B2B6C0U
#define MODULUS_WORD_10 0x662AE856U
#define MODULUS_WORD_11 0x8E4A169DU
#define MODULUS_WORD_12 0x9900ABA4U
#define MODULUS_WORD_13 0xFDBAD8A9U
#define MODULUS_WORD_14 0x9F98C215U
#define MODULUS_WORD_15 0xDEDA5C0BU
#define MODULUS_WORD_16 0x51B71CE0U
#define MODULUS_WORD_17 0xAFF25E06U
#define MODULUS_WORD_18 0x3AA949A5U
#define MODULUS_WORD_19 0x075D2634U
#define MODULUS_WORD_20 0x0154779FU
#define MODULUS_WORD_21 0x69616289U
#define MODULUS_WORD_22 0x5500377CU
#define MODULUS_WORD_23 0xB9ACA6EEU
#define MODULUS_WORD_24 0xBB01CB13U
#define MODULUS_WORD_25 0xF1ED2B18U
#define MODULUS_WORD_26 0x2CA6D7ACU
#define MODULUS_WORD_27 0x2D48967CU
#define MODULUS_WORD_28 0x9A2E5303U
#define MODULUS_WORD_29 0x0D1AF902U
#define MODULUS_WORD_30 0xA11B5E9DU
#define MODULUS_WORD_31 0xB2C0447EU
#define MODULUS_WORD_32 0x11D17EABU
#define MODULUS_WORD_33 0x4984E094U
#define MODULUS_WORD_34 0x9A29295EU
#define MODULUS_WORD_35 0x1AAFB348U
#define MODULUS_WORD_36 0x040F02A8U
#define MODULUS_WORD_37 0xA47EDD19U
#define MODULUS_WORD_38 0x991D31B8U
#define MODULUS_WORD_39 0x32C4DCA9U
#define MODULUS_WORD_40 0x71ECC8C1U
#define MODULUS_WORD_41 0x41EEE4F0U
#define MODULUS_WORD_42 0x0BC67343U
#define MODULUS_WORD_43 0x526370F0U
#define MODULUS_WORD_44 0x75887793U
#define MODULUS_WORD_45 0xF0170CB5U
#define MODULUS_WORD_46 0x604BC7F7U
#define MODULUS_WORD_47 0x62B730F4U
#define MODULUS_WORD_48 0x8182B681U
#define MODULUS_WORD_49 0xE64DB123U
#define MODULUS_WORD_50 0x6B458221U
#define MODULUS_WORD_51 0x253CBE37U
#define MODULUS_WORD_52 0x0FDE451BU
#define MODULUS_WORD_53 0xC02D4D41U
#define MODULUS_WORD_54 0x2ED88F92U
#define MODULUS_WORD_55 0x962EC4E9U
#define MODULUS_WORD_56 0x336FB076U
#define MODULUS_WORD_57 0xA2A08CBBU
#define MODULUS_WORD_58 0x72707932U
#define MODULUS_WORD_59 0xA16B9AABU
#define MODULUS_WORD_60 0x8817B003U
#define MODULUS_WORD_61 0xFDAC90E8U
#define MODULUS_WORD_62 0x3D33E955U
#define MODULUS_WORD_63 0xDECFB6FCU

// The low 32 bits of R - (modulus^-1) mod R.
#define MODULUS_INVERSE_LOW 0x85E8E66DU

// The 32-bit parts of MULTIPLIER_MONTGOMERY, least-significant first.
#define MULTIPLIER_WORD_00 0xB5F30EF7U
#define MULTIPLIER_WORD_01 0x1020BA70U
#define MULTIPLIER_WORD_02 0xFA2C6D29U
#define MULTIPLIER_WORD_03 0x22B71AECU
#define MULTIPLIER_WORD_04 0xEFBED8EEU
#define MULTIPLIER_WORD_05 0x71E2EEC1U
#define MULTIPLIER_WORD_06 0x17296E7CU
#define MULTIPLIER_WORD_07 0x6824C5EAU
#define MULTIPLIER_WORD_08 0x468A8D87U
#define MULTIPLIER_WORD_09 0x29EBC942U
#define MULTIPLIER_WORD_10 0xB48A741EU
#define MULTIPLIER_WORD_11 0xC2E095FBU
#define MULTIPLIER_WORD_12 0xFB66B5CAU
#define MULTIPLIER_WORD_13 0xE9F4FFC9U
#define MULTIPLIER_WORD_14 0xF9A20DB7U
#define MULTIPLIER_WORD_15 0x995AB257U
#define MULTIPLIER_WORD_16 0x395BF3CCU
#define MULTIPLIER_WORD_17 0x385EBC48U
#define MULTIPLIER_WORD_18 0x2DB55CBFU
#define MULTIPLIER_WORD_19 0xD3B4BF5AU
#define MULTIPLIER_WORD_20 0xEC8F0FFBU
#define MULTIPLIER_WORD_21 0x268291ADU
#define MULTIPLIER_WORD_22 0xAFD05422U
#define MULTIPLIER_WORD_23 0xCFE89D5FU
#define MULTIPLIER_WORD_24 0xF5D73ABBU
#define MULTIPLIER_WORD_25 0xD3BDEF92U
#define MULTIPLIER_WORD_26 0x95A383E1U
#define MULTIPLIER_WORD_27 0x0C70076CU
#define MULTIPLIER_WORD_28 0x7EA0B490U
#define MULTIPLIER_WORD_29 0xBD967333U
#define MULTIPLIER_WORD_30 0x6B37B4DAU
#define MULTIPLIER_WORD_31 0x6A402853U
#define MULTIPLIER_WORD_32 0x7752F30DU
#define MULTIPLIER_WORD_33 0xBE8D824CU
#define MULTIPLIER_WORD_34 0x68EE4D1EU
#define MULTIPLIER_WORD_35 0x9A18D6F8U
#define MULTIPLIER_WORD_36 0x9583AA49U
#define MULTIPLIER_WORD_37 0xAB129DBFU
#define MULTIPLIER_WORD_38 0x2FD7E863U
#define MULTIPLIER_WORD_39 0x08D8B04FU
#define MULTIPLIER_WORD_40 0x474FE9E0U
#define MULTIPLIER_WORD_41 0xDEFB34FCU
#define MULTIPLIER_WORD_42 0xE87C6774U
#define MULTIPLIER_WORD_43 0x43C85CA9U
#define MULTIPLIER_WORD_44 0xC1D24733U
#define MULTIPLIER_WORD_45 0xF0298AB0U
#define MULTIPLIER_WORD_46 0x696B969AU
#define MULTIPLIER_WORD_47 0x2EDA7839U
#define MULTIPLIER_WORD_48 0x02ADD7F1U
#define MULTIPLIER_WORD_49 0xBAB19C9CU
#define MULTIPLIER_WORD_50 0xAC1B4D12U
#define MULTIPLIER_WORD_51 0x38DCAE67U
#define MULTIPLIER_WORD_52 0x5C6D352AU
#define MULTIPLIER_WORD_53 0x05B0ADB2U
#define MULTIPLIER_WORD_54 0x48E344A7U
#define MULTIPLIER_WORD_55 0xFC175DC2U
#define MULTIPLIER_WORD_56 0xAC1B1DD3U
#define MULTIPLIER_WORD_57 0x3486B994U
#define MULTIPLIER_WORD_58 0x86448BDBU
#define MULTIPLIER_WORD_59 0xCEB1823FU
#define MULTIPLIER_WORD_60 0x744D79A9U
#define MULTIPLIER_WORD_61 0xCEF43461U
#define MULTIPLIER_WORD_62 0x6FE966F3U
#define MULTIPLIER_WORD_63 0x43E70197U

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


// Instruction set simulator.
// mov.u32
#define ins_mov(dest, src) dest = src;
// add.u32
#define ins_add(dest, left, right) dest = left + right;
// add.cc.u32
#define ins_add_cc(dest, left, right) carryRegister = AddFullHelper(dest, left, right, 0);
// adc.cc.u32
#define ins_addc_cc(dest, left, right) carryRegister = AddFullHelper(dest, left, right, carryRegister);
// adc.u32
#define ins_addc(dest, left, right) AddFullHelper(dest, left, right, carryRegister);
// sub.u32
#define ins_sub(dest, left, right) dest = left - right;
// sub.cc.u32
#define ins_sub_cc(dest, left, right) carryRegister = SubFullHelper(dest, left, right, 0);
// subc.cc.u32
#define ins_subc_cc(dest, left, right) carryRegister = SubFullHelper(dest, left, right, carryRegister);
// subc.u32
#define ins_subc(dest, left, right) SubFullHelper(dest, left, right, carryRegister);
// mul.lo.u32
#define ins_mul_lo(dest, left, right) dest = left * right;
// mul.hi.u32
#define ins_mul_hi(dest, left, right) dest = MultiplyHighHelper(left, right);
// mad.lo.u32
#define ins_mad_lo(dest, left, right, add) ins_add(dest, left * right, add);
// mad.lo.cc.u32
#define ins_mad_lo_cc(dest, left, right, add) ins_add_cc(dest, left * right, add);
// madc.lo.cc.u32
#define ins_madc_lo_cc(dest, left, right, add) ins_addc_cc(dest, left * right, add);
// madc.lo.u32
#define ins_madc_lo(dest, left, right, add) ins_addc(dest, left * right, add);
// mad.hi.u32
#define ins_mad_hi(dest, left, right, add) ins_add(dest, MultiplyHighHelper(left, right), add);
// mad.hi.cc.u32
#define ins_mad_hi_cc(dest, left, right, add) ins_add_cc(dest, MultiplyHighHelper(left, right), add);
// madc.hi.cc.u32
#define ins_madc_hi_cc(dest, left, right, add) ins_addc_cc(dest, MultiplyHighHelper(left, right), add);
// madc.hi.u32
#define ins_madc_hi(dest, left, right, add) ins_addc(dest, MultiplyHighHelper(left, right), add);


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


// From Kaiyong Zhao's Algorithm 1, with ideas from Algorithm 3.
template <unsigned LimbCount>
void MontgomeryReduce(Limb output[LimbCount + 2], const Limb modulus[LimbCount], Limb modulusInverseR0)
{
	for (unsigned i = 0; i < LimbCount; ++i)
	{
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


void MultiplyStuffKernel(Limb *dest, const Limb *src, const Limb *modulus, Limb modulusInverseR0)
{
	#define CONSTANT_ZERO 0

	#define DECLARE_OUTPUT_ROUND(num) Limb t##num;

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

	Limb mr0 = modulusInverseR0;

	Limb multiplicand;
	Limb m;
	Limb carry;
	Limb temp;

	Limb carryRegister;

	Limb &var_multiplicand = multiplicand;
	Limb &var_temp = temp;
	Limb &var_t64 = t64;

	for (unsigned i = 0; i < LIMB_COUNT; ++i)
	{
		// Multiplier for this round.
		var_multiplicand = src[i];  // src[(i * LIMB_COUNT) + threadIdx.x];

		// First round is special, because registers not set yet.
		if (i == 0)
		{
			ins_mul_lo(t00, multiplicand, MULTIPLIER_WORD_00);
			ins_mul_hi(carry, multiplicand, MULTIPLIER_WORD_00);

			#define MULTIPLY_FIRST_ROUND(num) \
				ins_mad_lo_cc(t##num, multiplicand, MULTIPLIER_WORD_##num, carry); \
				ins_madc_hi(carry, multiplicand, MULTIPLIER_WORD_##num, CONSTANT_ZERO)

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

			ins_mov(t64, carry);
			ins_mov(t65, CONSTANT_ZERO);
		}
		else
		{
			ins_mad_lo_cc(t00, multiplicand, MULTIPLIER_WORD_00, t00);
			ins_madc_hi(carry, multiplicand, MULTIPLIER_WORD_00, CONSTANT_ZERO);

			#define MULTIPLY_ROUND(num) \
				ins_mad_lo_cc(t##num, multiplicand, MULTIPLIER_WORD_##num, t##num); \
				ins_madc_hi(temp, multiplicand, MULTIPLIER_WORD_##num, CONSTANT_ZERO); \
				ins_add_cc(t##num, t##num, carry); \
				ins_addc(carry, temp, CONSTANT_ZERO);

			#define MULTIPLY_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
				MULTIPLY_ROUND(r1); MULTIPLY_ROUND(r2); MULTIPLY_ROUND(r3); \
				MULTIPLY_ROUND(r4); MULTIPLY_ROUND(r5); MULTIPLY_ROUND(r6); \
				MULTIPLY_ROUND(r7); MULTIPLY_ROUND(r8); MULTIPLY_ROUND(r9);

			MULTIPLY_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
			MULTIPLY_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
			MULTIPLY_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
			MULTIPLY_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
			MULTIPLY_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
			MULTIPLY_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
			MULTIPLY_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);

			ins_add_cc(t64, t64, carry);
			ins_addc(t65, t65, CONSTANT_ZERO);
		}

		// Reduce rounds.
		ins_mul_lo(m, t00, MODULUS_INVERSE_LOW);

		// First round of reduce discards the result.
		ins_mad_lo_cc(carry, m, MODULUS_WORD_00, t00);
		ins_madc_hi(carry, m, MODULUS_WORD_00, CONSTANT_ZERO);

		#define REDUCE_ROUND(prev, curr) \
			ins_mad_lo_cc(t##prev, m, MODULUS_WORD_##curr, carry); \
			ins_madc_hi(carry, m, MODULUS_WORD_##curr, CONSTANT_ZERO); \
			ins_add_cc(t##prev, t##prev, t##curr); \
			ins_addc(carry, carry, CONSTANT_ZERO);

		#define REDUCE_ROUND_9(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9) \
			REDUCE_ROUND(r0, r1); REDUCE_ROUND(r1, r2); REDUCE_ROUND(r2, r3); \
			REDUCE_ROUND(r3, r4); REDUCE_ROUND(r4, r5); REDUCE_ROUND(r5, r6); \
			REDUCE_ROUND(r6, r7); REDUCE_ROUND(r7, r8); REDUCE_ROUND(r8, r9);

		REDUCE_ROUND_9(00, 01, 02, 03, 04, 05, 06, 07, 08, 09);
		REDUCE_ROUND_9(09, 10, 11, 12, 13, 14, 15, 16, 17, 18);
		REDUCE_ROUND_9(18, 19, 20, 21, 22, 23, 24, 25, 26, 27);
		REDUCE_ROUND_9(27, 28, 29, 30, 31, 32, 33, 34, 35, 36);
		REDUCE_ROUND_9(36, 37, 38, 39, 40, 41, 42, 43, 44, 45);
		REDUCE_ROUND_9(45, 46, 47, 48, 49, 50, 51, 52, 53, 54);
		REDUCE_ROUND_9(54, 55, 56, 57, 58, 59, 60, 61, 62, 63);

		ins_add_cc(t63, t64, carry);
		ins_addc(t64, t65, CONSTANT_ZERO);
	}

	// Compare against the modulus.
	ins_sub_cc(temp, t00, MODULUS_WORD_00);

	#define COMPARE_ROUND(num) \
		ins_subc_cc(temp, t##num, MODULUS_WORD_##num);

	#define COMPARE_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
		COMPARE_ROUND(r1); COMPARE_ROUND(r2); COMPARE_ROUND(r3); \
		COMPARE_ROUND(r4); COMPARE_ROUND(r5); COMPARE_ROUND(r6); \
		COMPARE_ROUND(r7); COMPARE_ROUND(r8); COMPARE_ROUND(r9);

	COMPARE_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
	COMPARE_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
	COMPARE_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
	COMPARE_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
	COMPARE_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
	COMPARE_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
	COMPARE_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);

	ins_subc(temp, CONSTANT_ZERO, CONSTANT_ZERO);

	// If necessary, subtract.
	if ((var_temp == 0) || (var_t64 != 0))
	{
		ins_sub_cc(t00, t00, MODULUS_WORD_00);

	#define SUBTRACT_ROUND(num) \
		ins_subc_cc(t##num, t##num, MODULUS_WORD_##num);

	#define SUBTRACT_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
		SUBTRACT_ROUND(r1); SUBTRACT_ROUND(r2); SUBTRACT_ROUND(r3); \
		SUBTRACT_ROUND(r4); SUBTRACT_ROUND(r5); SUBTRACT_ROUND(r6); \
		SUBTRACT_ROUND(r7); SUBTRACT_ROUND(r8); SUBTRACT_ROUND(r9);

		SUBTRACT_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
		SUBTRACT_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
		SUBTRACT_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
		SUBTRACT_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
		SUBTRACT_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
		SUBTRACT_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
		SUBTRACT_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);
	}

	#define STORE_ROUND(num) dest[1##num - 100] = t##num;

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

	__nop();
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

	MPZNumber gmpMultiplier;
	mpz_set_str(gmpMultiplier, s_multiplier, 16);

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
		//ReadRandom(b.m_limbs, sizeof(b.m_limbs));
		//FromLimbArray(gmpB, b.m_gmp);
		//mpz_mod(gmpB, gmpB, gmpModulus);
		mpz_set(gmpB, gmpMultiplier);
		ToLimbArray(b.m_gmp, gmpB);

		MPZNumber gmpAMontgomery;
		Number<LIMB_COUNT> aMontgomery;
		mpz_mul(gmpAMontgomery, gmpA, gmpR);
		mpz_mod(gmpAMontgomery, gmpAMontgomery, gmpModulus);
		ToLimbArray(aMontgomery.m_gmp, gmpAMontgomery);

		MPZNumber gmpBMontgomery;
		Number<LIMB_COUNT> bMontgomery;
		mpz_mul(gmpBMontgomery, gmpB, gmpR);
		mpz_mod(gmpBMontgomery, gmpBMontgomery, gmpModulus);
		ToLimbArray(bMontgomery.m_gmp, gmpBMontgomery);

		MPZNumber gmpCheck;
		Number<LIMB_COUNT> check;
		mpz_mul(gmpCheck, gmpAMontgomery, gmpBMontgomery);
		mpz_mod(gmpCheck, gmpCheck, gmpModulus);
		mpz_mul(gmpCheck, gmpCheck, gmpRInverse);
		mpz_mod(gmpCheck, gmpCheck, gmpModulus);
		ToLimbArray(check.m_gmp, gmpCheck);

		Limb output[LIMB_COUNT + 2];
		std::memset(output, 0, sizeof(output));
		MontgomeryModularMultiply<LIMB_COUNT, false>(output, aMontgomery.m_limbs, bMontgomery.m_limbs, modulus.m_limbs, modulusInverseR0);
		std::memset(output, 0xCC, sizeof(output));
		MontgomeryModularMultiply<LIMB_COUNT, true>(output, aMontgomery.m_limbs, bMontgomery.m_limbs, modulus.m_limbs, modulusInverseR0);

		if (std::memcmp(output, check.m_limbs, sizeof(check.m_limbs)) != 0)
		{
			__debugbreak();
		}

		Limb output2[LIMB_COUNT + 2];
		MultiplyStuffKernel(output2, aMontgomery.m_limbs, modulus.m_limbs, modulusInverseR0);

		if (std::memcmp(output2, check.m_limbs, sizeof(check.m_limbs)) != 0)
		{
			__debugbreak();
		}

		MPZNumber gmpReduced;
		Number<LIMB_COUNT> reduced;
		mpz_mul(gmpReduced, gmpA, gmpB);
		mpz_mod(gmpReduced, gmpReduced, gmpModulus);
		ToLimbArray(reduced.m_gmp, gmpReduced);

		MPZNumber gmpReduced2;
		Number<LIMB_COUNT> reduced2;
		mpz_mul(gmpReduced2, gmpCheck, gmpRInverse);
		mpz_mod(gmpReduced2, gmpReduced2, gmpModulus);
		ToLimbArray(reduced2.m_gmp, gmpReduced2);

		if (std::memcmp(reduced.m_limbs, reduced2.m_limbs, sizeof(reduced2.m_limbs)) != 0)
		{
			__debugbreak();
		}

		Limb reduced3[LIMB_COUNT + 2];
		std::memcpy(reduced3, output, sizeof(reduced3));
		reduced3[LIMB_COUNT] = 0;
		reduced3[LIMB_COUNT + 1] = 0;
		MontgomeryReduce<LIMB_COUNT>(reduced3, modulus.m_limbs, modulusInverseR0);

		if (std::memcmp(reduced.m_limbs, reduced3, sizeof(reduced.m_limbs)) != 0)
		{
			__debugbreak();
		}

		__nop();
	}

	return 0;
}
