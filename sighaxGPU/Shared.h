#pragma once

#include <cinttypes>
#include <cstdint>
#include <climits>
#include <limits>

// The modulus to use.
#include "../Moduli/FIRM-NAND-retail.h"

//#define PROFILE_MODE


typedef std::uint32_t Limb;
typedef std::uint64_t DoubleLimb;

#define LIMB_PRINTF_FORMAT "%08" PRIX32

enum { MODULUS_BITS = 2048 };
enum { KEY_SIZE = MODULUS_BITS / CHAR_BIT };
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


enum
{
	NUM_THREADS = 512,
	NUM_BLOCKS = 4096,

	BLOCK_LIMB_COUNT = NUM_THREADS * LIMB_COUNT,
	TOTAL_LIMB_COUNT = NUM_BLOCKS * BLOCK_LIMB_COUNT,

	NUM_ROUNDS = 100000,
};


class GPUState
{
public:
	GPUState();
	~GPUState();

	cudaError_t Initialize(int device);

	cudaError_t Reseed(unsigned currentSrc, const Limb seed[TOTAL_LIMB_COUNT]);

	cudaError_t Execute(unsigned currentSrc, Limb output[TOTAL_LIMB_COUNT], bool &matchFound);

private:
	void *d_buffers[2];
	void *d_resultFlags;
	unsigned char *h_resultFlags;
};
