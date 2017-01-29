#pragma once

#include <cstdint>
#include <limits>

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


enum { NUM_THREADS = 256 };


cudaError_t GPUExecuteOperation(Limb (&dest)[NUM_THREADS * LIMB_COUNT * 2], const Limb (&src)[NUM_THREADS * LIMB_COUNT]);
