#pragma once

#include <cstdint>
#include <limits>

typedef std::uint32_t Limb;
typedef std::uint64_t DoubleLimb;

enum { TOTAL_BITS = 2048 };
enum { LIMB_BITS = std::numeric_limits<Limb>::digits };
enum { LIMB_COUNT = TOTAL_BITS / LIMB_BITS };

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
enum { NUM_REPEATS = 4 };

struct MathParameters
{
	Limb m_modulus[LIMB_COUNT];
	Limb m_inverse[LIMB_COUNT + 1];
};

struct OneIteration
{
	Limb m_limbs[LIMB_COUNT];
};

struct TransferBlob
{
	OneIteration m_iterations[NUM_THREADS];
};


cudaError_t GPUExecuteOperation(TransferBlob &dest, const TransferBlob &src, const MathParameters &mathParams);
