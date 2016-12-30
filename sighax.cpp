#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <gmp.h>

using std::size_t;
using std::uint8_t;
using std::uint32_t;
using std::uint64_t;

enum : unsigned { KEY_SIZE = 256 };

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
const char s_sample[] =
	"C26EAACEDD4FF31CD970262B2A6BE06D5CEC1115528CAA6F00BADA3A6B9A886B"
	"5E35DE4FB7E9E4356C4B06B310CCA15AED2B7B433DAB681B0366CC3C769F6D35"
	"79E6B816A8F01BE9C58C1A61A5AB817E2C2FC55C8C70F584D8D485E75584D71A"
	"0EA1A6092751DBE6BCBBE3C119A4CBA5E383E740813129AA4E9CB49DD396BB7F"
	"97F332FAA24F0A4BCBC362E34D4F09F1395B565CC6153D37F057A0496886E66E"
	"965BE08A1030EA038BC45DDF6D4F527F3ED41E2545C0E4772EA6A3F97DD2A0C7"
	"0D340769E8AF211CD1EEB504A96C70B4DE40AD146BF63F509FD56A55358211CC"
	"27A96914769E50864FF4EEA245A5FFA95265D5733EDB0D33D9D1602F5F3CC8E6";


void ReadRandom(void *data, size_t size)
{
	FILE *file = std::fopen("/dev/urandom", "rb");
	if (!file)
	{
		std::abort();
	}

	if (std::fread(data, 1, size, file) != size)
	{
		std::fclose(file);
		std::abort();
	}

	std::fclose(file);
}


void ToArray(uint8_t (&buffer)[KEY_SIZE], mpz_t number)
{
	size_t bits = mpz_sizeinbase(number, 2);
	if (bits > KEY_SIZE * CHAR_BIT)
	{
		std::abort();
	}

	size_t count;
	mpz_export(buffer, &count, -1, sizeof(uint64_t), -1, 0, number);
	std::memset(&buffer[count * sizeof(uint64_t)], 0, KEY_SIZE - (count * sizeof(uint64_t)));
}


void DumpNumber(mpz_t number)
{
	uint8_t buffer[KEY_SIZE];
	ToArray(buffer, number);

	size_t x = KEY_SIZE;
	do
	{
		std::printf("%02X", static_cast<unsigned>(buffer[x - 1]));
	}
	while ((--x) > 0);
	std::puts("");
}


bool IsWhatWeWant(mpz_t number)
{
	// NOTE: ToArray outputs little-endian!
	uint8_t buffer[KEY_SIZE];
	ToArray(buffer, number);

	if ((buffer[0xFF] != 0x00) || (buffer[0xFE] != 0x02))
	{
		return false;
	}

	// Find streak of nonzero values
	unsigned zeroIndex;
	for (zeroIndex = 0xFD; zeroIndex > 4; --zeroIndex)
	{
		if (buffer[zeroIndex] == 0x00)
		{
			break;
		}
	}

	if (zeroIndex <= 4)
	{
		return false;
	}

	if (buffer[zeroIndex - 1] != 0x30)
	{
		return false;
	}

	if (buffer[zeroIndex - 2] < zeroIndex - 2 + 0x20)
	{
		return false;
	}

	return true;
}


void BruteForce(mpz_t modulus)
{
	typedef decltype(0u + uint_fast64_t()) CounterType;
	enum : unsigned { NUM_FACTORS = 64 };
	enum : unsigned { ALLOC_BITS = (KEY_SIZE * CHAR_BIT) + (sizeof(mp_limb_t) * CHAR_BIT) };
//	enum : unsigned { NUM_ITERATIONS = 100000 };
//	enum : unsigned { NUM_ITERATIONS = 1000000 };
	enum : unsigned long long { NUM_ITERATIONS = static_cast<unsigned long long>(-1) >> 1 };

	static_assert(std::numeric_limits<CounterType>::digits >= NUM_FACTORS, "NUM_FACTORS is too small");

	std::printf("Randomly choosing base factors...\n");
	std::fflush(stdout);

	mpz_t base[NUM_FACTORS];
	mpz_t forward[NUM_FACTORS];
	mpz_t backward[NUM_FACTORS];

	for (unsigned x = 0; x < NUM_FACTORS; ++x)
	{
		for (;;)
		{
			uint8_t random[KEY_SIZE];
			ReadRandom(random, sizeof(random));

			mpz_init2(base[x], ALLOC_BITS);
			mpz_import(base[x], KEY_SIZE, 1, 1, 1, 0, random);
			mpz_mod(base[x], base[x], modulus);

			if (mpz_cmp_si(base[x], 1) <= 0)
			{
			my_continue:
				continue;
			}

			for (unsigned y = 0; y < x; ++y)
			{
				if (mpz_cmp(base[y], base[x]) == 0)
				{
					goto my_continue;
				}
			}

			break;
		}

		mpz_init2(forward[x], ALLOC_BITS);
		mpz_powm_ui(forward[x], base[x], s_publicExponent, modulus);

		mpz_init2(backward[x], ALLOC_BITS);
		if (mpz_invert(backward[x], forward[x], modulus) == 0)
		{
			// You're more likely to win the lottery several times in a row
			// than for this to happen.
			std::printf("Public key factored by accident!!!  Give this to Myria!\n");
			DumpNumber(forward[x]);
			std::fflush(stdout);
			std::exit(1);
		}

		// Self-test
		mpz_t test;
		mpz_init2(test, ALLOC_BITS * 2);
		mpz_mul(test, forward[x], backward[x]);
		mpz_mod(test, test, modulus);
		if (mpz_cmp_si(test, 1) != 0)
		{
			std::printf("Self-test error\n");
			DumpNumber(base[x]);
			std::fflush(stdout);
			std::exit(1);
		}
		mpz_clear(test);
	}

	std::printf("Brute-forcing...\n");
	std::fflush(stdout);

	CounterType counter = 0;

	mpz_t current;
	mpz_init2(current, ALLOC_BITS * 2);
	mpz_set_ui(current, 1);

	for (unsigned long long iteration = 0; iteration < NUM_ITERATIONS; ++iteration)
	{
		CounterType grayPrevious = counter ^ (counter >> 1);
		++counter;
		CounterType grayCurrent = counter ^ (counter >> 1);

		CounterType flip = grayCurrent ^ grayPrevious;

		if ((flip == 0) || ((flip & (flip - 1)) != 0))
		{
			std::printf("Gray counter error\n");
			std::fflush(stdout);
			std::exit(1);
		}

		int flipIndex = std::numeric_limits<CounterType>::digits - 1 - __builtin_clzll(flip);

		// Determine whether the bit is being turned on or off.
		bool on = flip & grayCurrent;

//		std::printf("%3llu: %08X %08X %08X %d %d\n", iteration, (unsigned) counter, (unsigned) grayCurrent, (unsigned) flip, flipIndex, on);

		// Calculate next iteration.
		mpz_mul(current, current, (on ? forward : backward)[flipIndex]);
		mpz_mod(current, current, modulus);

		// Is this iteration what we're looking for?
		if (IsWhatWeWant(current))
		{
			mpz_t test, test2, temp;
			mpz_init_set_ui(test, 1);
			mpz_init_set_ui(test2, 1);
			mpz_init2(temp, ALLOC_BITS);
			for (int x = 0; x < std::numeric_limits<CounterType>::digits; ++x)
			{
				CounterType bit = CounterType(1) << x;
				if (grayCurrent & bit)
				{
					mpz_mul(test, test, forward[x]);
					mpz_mod(test, test, modulus);
					mpz_mul(test2, test2, base[x]);
					mpz_mod(test2, test2, modulus);
				}
			}
			if (mpz_cmp(test, current) != 0)
			{
				std::printf("Multiplication build error\n");
				std::fflush(stdout);
				std::exit(1);
			}
			mpz_powm_ui(temp, test2, s_publicExponent, modulus);
			if (mpz_cmp(temp, test) != 0)
			{
				std::printf("Exponential build error\n");
				std::fflush(stdout);
				std::exit(1);
			}

			std::printf("Match found!!\n");
			std::printf("iteration = %llu\n", iteration);
			std::printf("Buffer:\n");
			DumpNumber(temp);
			std::printf("Signature:\n");
			DumpNumber(test2);
			std::fflush(stdout);

			mpz_clears(test, test2, nullptr);
			//break;
		}
	}

	mpz_clear(current);
	for (unsigned x = 0; x < NUM_FACTORS; ++x)
	{
		mpz_clears(base[x], forward[x], backward[x], nullptr);
	}
}


int main()
{
	mpz_t modulus;
	mpz_init_set_str(modulus, s_modulusText, 16);

	mpz_t sample;
	mpz_init_set_str(sample, s_sample, 16);

	mpz_powm_ui(sample, sample, s_publicExponent, modulus);

	//DumpNumber(sample);

	BruteForce(modulus);

	mpz_clears(modulus, sample, nullptr);
	return 0;
}
