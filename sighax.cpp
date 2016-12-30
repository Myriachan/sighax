#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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


void BruteForce(mpz_t modulus)
{
	uint64_t start;
	static_assert(sizeof(start) <= sizeof(unsigned long), "bad size for GMP");

	start = static_cast<decltype(0u + uint64_t())>(1) << 56;

	mpz_t current;
	mpz_init_set_ui(current, static_cast<unsigned long>(start));

	mpz_t result;
	mpz_init2(result, (KEY_SIZE * CHAR_BIT) + (sizeof(mp_limb_t) * CHAR_BIT));

	for (unsigned x = 0; x < 1000000; ++x)
	{
		if (x % 100000 == 0) std::printf("%8u\n", x);

		mpz_powm_ui(result, current, s_publicExponent, modulus);

		uint8_t bytes[KEY_SIZE];
		ToArray(bytes, result);

		mpz_add_ui(current, current, 1);
	}

	mpz_clears(current, result, nullptr);
}


int main()
{
	mpz_t modulus;
	mpz_init_set_str(modulus, s_modulusText, 16);

	mpz_t sample;
	mpz_init_set_str(sample, s_sample, 16);

	mpz_powm_ui(sample, sample, s_publicExponent, modulus);

	DumpNumber(sample);

	BruteForce(modulus);

	mpz_clears(modulus, sample, nullptr);
	return 0;
}
