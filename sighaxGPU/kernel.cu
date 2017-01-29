#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "Shared.h"


// The 32-bit parts of s_modulus, least-significant first.
#define MODULUS_WORD_00 "0x9BD0EC9BU"
#define MODULUS_WORD_01 "0x045BE017U"
#define MODULUS_WORD_02 "0x3085E8A1U"
#define MODULUS_WORD_03 "0xB2CC3687U"
#define MODULUS_WORD_04 "0x4A7258FDU"
#define MODULUS_WORD_05 "0x6BC95A59U"
#define MODULUS_WORD_06 "0xBD62C246U"
#define MODULUS_WORD_07 "0xA0165CE4U"
#define MODULUS_WORD_08 "0x93AF4D38U"
#define MODULUS_WORD_09 "0x64B2B6C0U"
#define MODULUS_WORD_10 "0x662AE856U"
#define MODULUS_WORD_11 "0x8E4A169DU"
#define MODULUS_WORD_12 "0x9900ABA4U"
#define MODULUS_WORD_13 "0xFDBAD8A9U"
#define MODULUS_WORD_14 "0x9F98C215U"
#define MODULUS_WORD_15 "0xDEDA5C0BU"
#define MODULUS_WORD_16 "0x51B71CE0U"
#define MODULUS_WORD_17 "0xAFF25E06U"
#define MODULUS_WORD_18 "0x3AA949A5U"
#define MODULUS_WORD_19 "0x075D2634U"
#define MODULUS_WORD_20 "0x0154779FU"
#define MODULUS_WORD_21 "0x69616289U"
#define MODULUS_WORD_22 "0x5500377CU"
#define MODULUS_WORD_23 "0xB9ACA6EEU"
#define MODULUS_WORD_24 "0xBB01CB13U"
#define MODULUS_WORD_25 "0xF1ED2B18U"
#define MODULUS_WORD_26 "0x2CA6D7ACU"
#define MODULUS_WORD_27 "0x2D48967CU"
#define MODULUS_WORD_28 "0x9A2E5303U"
#define MODULUS_WORD_29 "0x0D1AF902U"
#define MODULUS_WORD_30 "0xA11B5E9DU"
#define MODULUS_WORD_31 "0xB2C0447EU"
#define MODULUS_WORD_32 "0x11D17EABU"
#define MODULUS_WORD_33 "0x4984E094U"
#define MODULUS_WORD_34 "0x9A29295EU"
#define MODULUS_WORD_35 "0x1AAFB348U"
#define MODULUS_WORD_36 "0x040F02A8U"
#define MODULUS_WORD_37 "0xA47EDD19U"
#define MODULUS_WORD_38 "0x991D31B8U"
#define MODULUS_WORD_39 "0x32C4DCA9U"
#define MODULUS_WORD_40 "0x71ECC8C1U"
#define MODULUS_WORD_41 "0x41EEE4F0U"
#define MODULUS_WORD_42 "0x0BC67343U"
#define MODULUS_WORD_43 "0x526370F0U"
#define MODULUS_WORD_44 "0x75887793U"
#define MODULUS_WORD_45 "0xF0170CB5U"
#define MODULUS_WORD_46 "0x604BC7F7U"
#define MODULUS_WORD_47 "0x62B730F4U"
#define MODULUS_WORD_48 "0x8182B681U"
#define MODULUS_WORD_49 "0xE64DB123U"
#define MODULUS_WORD_50 "0x6B458221U"
#define MODULUS_WORD_51 "0x253CBE37U"
#define MODULUS_WORD_52 "0x0FDE451BU"
#define MODULUS_WORD_53 "0xC02D4D41U"
#define MODULUS_WORD_54 "0x2ED88F92U"
#define MODULUS_WORD_55 "0x962EC4E9U"
#define MODULUS_WORD_56 "0x336FB076U"
#define MODULUS_WORD_57 "0xA2A08CBBU"
#define MODULUS_WORD_58 "0x72707932U"
#define MODULUS_WORD_59 "0xA16B9AABU"
#define MODULUS_WORD_60 "0x8817B003U"
#define MODULUS_WORD_61 "0xFDAC90E8U"
#define MODULUS_WORD_62 "0x3D33E955U"
#define MODULUS_WORD_63 "0xDECFB6FCU"

// The low 32 bits of R - (modulus^-1) mod R.
#define MODULUS_INVERSE_LOW "0x85E8E66DU"

// The 32-bit parts of MULTIPLIER_MONTGOMERY, least-significant first.
#define MULTIPLIER_WORD_00 "0xB5F30EF7U"
#define MULTIPLIER_WORD_01 "0x1020BA70U"
#define MULTIPLIER_WORD_02 "0xFA2C6D29U"
#define MULTIPLIER_WORD_03 "0x22B71AECU"
#define MULTIPLIER_WORD_04 "0xEFBED8EEU"
#define MULTIPLIER_WORD_05 "0x71E2EEC1U"
#define MULTIPLIER_WORD_06 "0x17296E7CU"
#define MULTIPLIER_WORD_07 "0x6824C5EAU"
#define MULTIPLIER_WORD_08 "0x468A8D87U"
#define MULTIPLIER_WORD_09 "0x29EBC942U"
#define MULTIPLIER_WORD_10 "0xB48A741EU"
#define MULTIPLIER_WORD_11 "0xC2E095FBU"
#define MULTIPLIER_WORD_12 "0xFB66B5CAU"
#define MULTIPLIER_WORD_13 "0xE9F4FFC9U"
#define MULTIPLIER_WORD_14 "0xF9A20DB7U"
#define MULTIPLIER_WORD_15 "0x995AB257U"
#define MULTIPLIER_WORD_16 "0x395BF3CCU"
#define MULTIPLIER_WORD_17 "0x385EBC48U"
#define MULTIPLIER_WORD_18 "0x2DB55CBFU"
#define MULTIPLIER_WORD_19 "0xD3B4BF5AU"
#define MULTIPLIER_WORD_20 "0xEC8F0FFBU"
#define MULTIPLIER_WORD_21 "0x268291ADU"
#define MULTIPLIER_WORD_22 "0xAFD05422U"
#define MULTIPLIER_WORD_23 "0xCFE89D5FU"
#define MULTIPLIER_WORD_24 "0xF5D73ABBU"
#define MULTIPLIER_WORD_25 "0xD3BDEF92U"
#define MULTIPLIER_WORD_26 "0x95A383E1U"
#define MULTIPLIER_WORD_27 "0x0C70076CU"
#define MULTIPLIER_WORD_28 "0x7EA0B490U"
#define MULTIPLIER_WORD_29 "0xBD967333U"
#define MULTIPLIER_WORD_30 "0x6B37B4DAU"
#define MULTIPLIER_WORD_31 "0x6A402853U"
#define MULTIPLIER_WORD_32 "0x7752F30DU"
#define MULTIPLIER_WORD_33 "0xBE8D824CU"
#define MULTIPLIER_WORD_34 "0x68EE4D1EU"
#define MULTIPLIER_WORD_35 "0x9A18D6F8U"
#define MULTIPLIER_WORD_36 "0x9583AA49U"
#define MULTIPLIER_WORD_37 "0xAB129DBFU"
#define MULTIPLIER_WORD_38 "0x2FD7E863U"
#define MULTIPLIER_WORD_39 "0x08D8B04FU"
#define MULTIPLIER_WORD_40 "0x474FE9E0U"
#define MULTIPLIER_WORD_41 "0xDEFB34FCU"
#define MULTIPLIER_WORD_42 "0xE87C6774U"
#define MULTIPLIER_WORD_43 "0x43C85CA9U"
#define MULTIPLIER_WORD_44 "0xC1D24733U"
#define MULTIPLIER_WORD_45 "0xF0298AB0U"
#define MULTIPLIER_WORD_46 "0x696B969AU"
#define MULTIPLIER_WORD_47 "0x2EDA7839U"
#define MULTIPLIER_WORD_48 "0x02ADD7F1U"
#define MULTIPLIER_WORD_49 "0xBAB19C9CU"
#define MULTIPLIER_WORD_50 "0xAC1B4D12U"
#define MULTIPLIER_WORD_51 "0x38DCAE67U"
#define MULTIPLIER_WORD_52 "0x5C6D352AU"
#define MULTIPLIER_WORD_53 "0x05B0ADB2U"
#define MULTIPLIER_WORD_54 "0x48E344A7U"
#define MULTIPLIER_WORD_55 "0xFC175DC2U"
#define MULTIPLIER_WORD_56 "0xAC1B1DD3U"
#define MULTIPLIER_WORD_57 "0x3486B994U"
#define MULTIPLIER_WORD_58 "0x86448BDBU"
#define MULTIPLIER_WORD_59 "0xCEB1823FU"
#define MULTIPLIER_WORD_60 "0x744D79A9U"
#define MULTIPLIER_WORD_61 "0xCEF43461U"
#define MULTIPLIER_WORD_62 "0x6FE966F3U"
#define MULTIPLIER_WORD_63 "0x43E70197U"


// The GPU code.
__global__ void MultiplyStuffKernel(Limb *dest, const Limb *src)
{
	// Start of code.
	#define DECLARE_OUTPUT_ROUND(num) Limb var_t##num = 0

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

	Limb var_multiplicand = 0;
	Limb var_m = 0;
	Limb var_carry = 0;
	Limb var_temp = 0;

	// Main multiplication-reduction loop.
	for (unsigned i = 0; i < LIMB_COUNT; ++i)
	{
		// Multiplier for this round.
		var_multiplicand = src[(i * LIMB_COUNT) + threadIdx.x];

		// First round is special, because registers not set yet.
		if (i == 0)
		{
			asm(
				"mul.lo.u32 %0, %1, " MULTIPLIER_WORD_00 ";\n\t"
				"mul.hi.u32 %2, %1, " MULTIPLIER_WORD_00 ";\n\t"
				:	"+r"(var_t00), "+r"(var_multiplicand), "+r"(var_carry));

			#define MULTIPLY_FIRST_ROUND(num) \
				asm( \
					"mad.lo.cc.u32 %0, %1, " MULTIPLIER_WORD_##num ", %2;\n\t" \
					"madc.hi.u32 %2, %1, " MULTIPLIER_WORD_##num ", 0;\n\t" \
					:	"+r"(var_t##num), "+r"(var_multiplicand), "+r"(var_carry))

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

			var_t64 = var_carry;
			var_t65 = 0;
		}
		else
		{
			asm(
				"mad.lo.cc.u32 %0, %1, " MULTIPLIER_WORD_00 ", %0;\n\t"
				"madc.hi.u32 %2, %1, " MULTIPLIER_WORD_00 ", 0;\n\t"  // note no % on this 0
				:	"+r"(var_t00), "+r"(var_multiplicand), "+r"(var_carry));

			#define MULTIPLY_ROUND(num) \
				asm( \
					"mad.lo.cc.u32 %0, %1, " MULTIPLIER_WORD_##num ", %0;\n\t" \
					"madc.hi.u32 %2, %1, " MULTIPLIER_WORD_##num ", 0;\n\t" /* note no % on 0 */ \
					"add.cc.u32 %0, %0, %3;\n\t" \
					"addc.u32 %3, %2, 0;\n\t" /* note no % on 0 */ \
					:	"+r"(var_t##num), "+r"(var_multiplicand), "+r"(var_temp), "+r"(var_carry))

			#define MULTIPLY_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
				MULTIPLY_ROUND(r1); MULTIPLY_ROUND(r2); MULTIPLY_ROUND(r3); \
				MULTIPLY_ROUND(r4); MULTIPLY_ROUND(r5); MULTIPLY_ROUND(r6); \
				MULTIPLY_ROUND(r7); MULTIPLY_ROUND(r8); MULTIPLY_ROUND(r9)

			MULTIPLY_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
			MULTIPLY_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
			MULTIPLY_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
			MULTIPLY_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
			MULTIPLY_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
			MULTIPLY_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
			MULTIPLY_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);

			asm(
				"add.cc.u32 %0, %0, %1;\n\t"
				"addc.u32 %2, %2, 0;\n\t" // note no % on 0
				:	"+r"(var_t64), "+r"(var_carry), "+r"(var_t65));
		}

		// Reduce rounds.
		asm(
			// Get multiplicand used for reduction.
			"mul.lo.u32 %0, %1, " MODULUS_INVERSE_LOW ";\n\t"
			// The first round of reduction discards the result.
			"mad.lo.cc.u32 %2, %0, " MODULUS_WORD_00 ", %1;\n\t"
			"madc.hi.u32 %2, %0, " MODULUS_WORD_00 ", 0;\n\t" // note no % on 0
			:	"+r"(var_m), "+r"(var_t00), "+r"(var_carry));

		#define REDUCE_ROUND(prev, curr) \
			asm( \
				"mad.lo.cc.u32 %0, %2, " MODULUS_WORD_##curr ", %3;\n\t" \
				"madc.hi.u32 %3, %2, " MODULUS_WORD_##curr ", 0;\n\t" /* note no % on 0 */ \
				"add.cc.u32 %0, %0, %1;\n\t" \
				"addc.u32 %3, %3, 0;\n\t" /* note no % on 0*/ \
				:	"+r"(var_t##prev), "+r"(var_t##curr), "+r"(var_m), "+r"(var_carry))

		#define REDUCE_ROUND_9(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9) \
			REDUCE_ROUND(r0, r1); REDUCE_ROUND(r1, r2); REDUCE_ROUND(r2, r3); \
			REDUCE_ROUND(r3, r4); REDUCE_ROUND(r4, r5); REDUCE_ROUND(r5, r6); \
			REDUCE_ROUND(r6, r7); REDUCE_ROUND(r7, r8); REDUCE_ROUND(r8, r9)

		REDUCE_ROUND_9(00, 01, 02, 03, 04, 05, 06, 07, 08, 09);
		REDUCE_ROUND_9(09, 10, 11, 12, 13, 14, 15, 16, 17, 18);
		REDUCE_ROUND_9(18, 19, 20, 21, 22, 23, 24, 25, 26, 27);
		REDUCE_ROUND_9(27, 28, 29, 30, 31, 32, 33, 34, 35, 36);
		REDUCE_ROUND_9(36, 37, 38, 39, 40, 41, 42, 43, 44, 45);
		REDUCE_ROUND_9(45, 46, 47, 48, 49, 50, 51, 52, 53, 54);
		REDUCE_ROUND_9(54, 55, 56, 57, 58, 59, 60, 61, 62, 63);

		asm(
			"add.cc.u32 %0, %1, %3;\n\t"
			"addc.u32 %1, %2, 0;\n\t" // note no % on 0
			:	"+r"(var_t63), "+r"(var_t64), "+r"(var_t65), "+r"(var_carry));
	}

	// Compare against the modulus.
	asm(
		"sub.cc.u32 %0, %1, " MODULUS_WORD_00 ";\n\t"
		"addc.u32 %2, 0, 0;\n\t"  // extract carry flag to carry variable
		:	"+r"(var_temp), "+r"(var_t00), "+r"(var_carry));

	#define COMPARE_ROUND(regnum, wordnum) \
		"subc.cc.u32 %0, %" #regnum ", " MODULUS_WORD_##wordnum ";\n\t"

	#define COMPARE_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
		asm( \
			"add.cc.u32 %0, %10, 0xFFFFFFFF;\n\t" /* put carry variable into carry flag */ \
			COMPARE_ROUND(1, r1) COMPARE_ROUND(2, r2) COMPARE_ROUND(3, r3) \
			COMPARE_ROUND(4, r4) COMPARE_ROUND(5, r5) COMPARE_ROUND(6, r6) \
			COMPARE_ROUND(7, r7) COMPARE_ROUND(8, r8) COMPARE_ROUND(9, r9) \
			"addc.u32 %10, 0, 0;\n\t"  /* extract carry flag to carry variable */ \
			:	"+r"(var_temp), \
				"+r"(var_t##r1), "+r"(var_t##r2), "+r"(var_t##r3), \
				"+r"(var_t##r4), "+r"(var_t##r5), "+r"(var_t##r6), \
				"+r"(var_t##r7), "+r"(var_t##r8), "+r"(var_t##r9), \
				"+r"(var_carry))

	COMPARE_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
	COMPARE_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
	COMPARE_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
	COMPARE_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
	COMPARE_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
	COMPARE_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
	COMPARE_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);

	// If necessary, subtract.
	if ((var_carry == 0) || (var_t64 != 0))
	{
		asm(
			"sub.cc.u32 %1, %1, " MODULUS_WORD_00 ";\n\t"
			"addc.u32 %2, 0, 0;\n\t"  // extract carry flag to carry variable
			:	"+r"(var_temp), "+r"(var_t00), "+r"(var_carry));

		#define SUBTRACT_ROUND(regnum, wordnum) \
			"subc.cc.u32 %" #regnum ", %" #regnum ", " MODULUS_WORD_##wordnum ";\n\t"

		#define SUBTRACT_ROUND_9(r1, r2, r3, r4, r5, r6, r7, r8, r9) \
			asm( \
				"add.cc.u32 %0, %10, 0xFFFFFFFF;\n\t" /* put carry variable into carry flag */ \
				SUBTRACT_ROUND(1, r1) SUBTRACT_ROUND(2, r2) SUBTRACT_ROUND(3, r3) \
				SUBTRACT_ROUND(4, r4) SUBTRACT_ROUND(5, r5) SUBTRACT_ROUND(6, r6) \
				SUBTRACT_ROUND(7, r7) SUBTRACT_ROUND(8, r8) SUBTRACT_ROUND(9, r9) \
				"addc.u32 %10, 0, 0;\n\t"  /* extract carry flag to carry variable */ \
				:	"+r"(var_temp), \
					"+r"(var_t##r1), "+r"(var_t##r2), "+r"(var_t##r3), \
					"+r"(var_t##r4), "+r"(var_t##r5), "+r"(var_t##r6), \
					"+r"(var_t##r7), "+r"(var_t##r8), "+r"(var_t##r9), \
					"+r"(var_carry))

		SUBTRACT_ROUND_9(01, 02, 03, 04, 05, 06, 07, 08, 09);
		SUBTRACT_ROUND_9(10, 11, 12, 13, 14, 15, 16, 17, 18);
		SUBTRACT_ROUND_9(19, 20, 21, 22, 23, 24, 25, 26, 27);
		SUBTRACT_ROUND_9(28, 29, 30, 31, 32, 33, 34, 35, 36);
		SUBTRACT_ROUND_9(37, 38, 39, 40, 41, 42, 43, 44, 45);
		SUBTRACT_ROUND_9(46, 47, 48, 49, 50, 51, 52, 53, 54);
		SUBTRACT_ROUND_9(55, 56, 57, 58, 59, 60, 61, 62, 63);
	}

	#define STORE_ROUND(num) dest[((1##num - 100) * LIMB_COUNT * 2) + threadIdx.x] = var_t##num;

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

#undef DECLARE_OUTPUT_ROUND
#undef DECLARE_OUTPUT_ROUND_9
#undef MULTIPLY_FIRST_ROUND
#undef MULTIPLY_FIRST_ROUND_9
#undef MULTIPLY_ROUND
#undef MULTIPLY_ROUND_9
#undef REDUCE_ROUND
#undef REDUCE_ROUND_9
#undef COMPARE_ROUND
#undef COMPARE_ROUND_9
#undef SUBTRACT_ROUND
#undef SUBTRACT_ROUND_9
#undef STORE_ROUND
#undef STORE_ROUND_9
}


// Execute the math operation.
cudaError_t GPUExecuteOperation(Limb (&dest)[NUM_THREADS * LIMB_COUNT * 2], const Limb (&src)[NUM_THREADS * LIMB_COUNT])
{
	cudaError_t cudaStatus;

	// Allocate memory for data transfers.
	void *d_src;
	cudaStatus = cudaMalloc(&d_src, sizeof(src));
	if (cudaStatus != cudaSuccess)
	{
		return cudaStatus;
	}

	void *d_dest;
	cudaStatus = cudaMalloc(&d_dest, sizeof(dest));
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_src);
		return cudaStatus;
	}

	// Copy parameters over.
	cudaStatus = cudaMemcpy(d_src, &src, sizeof(src), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_src);
		cudaFree(d_dest);
		return cudaStatus;
	}

	// Execute operation.
	MultiplyStuffKernel<<<1, NUM_THREADS>>>(static_cast<Limb *>(d_dest), static_cast<Limb *>(d_src));

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_src);
		cudaFree(d_dest);
		return cudaStatus;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		cudaFree(d_src);
		cudaFree(d_dest);
		return cudaStatus;
	}

	// Copy result to host.
	cudaStatus = cudaMemcpy(&dest, d_dest, sizeof(dest), cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_dest);

	return cudaSuccess;
}
