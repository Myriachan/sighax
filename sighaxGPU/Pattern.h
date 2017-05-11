#pragma once

// Determine whether the given buffer is what we want.
template <typename GetByteLambda>
__device__ __host__ bool IsWhatWeWant(GetByteLambda &getByte)
{
	// Test code - used when profiling so that we never find anything.
#ifdef PROFILE_MODE
	static volatile bool s_meow = false;
	if (!s_meow) return false;
#endif

	// A match must begin with 00 02.
	if ((getByte(0x00) != 0x00) || (getByte(0x01) != 0x02))
	{
		return false;
	}

	// Count how many nonzero bytes are after the 0x02.
	unsigned zeroIndex;
	for (zeroIndex = 0x02; zeroIndex < KEY_SIZE; ++zeroIndex)
	{
		if (getByte(zeroIndex) == 0x00)
		{
			break;
		}
	}

	if (zeroIndex >= KEY_SIZE)
	{
		return false;
	}

	// TODO: Rest of implementation.

	return true;
}
