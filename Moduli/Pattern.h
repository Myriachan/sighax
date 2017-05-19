#pragma once

template <bool B>
__device__ __host__ bool ConstantValue()
{
	return B;
}

// Determine whether the given buffer is what we want.
template <typename GetByteLambda>
__device__ __host__ bool IsWhatWeWant(const GetByteLambda &getByte)
{
	// Test code - used when profiling so that we never find anything.
#ifdef PROFILE_MODE
	if (!ConstantValue<false>())
	{
		return false;
	}
#endif

	// A match must begin with 00 02.
	if ((getByte(0x00) != 0x00) || (getByte(0x01) != 0x02))
	{
		return false;
	}

	// Find a 00 byte.
	unsigned zeroIndex;
	for (zeroIndex = 2; zeroIndex < KEY_SIZE; ++zeroIndex)
	{
		if (getByte(zeroIndex) == 0x00)
		{
			break;
		}
	}

	if (zeroIndex >= KEY_SIZE - 1 - 1)
	{
		return false;
	}

	// Rest of signature must be no bigger than 0x33 bytes.
	if (zeroIndex < KEY_SIZE - 0x33 - 1)
	{
		return false;
	}

	// Check for the first DER number having a 0x30 type.
	if (getByte(zeroIndex + 1) != 0x30)
	{
		return false;
	}

	// Verify the size.
	unsigned firstNumSize = getByte(zeroIndex + 2);
	if (firstNumSize < 0x80)
	{
		firstNumSize = 1;
	}
	else if (firstNumSize > 0x84)
	{
		return false;
	}
	else
	{
		firstNumSize = 1 + (firstNumSize & 0x7F);
	}

	// Verify that that many bytes remain in the buffer.
	if (zeroIndex + 2 + firstNumSize + 1 + 1 > KEY_SIZE)
	{
		return false;
	}

	// Index of start of next number.
	unsigned secondNumIndex = zeroIndex + 2 + firstNumSize;

	// Verify that the second number sequence starts with 30.
	if (getByte(secondNumIndex) != 0x30)
	{
		return false;
	}

	// Read the second number.
	unsigned finalIndex = secondNumIndex + 1;
	if (finalIndex >= KEY_SIZE)
	{
		return false;
	}

	unsigned secondNumValue;
	unsigned secondNumHeader = getByte(finalIndex);
	++finalIndex;

	if (secondNumHeader < 0x80)
	{
		secondNumValue = secondNumHeader;
	}
	else if (secondNumHeader < 0x85)
	{
		unsigned size = (secondNumHeader & 0x7F);
		secondNumValue = 0;
		for (unsigned x = 0; x < size; ++x)
		{
			if (finalIndex >= KEY_SIZE)
			{
				return false;
			}

			secondNumValue <<= 8;
			secondNumValue += getByte(finalIndex);
			++finalIndex;
		}
	}
	else
	{
		// Boot9 will reject this.
		return false;
	}

	// Check the offset as being 2 from the end.
	unsigned final = finalIndex + secondNumValue;
	if (final != KEY_SIZE - 2)
	{
		return false;
	}

	// The last byte has to be 00-80 or things break.
	if (getByte(KEY_SIZE - 1) >= 0x81)
	{
		return false;
	}

	return true;
}
