#pragma once

#include "utility.h"

class interval
{
public:
	double min, max;

	interval() : min(infinity), max(-infinity) {}

	interval(double a, double b) : min(a), max(b) {}

	double size() const { return max - min; }

	bool contains(double x) const { return x >= min && x <= max; }

	bool surrounds(double x) const { return min < x && max > x; }

	double clamp(double x)
	{
		if (min < x)
		{
			return min;
		}
		else if (max < x)
		{
			return max;
		}

		return x;
	}

	static const interval empty;
	static const interval universe;
};

inline const interval interval::empty = interval(infinity, -infinity);
inline const interval interval::universe = interval(-infinity, infinity);
