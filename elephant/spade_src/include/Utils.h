/* 
 *  File: Utils.h
 *  Copyright (c) 2020 Florian Porrmann
 *  
 *  MIT License
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in all
 *  copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *  SOFTWARE.
 *  
 */

#pragma once

#include <string>
#include <vector>
#include <deque>
#include <sstream>
#include <tuple>
#include <memory>
#include <exception>
#include <iostream>
#include <cmath>
#ifndef _WIN32
#include <unistd.h>
#endif

#define CLASS_TAG(_C_) "[" << _C_ << "::" << __func__ << "] "

#define WARNING_TAG "[WARNING]: "

#define DISABLE_COPY_ASSIGN_MOVE(_C_) \
_C_(_C_ const &) = delete; /* disable copy constructor */ \
_C_& operator=(_C_ const &) = delete; /* disable assignment constructor */ \
_C_(_C_ &&) = delete;

#define UNUSED(x) (void)(x)


#define DEFINE_EXCEPTION(__NAME__) \
class __NAME__ : public std::exception \
{ \
public: \
	explicit __NAME__(const std::string& what) : m_what(what) {} \
\
	virtual ~__NAME__() throw() {} \
\
	virtual const char* what() const throw() \
	{ \
		return m_what.c_str(); \
	} \
\
private: \
	std::string m_what; \
};

template <typename T>
void printVector(const std::deque<T>& vec)
{
	for (const T& elem : vec)
		std::cout << elem << " " << std::flush;
	std::cout << std::endl;
}
template <typename T>
void printVector(const std::vector<T>& vec)
{
	for (const T& elem : vec)
		std::cout << elem << " " << std::flush;
	std::cout << std::endl;
}

template<typename T, class InputIt, class InputIt2, class OutputIt, class UnaryPredicate>
OutputIt copy_from_second_if(InputIt first, InputIt last, InputIt2 first2,
							 OutputIt d_first, UnaryPredicate pred)
{
	while (first != last)
	{
		if (pred(*first, *first2))
			*d_first++ = static_cast<T>(*first2);
		first++;
		first2++;
	}
	return d_first;
}

static inline std::vector<std::string> splitString(const std::string& s, const char& delimiter = ' ')
{
	std::vector<std::string> split;
	std::string item;
	std::istringstream stream(s);

	while (std::getline(stream, item, delimiter))
		split.push_back(item);

	return split;
}

//
// From: https://gist.github.com/arvidsson/7231973
//

template <typename T>
class ReverseRange
{
	T& x;

public:
	ReverseRange(T& x) : x(x) {}

	auto begin() const -> decltype(this->x.rbegin())
	{
		return x.rbegin();
	}

	auto end() const -> decltype(this->x.rend())
	{
		return x.rend();
	}
};

template <typename T>
ReverseRange<T> ReverseIterate(T& x)
{
	return ReverseRange<T>(x);
}

//
// From: http://reedbeta.com/blog/python-like-enumerate-in-cpp17/
//
template <typename T,
	typename TIter = decltype(std::begin(std::declval<T>())),
	typename = decltype(std::end(std::declval<T>()))>
	constexpr auto enumerate(T&& iterable)
{
	struct iterator
	{
		size_t i;
		TIter iter;
		bool operator != (const iterator& other) const { return iter != other.iter; }
		iterator& operator ++ () { ++i; ++iter; return *this; }
		auto operator * () const { return std::tie(i, *iter); }
	};
	struct iterable_wrapper
	{
		T iterable;
		auto begin() { return iterator{ 0, std::begin(iterable) }; }
		auto end() { return iterator{ 0, std::end(iterable) }; }
	};
	return iterable_wrapper{ std::forward<T>(iterable) };
}

template <typename T>
constexpr auto enumerate(T&& begin, T&& end)
{
	struct iterator
	{
		size_t i;
		T iter;
		bool operator != (const iterator& other) const { return iter != other.iter; }
		iterator& operator ++ () { ++i; ++iter; return *this; }
		auto operator * () const { return std::tie(i, *iter); }
	};
	struct iterable_wrapper
	{
		T b;
		T e;
		auto begin() { return iterator{ 0, b }; }
		auto end() { return iterator{ 0, e }; }
	};
	return iterable_wrapper{ std::forward<T>(begin), std::forward<T>(end) };
}

//
// From: https://stackoverflow.com/a/26221725
//

template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
	std::size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	if (size == 0) { throw std::runtime_error("Error during formatting."); }
	std::unique_ptr<char[]> buf(new char[size]);
	snprintf(buf.get(), size, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

template <typename A, typename B>
uint32_t partition(std::vector<std::pair<A, B>>& values, const uint32_t& left, const uint32_t& right)
{
	uint32_t pivotIndex = left + (right - left) / 2;
	uint32_t pivotValue = values[pivotIndex].second;
	uint32_t i = left, j = right;
	std::pair<A, B> temp;

	while (i <= j)
	{
		while (values[i].second < pivotValue) i++;
		while (values[j].second > pivotValue) j--;

		if (i <= j)
		{
			temp = values[i];
			values[i] = values[j];
			values[j] = temp;
			i++;
			j--;
		}
	}

	return i;
}

template <typename A, typename B>
void quicksort(std::vector<std::pair<A, B>>& values, const uint32_t& left, const uint32_t& right)
{
	if (left < right)
	{
		uint32_t pivotIndex = partition<A, B>(values, left, right);
		quicksort(values, left, pivotIndex - 1);
		quicksort(values, pivotIndex, right);
	}
}

//
// From: https://stackoverflow.com/a/37369858
//

// Fill the zipped vector with pairs consisting of the
// corresponding elements of a and b. (This assumes 
// that the vectors have equal length)
template <typename A, typename B>
void zip(const std::vector<A>& a, const std::vector<B>& b, std::vector<std::pair<A, B>>& zipped)
{
	std::transform(std::begin(a), std::end(a), std::begin(b), std::back_inserter(zipped), [](const A& a, const B& b) { return std::make_pair(a, b); });
}

// Write the first and second element of the pairs in 
// the given zipped vector into a and b. (This assumes 
// that the vectors have equal length)
template <typename A, typename B>
void unzip(const std::vector<std::pair<A, B>>& zipped, std::vector<A>& a, std::vector<B>& b)
{
	for (size_t i = 0; i < a.size(); i++)
	{
		a[i] = zipped[i].first;
		b[i] = zipped[i].second;
	}
}

template <typename A, typename B>
void zipSort(std::vector<A>& data, std::vector<B>& sortBy)
{
	std::vector<std::pair<A, B>> zipped;
	zip(data, sortBy, zipped);
	std::sort(std::begin(zipped), std::end(zipped), [](const std::pair<A, B>& a, const std::pair<A, B>& b) { return a.second < b.second; });
	//	quicksort<A, B>(zipped, 0, zipped.size() - 1);

	unzip(zipped, data, sortBy);

}

//
// From: https://stackoverflow.com/a/7008476
//
template <typename Map, typename F>
void map_erase_if(Map& m, F pred)
{
	typename Map::iterator i = m.begin();
	while ((i = std::find_if(i, m.end(), pred)) != m.end())
		m.erase(i++);
}

template <typename T>
static std::string ToStringWithPrecision(const T val, const uint32_t& n = 6)
{
	std::ostringstream out;
	out.precision(n);
	out << std::fixed << val;
	return out.str();
}

static uint32_t CalcOrder(double val)
{
	uint32_t cnt = 0;

	while (val / 1000.0 > 1.0)
	{
		val /= 1000.0;
		cnt++;
	}

	return cnt;
}

static std::string GetPrefix(const uint32_t& order)
{
	switch (order)
	{
		// Byte
		case 0:
			return " B";
		// Kilo
		case 1:
			return " KB";
		// Mega Byte
		case 2:
			return " MB";
		// Giga Byte
		case 3:
			return " GB";
		// Tera Byte
		case 4:
			return " TB";
	}

	return "UNKNOWN ORDER: " + std::to_string(order);
}

static inline std::string SizeWithSuffix(const double& val)
{
	std::string str = "";
	uint32_t order = CalcOrder(val);

	str = ToStringWithPrecision(val / (std::pow(1000.0, order)), 2);

	str.append(GetPrefix(order));

	return str;
}

static inline std::string SizeWithSuffix(const uint64_t& val)
{
	return SizeWithSuffix(static_cast<double>(val));
}

static inline std::string SizeWithSuffix(const int64_t& val)
{
	return SizeWithSuffix(static_cast<double>(val));
}

// //////////////////////////////////////
// ====== Get Process Memory Usage ======
// //////////////////////////////////////

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX // Disable the build in MIN/MAX macros to prevent collisions
#endif
#include <windows.h>
#include <psapi.h>
#endif

#ifdef __linux__ 
#include <stdio.h>
#endif


std::size_t GetCurrentRSS()
{
#ifdef _WIN32 // Windows
	PROCESS_MEMORY_COUNTERS pmc;
	GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
	return static_cast<std::size_t>(pmc.WorkingSetSize);
#endif

#ifdef __linux__ // Linux
	std::size_t tSize;
	std::size_t resident;
	std::ifstream in("/proc/self/statm");

	if (!in.is_open())
	{
		std::cerr << "Unable to read /proc/self/statm for current process" << std::endl;
		return 0;
	}

	in >> tSize >> resident;
	in.close();

	return resident * sysconf(_SC_PAGE_SIZE);
#endif
}

std::string GetMemString()
{
	return SizeWithSuffix(GetCurrentRSS());
}
