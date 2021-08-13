/* 
 *  File: Logger.h
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

#include <sstream>
#include <iostream>

enum class  Verbosity
{
	VB_DEBUG = 0,
	VB_VERBOSE = 1,
	VB_INFO = 2,
	VB_WARNING = 3,
	VB_ERROR = 4,
	VB_NONE = 255
};

class Logger
{
	using EndlType = std::ostream& (std::ostream&);

public:
	Logger(Verbosity lvl, Verbosity verbosity = Verbosity::VB_VERBOSE) :
		m_lvl(lvl),
		m_verbosity(verbosity),
		m_outStream(std::cout)
	{}

	void SetVerbosity(Verbosity v)
	{
		m_verbosity = v;
	}

	Logger& operator<<(EndlType endl)
	{
		if (m_lvl >= m_verbosity)
			m_outStream << endl;
		return *this;
	}

	template<typename T>
	Logger& operator<<(const T& data)
	{
		if (m_lvl >= m_verbosity)
			m_outStream << data;
		return *this;
	}


private:
	Verbosity m_lvl;
	Verbosity m_verbosity;
	std::ostream& m_outStream;
};

static Logger g_debug(Verbosity::VB_DEBUG);
static Logger g_verbose(Verbosity::VB_VERBOSE);
static Logger g_info(Verbosity::VB_INFO);
static Logger g_warning(Verbosity::VB_WARNING);
static Logger g_error(Verbosity::VB_ERROR);

#ifndef EVAL_MODE
#define LOG_DEBUG g_debug
#define LOG_VERBOSE g_verbose
#define LOG_INFO g_info
#define LOG_WARNING g_warning
#define LOG_ERROR g_error
#else
static Logger g_none(Verbosity::VB_DEBUG, Verbosity::VB_NONE);
#define LOG_DEBUG g_none
#define LOG_VERBOSE g_none
#define LOG_INFO g_none
#define LOG_WARNING g_none
#define LOG_ERROR g_none
#endif

#define LOG_INFO_EVAL g_info

void SetVerbosity(Verbosity v)
{
	g_debug.SetVerbosity(v);
	g_verbose.SetVerbosity(v);
	g_info.SetVerbosity(v);
	g_warning.SetVerbosity(v);
	g_error.SetVerbosity(v);
}

template <typename E>
constexpr typename std::underlying_type<E>::type ToUnderlying(E e) noexcept
{
	return static_cast<typename std::underlying_type<E>::type>(e);
}

Verbosity ToVerbosity(const int32_t& val)
{
	if (val < ToUnderlying(Verbosity::VB_DEBUG) || val > ToUnderlying(Verbosity::VB_ERROR))
		return Verbosity::VB_INFO;

	return static_cast<Verbosity>(val);
}