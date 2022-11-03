/*
 *  File: HeapAlloc.h
 *  Copyright (c) 2021 Florian Porrmann
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

#ifdef _WIN32

// In a multi-threaded environmet the Windows runtime library spends a lot of time waiting
// when allocating memory as each thread uses the same heap. Therefore, by creating a dedicated
// heap for each thread, the wait time is removed and the overall performance increases significantly.
// Implementation based on: https://stackoverflow.com/a/63749764

#include "Logger.h"

#ifndef NOMINMAX
#define NOMINMAX // Disable the build in MIN/MAX macros to prevent collisions
#endif
#include <windows.h>

namespace
{
thread_local HANDLE g_tl_heapHandle;

const char* lastSystemErrorText()
{
	static char err[BUFSIZ];
	FormatMessageW(FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPWSTR>(err), 255, NULL);
	return err;
}

HANDLE createNewHeap()
{
	HANDLE handle = HeapCreate(0, 0, 0);
	if (handle == nullptr)
		LOG_ERROR << "Could not create large object heap" << lastSystemErrorText() << std::endl;

	return handle;
}

inline bool heapFree(HANDLE handle, void* ptr)
{
	bool success = HeapFree(handle, 0, ptr);
	if (!success)
		LOG_ERROR << "Failed to free memory: " << lastSystemErrorText() << std::endl;

	return success;
}

inline void* newImpl(std::size_t bytes)
{
	// Allocate additional space to store the handle for the allocating heap.
	std::size_t sz = bytes + sizeof(HANDLE);

	if (g_tl_heapHandle == nullptr)
		g_tl_heapHandle = createNewHeap();

	void* ptr = HeapAlloc(g_tl_heapHandle, 0, sz);
	if (ptr)
	{
		*(reinterpret_cast<HANDLE*>(ptr)) = g_tl_heapHandle;
		return reinterpret_cast<void*>((reinterpret_cast<char*>(ptr)) + sizeof(HANDLE));
	}
	else
		throw std::bad_alloc{};
}

inline void deleteImpl(void* ptr)
{
	if (!ptr) return;

	void* handlePtr = reinterpret_cast<void*>(((reinterpret_cast<char*>(ptr)) - sizeof(HANDLE)));
	HANDLE handle = *(reinterpret_cast<HANDLE*>(handlePtr));
	if(handle)
		heapFree(handle, handlePtr);
}
}

class HeapAlloc
{
	public:
		void* operator new(std::size_t sz)
		{
			return newImpl(sz);
		}

		void* operator new[](std::size_t sz)
		{
			return newImpl(sz);
		}

		void operator delete(void* ptr) noexcept
		{
			deleteImpl(ptr);
		}

		void operator delete[](void* ptr) noexcept
		{
			deleteImpl(ptr);
		}
};
#endif
