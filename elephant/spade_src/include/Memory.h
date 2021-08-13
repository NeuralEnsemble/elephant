/* 
 *  File: Memory.h
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

#include "Types.h"
#include "Logger.h"
#include "FPNode.h"
#include "Utils.h"

#include <iostream>
#include <sstream>

template<typename T>
class Memory
{
	DISABLE_COPY_ASSIGN_MOVE(Memory)
		struct MemoryState
	{
		std::size_t inUse;
		std::size_t nextIdx;
		std::size_t memBlock;
		T* pFrees;
	};


public:
	Memory() :
		m_elems(0),
		m_inUse(0),
		m_nextIdx(0),
		m_memBlock(0),
		m_pMem(),
		m_pFrees(nullptr),
		m_memStates()
	{}

	Memory(const std::size_t& elems) :
		m_elems(elems),
		m_inUse(0),
		m_nextIdx(0),
		m_memBlock(0),
		m_pMem(),
		m_pFrees(nullptr),
		m_memStates()
	{
		allocNewMemBlock();
	}

	~Memory()
	{
		for (T* pP : m_pMem)
			delete[] pP;
	}

	void Init(const std::size_t& elems)
	{
		m_elems = elems;
		allocNewMemBlock();
	}

	void PushState()
	{
#ifdef MEMORY_VERBOSE
		LOG_DEBUG << "Push InUse=" << m_inUse << "; NextIDX=" << m_nextIdx << "; memBlock=" << m_memBlock << std::endl;
#endif
		MemoryState ms;
		ms.inUse = m_inUse;
		ms.nextIdx = m_nextIdx;
		ms.memBlock = m_memBlock;
		ms.pFrees = m_pFrees;
		m_memStates.push(ms);
	}

	void PopState()
	{
		if (m_memStates.empty()) return;

		MemoryState ms = m_memStates.top();
		m_memStates.pop();
#ifdef MEMORY_VERBOSE
		LOG_DEBUG << "Pop (before) InUse=" << m_inUse << "; NextIDX=" << m_nextIdx << "; memBlock=" << m_memBlock << std::endl;
#endif
		m_inUse = ms.inUse;
		m_nextIdx = ms.nextIdx;
		m_memBlock = ms.memBlock;
		m_pFrees = ms.pFrees;
#ifdef MEMORY_VERBOSE
		LOG_DEBUG << "Pop (after)  InUse=" << m_inUse << "; NextIDX=" << m_nextIdx << "; memBlock=" << m_memBlock << std::endl;
#endif
	}

	T* Alloc()
	{
#ifdef MEMORY_VERBOSE
		LOG_DEBUG << "Alloc ... " << std::flush;
#endif
		m_inUse++;
		if (m_pFrees)
		{
			T* pNode = m_pFrees;
			m_pFrees = pNode->GetFreeNode();
			pNode->SetFreeNode(nullptr);
#ifdef MEMORY_VERBOSE
			LOG_DEBUG << "(Free) Done" << std::endl;
#endif
			return pNode;
		}

		if (m_nextIdx >= m_elems)
			allocNewMemBlock();

#ifdef MEMORY_VERBOSE
		LOG_DEBUG << "Done" << std::endl;
#endif
		return &m_pMem[m_memBlock - 1][m_nextIdx++];
	}

	void Free(T* pNode)
	{
#ifdef MEMORY_VERBOSE
		LOG_DEBUG << "Free ... " << std::flush;
#endif
		pNode->SetFreeNode(m_pFrees);
		m_pFrees = pNode;
		m_inUse--;
#ifdef MEMORY_VERBOSE
		LOG_DEBUG << "Done" << std::endl;
#endif
	}

	void Clear()
	{
		m_inUse = 0;
		m_memBlock = 1;
		m_nextIdx = 0;
		m_pFrees = nullptr;
	}

private:
	void allocNewMemBlock()
	{
#ifdef MEMORY_VERBOSE
		LOG_DEBUG << "Allocating new Memory Block ... " << std::flush;
#endif
		// After restoring a pushed state that was on a different memory block make sure to not allocate the next block again
		if (m_memBlock == m_pMem.size())
			m_pMem.push_back(new T[m_elems]());

		m_memBlock++;
		m_nextIdx = 0;
#ifdef MEMORY_VERBOSE
		LOG_DEBUG << "Done" << std::endl;
#endif
	}

	friend std::ostream& operator<<(std::ostream& os, const Memory<T>& rhs)
	{
		os << "Elements  : " << rhs.m_elems << std::endl;
		os << "Mem Blocks: " << rhs.m_memBlock << std::endl;
		os << "In Use    : " << rhs.m_inUse << std::endl;
		os << "Next Idx  : " << rhs.m_nextIdx << std::endl;

		for (std::size_t i = 0; i < rhs.m_memBlock; i++)
		{
			os << "Mem Block [" << i << "]" << std::endl;
			for (std::size_t j = 0; j < rhs.m_elems; j++)
				os << rhs.m_pMem[i][j] << std::endl;
		}

		return os;
	}

private:
	std::size_t m_elems;
	std::size_t m_inUse;
	std::size_t m_nextIdx;
	std::size_t m_memBlock;
	std::vector<T*> m_pMem;
	T* m_pFrees;
	std::stack<MemoryState> m_memStates;
};

using FPNMemory = Memory<FPNode>;
