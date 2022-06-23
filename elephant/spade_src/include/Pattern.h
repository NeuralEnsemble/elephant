/* 
 *  File: Pattern.h
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

#include "Logger.h"
#include "Types.h"
#include "Utils.h"

#include <iostream>

class Pattern
{
	DISABLE_COPY_ASSIGN_MOVE(Pattern)

	static constexpr std::size_t BLOCK_SIZE = 16384;

public:
	static constexpr PatternType OFFSET   = 2;
	static constexpr PatternType LEN_IDX  = 0;
	static constexpr PatternType SUPP_IDX = 1;
	static constexpr PatternType DATA_IDX = 2;

public:
	Pattern() :
		m_nextIdx(0),
		m_block(0),
		m_patternCnt(0),
		m_mem(),
		m_pEndPtr(nullptr)
	{
		allocNewPatternBlock();
	}

	~Pattern()
	{
		for (std::size_t i = 0; i < m_block; i++)
			delete[] m_mem[i];
	}

	template<typename T>
	class iterator
	{
		DISABLE_COPY_ASSIGN_MOVE(iterator)
	public:
		using ValueType = T;
		using Reference = T&;
		using Pointer   = T*;

		explicit iterator(std::vector<PatternType*> mem, const std::size_t& maxBlocks, PatternType* pItr = nullptr) :
			m_idx(0),
			m_block(0),
			m_maxBlocks(maxBlocks),
			m_mem(mem),
			m_pItr(pItr)
		{
			if (m_pItr == nullptr)
				m_pItr = m_mem[m_block];
		}

		bool operator!=(const iterator& other) const
		{
			return m_pItr != other.m_pItr;
		}
		iterator& operator++()
		{
			m_idx += static_cast<std::size_t>(m_pItr[LEN_IDX] + OFFSET);

			if ((m_idx >= BLOCK_SIZE) || (m_mem[m_block][m_idx] == 0 && (m_block + 1) < m_maxBlocks))
			{
				m_block++;
				m_idx = 0;
			}

			m_pItr = m_mem[m_block] + m_idx;

			return *this;
		}

		Pointer operator*() const
		{
			return m_pItr;
		}

		Pointer operator->() const
		{
			return m_pItr;
		}

	private:
		std::size_t m_idx;
		std::size_t m_block;
		std::size_t m_maxBlocks;
		std::vector<PatternType*> m_mem;
		PatternType* m_pItr;
	};

	using Iterator      = iterator<PatternType>;
	using ConstIterator = iterator<const PatternType>;

	const std::size_t& GetCount() const
	{
		return m_patternCnt;
	}

	bool Empty() const
	{
		return m_patternCnt == 0;
	}

	Iterator begin()
	{
		return Iterator(m_mem, m_block);
	}

	Iterator end()
	{
		return Iterator(m_mem, m_block, m_pEndPtr);
	}

	Iterator begin() const
	{
		return Iterator(m_mem, m_block);
	}

	Iterator end() const
	{
		return Iterator(m_mem, m_block, m_pEndPtr);
	}

	void AddPattern(const std::size_t& patternLength, const Support& support, PatternType* pData)
	{
		PatternType* pPattern = getNextPattern(patternLength);

		pPattern[LEN_IDX]  = patternLength; // Set pattern length
		pPattern[SUPP_IDX] = support;       // Set pattern support
		// Set pattern data
		std::memcpy(pPattern + OFFSET, pData, patternLength * sizeof(PatternType));

#ifdef DEBUG
		LOG_DEBUG << "Adding Pattern: " << std::flush;
		for (PatternType i = 0; i < patternLength; i++)
			LOG_DEBUG << (char)pData[i] << " ";
		LOG_DEBUG << "(" << support << ")" << std::endl;
#endif

		m_patternCnt++;
	}

	void AddPattern(const std::size_t& patternLength, const Support& support, PatternType* pData, const ItemC* pId2Item, const Support& maxSupport, const std::size_t& minNeuronCount, const ItemC& winLen)
	{
		const PatternType* pStart = pData;
		const PatternType* pEnd   = pData + patternLength;
		if (std::any_of(pStart, pEnd, [&winLen, &pId2Item](const PatternType& i) { return ((pId2Item[i & 0xFFFFFFFF]) % winLen) == 0; }))
		{
			if (support <= maxSupport)
			{
				std::set<PatternType> v;
				std::transform(pStart, pEnd, std::inserter(v, std::begin(v)), [&winLen, &pId2Item](const PatternType& i) { return (pId2Item[i & 0xFFFFFFFF]) / winLen; });
				if (v.size() >= minNeuronCount)
				{
					PatternType* pPattern = getNextPattern(patternLength);
					pPattern[LEN_IDX]     = patternLength; // Set pattern length
					pPattern[SUPP_IDX]    = support;       // Set pattern support
					// Set pattern data
					std::memcpy(pPattern + OFFSET, pData, patternLength * sizeof(PatternType));
#ifdef DEBUG
					LOG_DEBUG << "Adding Pattern: " << std::flush;
					for (PatternType i = 0; i < patternLength; i++)
						LOG_DEBUG << (char)pData[i] << " ";
					LOG_DEBUG << "(" << support << ")" << std::endl;
#endif
					m_patternCnt++;
				}
			}
		}
	}

private:
	PatternType* getNextPattern(const std::size_t& length)
	{
		if (m_nextIdx + (length + OFFSET) >= BLOCK_SIZE)
			allocNewPatternBlock();

		PatternType* pPtr = m_mem[m_block - 1] + m_nextIdx;
		m_nextIdx += length + OFFSET;

		m_pEndPtr = m_mem[m_block - 1] + m_nextIdx;

		return pPtr;
	}

	void allocNewPatternBlock()
	{
#ifdef PATTERN_VERBOSE
		LOG_DEBUG << "Allocating new Pattern Block ... " << std::flush;
#endif

		m_mem.push_back(new PatternType[BLOCK_SIZE]());

		m_block++;
		m_nextIdx = 0;

#ifdef PATTERN_VERBOSE
		LOG_DEBUG << "Done" << std::endl;
#endif
	}

private:
	std::size_t m_nextIdx;
	std::size_t m_block;
	std::size_t m_patternCnt;
	std::vector<PatternType*> m_mem;
	PatternType* m_pEndPtr;
};
