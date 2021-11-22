/* 
 *  File: FrequencyRef.h
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

#include <sstream>

struct FrequencyRef
{
	Support support;

	FrequencyRef(const std::size_t idx) :
		support(0),
		m_idx(idx),
		m_refs()
	{}

	~FrequencyRef();

	ItemC item() const;

	const std::size_t& Idx() const
	{
		return m_idx;
	}

	void SetIdx(const std::size_t& idx)
	{
		m_idx = idx;
	}

	bool operator< (const struct FrequencyRef& rhs) const
	{
		if (support == rhs.support) return m_idx < rhs.m_idx;
		return support < rhs.support;
	}

	bool operator> (const struct FrequencyRef& rhs) const
	{
		if (support == rhs.support) return m_idx < rhs.m_idx;
		return support > rhs.support;
	}

	bool operator< (const uint64_t& sup) const
	{
		return support < sup;
	}

	bool operator> (const uint64_t& sup) const
	{
		return support > sup;
	}

	bool operator== (const struct FrequencyRef& rhs) const
	{
		return this->support == rhs.support;
	}

	void Inc(struct ItemRef* pItemRef);

	void Dec(struct ItemRef* pItemRef)
	{
		UNUSED(pItemRef);
		support--;
		m_refs.erase(std::remove(std::begin(m_refs), std::end(m_refs), pItemRef), std::end(m_refs));
	}

private:
	std::size_t m_idx;
	std::vector<struct ItemRef*> m_refs;
};


struct ItemRef
{
	ItemC item;
	struct FrequencyRef* pFRef;

	ItemRef(const ItemC& item) :
		item(item),
		pFRef(nullptr)
	{}

	ItemRef(const ItemRef& ref) :
		item(ref.item),
		pFRef(ref.pFRef)
	{}

	~ItemRef() {}

	ItemRef& operator=(const ItemRef& ref)
	{
		this->item = ref.item;
		this->pFRef = ref.pFRef;
		return *this;
	}


	void SetRef(struct FrequencyRef* pRef)
	{
		pFRef = pRef;
	}

	bool operator!= (const ItemRef& rhs) const
	{
		return item != rhs.item;
	}

	bool operator< (const ItemRef& rhs) const
	{
		return item < rhs.item;
	}

	bool operator> (const ItemRef& rhs) const
	{
		return item > rhs.item;
	}

	std::size_t Idx() const
	{
		if (pFRef == nullptr) return IDX_MAX;
		return pFRef->Idx();
	}

private:
	friend std::ostream& operator<<(std::ostream& os, const ItemRef& ref)
	{
		os << ref.item;
		return os;
	}
};

FrequencyRef::~FrequencyRef()
{
	// Invalidate all related items
	for (ItemRef* pRef : m_refs)
	{
		if (pRef) pRef->pFRef = nullptr;
	}
}

ItemC FrequencyRef::item() const
{
	return m_refs.front()->item;
}

void FrequencyRef::Inc(struct ItemRef* pItemRef)
{
	support++;
	m_refs.push_back(pItemRef);
	pItemRef->SetRef(this);
}

using FrequencyRefShr = std::shared_ptr<FrequencyRef>;

using TransactionC = std::vector<ItemRef>;
using DataBase = std::vector<TransactionC>;


#define ITEM_PAIR ItemC, FrequencyRefShr

using FrequencyMapC = std::map<ITEM_PAIR>;
using RefPair = std::pair<ITEM_PAIR>;
