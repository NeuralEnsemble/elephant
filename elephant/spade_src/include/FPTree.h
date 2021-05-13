/*
 *  File: FPTree.h
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

#include "HeapAlloc.h"
#include "Types.h"
#include "Logger.h"
#include "Utils.h"
#include "Memory.h"
#include "FrequencyRef.h"


struct FPHead
#ifdef _WIN32
 : public HeapAlloc
#endif
{
	ItemID item;
	Support support;
	FPNode* list;
	FPNMemory* pMemory;
};

struct FPTree
#ifdef _WIN32
 : public HeapAlloc
#endif
{
	DISABLE_COPY_ASSIGN_MOVE(FPTree)

		std::size_t cnt;
	FPNode root;
	FPHead* pHeads;
	std::uint32_t* pIdx2Id;
	ItemC* pId2Item;
	FPNMemory* pMemory;

	FPTree() :
		cnt(0),
		root(),
		pHeads(nullptr),
		pIdx2Id(nullptr),
		pId2Item(nullptr),
		pMemory(nullptr)
	{}

	FPTree(const std::size_t& items, uint32_t* pIdx2Id_g, ItemC* pId2Item_g, FPNMemory* pMem) :
		cnt(items),
		root(),
		pHeads(nullptr),
		pIdx2Id(pIdx2Id_g),
		pId2Item(pId2Item_g),
		pMemory(pMem)
	{
		pHeads = new FPHead[cnt];
	}

	FPTree(const std::vector<RefPair>& F, uint32_t* pIdx2Id_g, ItemC* pId2Item_g, FPNMemory* pMem) :
		cnt(F.size()),
		root(),
		pHeads(nullptr),
		pIdx2Id(pIdx2Id_g),
		pId2Item(pId2Item_g),
		pMemory(pMem)
	{
		pHeads = new FPHead[cnt];
		uint32_t id = 0;
		for (std::size_t idx = 0; idx < F.size(); idx++)
		{
			pId2Item[idx] = F[idx].first;
			pIdx2Id[idx] = id;
			pHeads[id].item = idx;
			F[idx].second->SetIdx(idx);
			pHeads[id].support = F[idx].second->support;
			pHeads[id].list = nullptr;
			pHeads[id].pMemory = pMemory;
			id++;
		}
	}

	~FPTree()
	{
		delete[] pHeads;
	}

	void Add(const TransactionC& trans, const Support& support)
	{
		std::size_t n = trans.size();
		std::size_t i = 0;
		std::size_t id = 0;
		FPNode* c;
		FPNode* pNode = &root;
#ifdef DEBUG
		ItemC item;
#endif

		// Traverse tree until no valid child is found
		while (1)
		{
			pNode->support += support;
			if (i >= n) return;
#ifdef DEBUG
			item = trans[i].item;
#endif
			id = pIdx2Id[trans[i++].Idx()];
			c = pHeads[id].list;
			if (!c || (c->parent != pNode)) break;
			pNode = c;
		}

		// Create ne children until the transaction processed
		while (1)
		{
			c = pMemory->Alloc();
			c->id = id;
			c->support = support;
			c->parent = pNode;
			c->succ = pHeads[id].list;
#ifdef DEBUG
			c->item = item;
#endif
			pHeads[id].list = pNode = c;
			if (i >= n) return;
#ifdef DEBUG
			item = trans[i].item;
#endif
			id = pIdx2Id[trans[i++].Idx()];
		}
	}

	void Add(const std::size_t* pData, const std::size_t& n, const Support& support)
	{
		std::size_t i = 0;
		std::size_t id = 0;
		FPNode* c;
		FPNode* pNode = &root;

		// Traverse tree until no valid child is found
		while (1)
		{
			pNode->support += support;
			if (i >= n) return;
			id = pData[i++];
			c = pHeads[id].list;
			if (!c || (c->parent != pNode)) break;
			pNode = c;
		}

		// Create new children until the transaction processed
		while (1)
		{
			c = pMemory->Alloc();
			c->id = id;
			c->support = support;
			c->parent = pNode;
			c->succ = pHeads[id].list;
#ifdef DEBUG
			c->item = pId2Item[pHeads[id].item];
#endif
			pHeads[id].list = pNode = c;
			if (i >= n) return;
			id = pData[i++];
		}
	}

	void PrintTree() const
	{
		LOG_VERBOSE << "root" << std::endl;


		// Enter the next tree level - left and right branch
		for (std::size_t i = 0; i < cnt; i++)
		{
			if (pHeads[i].list != nullptr)
				pHeads[i].list->PrintTree("");
		}
	}
};
