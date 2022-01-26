/* 
 *  File: ClosedTree.h
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

/*
 * The following implementation is in large parts, based on the closed item set
 * filter implemented by Christian Borgelt (https://borgelt.net/fpgrowth.html)
 */

#pragma once

#include "Memory.h"
#include "Types.h"
#include "Utils.h"

struct ClosedNode
{
	ItemID item;
	Support supp;
	ClosedNode* sibling;
	ClosedNode* children;

	void SetFreeNode(ClosedNode* pNode)
	{
		sibling = pNode;
	}

	ClosedNode* GetFreeNode() const
	{
		return sibling;
	}
};

using CNMemory = Memory<ClosedNode>;

class ClosedTree
{
	DISABLE_COPY_ASSIGN_MOVE(ClosedTree)

public:
	ClosedTree() :
		m_pMem(nullptr),
		m_item(ITEM_MAX),
		m_max(0),
		m_root()
	{
	}

	~ClosedTree()
	{
		delete m_pMem;
	}

	void Init()
	{
		if (m_pMem == nullptr) m_pMem = new CNMemory(4095);
		m_item         = ITEM_MAX;
		m_max          = 0;
		m_root.sibling = m_root.children = nullptr;
		m_root.item                      = ITEM_MAX;
		m_root.supp                      = 0;
	}

	bool Valid() const
	{
		return m_item < ITEM_MAX;
	}

	void Add(ItemID* pItems, int32_t n, Support supp)
	{
		ItemID i;
		ClosedNode** p;
		ClosedNode* pNode;

		if (supp > m_max) m_max = supp;

		pNode = &m_root;

		do
		{
			if (supp > pNode->supp) pNode->supp = supp;
			if (--n < 0) return;

			i = *pItems++;
			p = &pNode->children;
			while (*p && ((*p)->item > i)) p = &(*p)->sibling;
			pNode = *p;
		} while (pNode && (pNode->item == i));

		pNode          = m_pMem->Alloc();
		pNode->supp    = supp;
		pNode->item    = i;
		pNode->sibling = *p;
		*p             = pNode;

		while (--n >= 0)
		{
			pNode = pNode->children = m_pMem->Alloc();
			pNode->supp             = supp;
			pNode->item             = *pItems++;
			pNode->sibling          = nullptr;
		}

		pNode->children = nullptr;
	}

	ClosedTree* Project(ClosedTree* pDst)
	{
		ClosedNode* p;

		pDst->Init();

		pDst->SetItem(ITEM_MAX - 1);
		pDst->SetMax(0);
		m_max                = 0;
		pDst->GetRoot().supp = 0;

		p = &m_root;

		if (!p->children) return pDst;
		p = p->children = prune(p->children, m_item);

		if (!p || (p->item != m_item)) return pDst;

		pDst->GetRoot().supp = p->supp;
		m_max                = p->supp;

		if (p->children)
			pDst->GetRoot().children = p = pDst->copy(p->children);

		p           = &m_root;
		p->children = prune(p->children, m_item + 1);

		return pDst;
	}

	void Prune(const ItemID& item)
	{
		ClosedNode* p;

		m_item = item;
		p      = &m_root;
		p = p->children = prune(p->children, item);
		m_max           = (p && (p->item == item)) ? p->supp : 0;
	}

	void Clear()
	{
		m_pMem->Clear();
		m_max           = 0;
		m_item          = ITEM_MAX;
		m_root.sibling  = nullptr;
		m_root.children = nullptr;
		m_root.supp     = 0;
	}

	const ItemID& GetItem() const
	{
		return m_item;
	}

	const Support& GetMax() const
	{
		return m_max;
	}

	const Support& GetSupport() const
	{
		return m_root.supp;
	}

	ClosedNode& GetRoot()
	{
		return m_root;
	}

	CNMemory* GetMem()
	{
		return m_pMem;
	}

	void SetItem(const ItemID& item)
	{
		m_item = item;
	}

	void SetMax(const Support& max)
	{
		m_max = max;
	}

private:
	ClosedNode* merge(ClosedNode* s1, ClosedNode* s2)
	{
		ClosedNode* pOut;
		ClosedNode** ppEnd;
		ClosedNode* p;

		if (!s1) return s2;
		if (!s2) return s1;
		ppEnd = &pOut;

		while (1)
		{
			if (s1->item > s2->item)
			{
				*ppEnd = s1;
				ppEnd  = &s1->sibling;
				s1     = *ppEnd;
				if (!s1) break;
			}
			else if (s2->item > s1->item)
			{
				*ppEnd = s2;
				ppEnd  = &s2->sibling;
				s2     = *ppEnd;
				if (!s2) break;
			}
			else
			{
				s1->children = merge(s1->children, s2->children);
				if (s1->supp < s2->supp)
					s1->supp = s2->supp;

				p  = s2;
				s2 = s2->sibling;
				m_pMem->Free(p);

				*ppEnd = s1;
				ppEnd  = &s1->sibling;
				s1     = *ppEnd;
				if (!s1 || !s2) break;
			}
		}

		*ppEnd = (s1) ? s1 : s2;
		return pOut;
	}

	ClosedNode* prune(ClosedNode* node, const ItemID& item)
	{
		ClosedNode *p, *b = NULL;

		while (node && (node->item > item))
		{
			node->children = p = prune(node->children, item);
			if (p) b = (!b) ? p : merge(b, p);
			p    = node;
			node = node->sibling;
			m_pMem->Free(p);
		}

		return (!node) ? b : (!b) ? node
								  : merge(b, node);
	}

	ClosedNode* copy(const ClosedNode* pSrc)
	{
		ClosedNode* pDst;
		ClosedNode* pNode;
		ClosedNode** ppEnd = &pDst;
		ClosedNode* pC;

		do
		{
			*ppEnd = pNode = m_pMem->Alloc();
			if (!pNode) return nullptr;

			pNode->item = pSrc->item;
			pNode->supp = pSrc->supp;
			pC          = pSrc->children;
			if (pC)
			{
				pC = copy(pC);
				if (!pC) return nullptr;
			}

			pNode->children = pC;
			ppEnd           = &pNode->sibling;
			pSrc            = pSrc->sibling;
		} while (pSrc);

		*ppEnd = nullptr;
		return pDst;
	}

private:
	CNMemory* m_pMem;
	ItemID m_item;
	Support m_max;
	ClosedNode m_root;
};
