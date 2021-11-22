/* 
 *  File: FPNode.h
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

#include <string>
#include <iostream>
#include <sstream>

struct FPNode
{
	std::size_t id;
	Support support;
	struct FPNode* parent;
	struct FPNode* succ;
#ifdef DEBUG
	ItemC item;
#endif

	FPNode() :
		id(std::numeric_limits<size_t>::max()),
		support(0),
		parent(nullptr),
		succ(nullptr)
#ifdef DEBUG
		, item(0)
#endif
	{}

#ifdef DEBUG
	~FPNode()
	{
		parent = nullptr;
		succ = nullptr;
	}
#endif

	void SetFreeNode(FPNode* pNode)
	{
		parent = pNode;
	}

	FPNode* GetFreeNode() const
	{
		return parent;
	}

	void PrintTree(const std::string& prefix = "") const
	{
		const std::string space = "    ";
		const std::string connectSpace = u8"│   ";
		const bool isLast = parent == nullptr;

		LOG_VERBOSE << prefix;
		LOG_VERBOSE << (isLast ? u8"└──" : u8"├──");
		// print the value of the node
#ifdef DEBUG
		LOG_DEBUG << (char)item << ":" << support << std::endl;
#endif

		// enter the next tree level - left and right branch
		if (parent != nullptr)
			parent->PrintTree(prefix + (isLast ? space : connectSpace));
		if (succ != nullptr)
			succ->PrintTree(prefix/* + (isLast ? space : connectSpace)*/);
	}

	friend std::ostream& operator<<(std::ostream& os, const FPNode& rhs)
	{
		os << "id=" << rhs.id << "; support=" << rhs.support << "; parent=" << rhs.parent << "; succ=" << rhs.succ;
		return os;
	}

};
