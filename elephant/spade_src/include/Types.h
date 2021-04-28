/* 
 *  File: Types.h
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

#include "Defines.h"

#include <stdint.h>
#include <limits>
#include <vector>
#include <map>

using ItemC = uint32_t;
using Support = uint32_t;
using ItemID = std::size_t;

using Transaction = std::vector<ItemC>;
using Transactions = std::vector<Transaction>;
using FrequencyMap = std::map<ItemC, Support>;

const std::size_t IDX_MAX = std::numeric_limits<std::size_t>::max();
const Support SUPP_MAX = std::numeric_limits<Support>::max();
const ItemC ITEM_MAX = std::numeric_limits<ItemC>::max();
const ItemID ITEM_ID_MAX = std::numeric_limits<ItemID>::max();

using ItemOccurence = std::pair<ItemC, Support>;
using ItemOccurences = std::vector<ItemOccurence>;

using PatternType = ItemID;
using PatternVec = std::vector<PatternType>;
using PatternPair = std::pair<PatternVec, Support>;

