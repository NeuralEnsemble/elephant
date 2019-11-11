"""
This module provides an implementation of the formal concept analysis (FCA) in
pure Python. It is used by the SPADE analysis. The code builds on C. Lindig's
Fast Concept Analysis work (1999,2002).

Original code available at:

Copyright (C) 2008-2019 by Dominik Endres (dominik.endres@gmail.com).
Relicensed for Elephant by permission.

Usage example:
--------------

>>> relation=[]
>>> relation+=[('monkeyHand','neuron2')]
>>> relation+=[('monkeyFace','neuron1')]
>>> relation+=[('monkeyFace','neuron2')]
>>> relation+=[('humanFace','neuron1')]
>>> relation+=[('spider','neuron3')]
>>> concepts=FormalConcepts(relation)
>>> concepts.computeLattice()
>>> print(concepts)

If you generate publications based on this code, please cite the following
paper:

Endres D., Adam R., Giese M.A. & Noppeney U.. (2012).
Understanding the Semantic Structure of Human fMRI Brain Recordings with Formal
Concept Analysis. Proceedings of the 10th International Conference on Formal
Concept Analysis (ICFCA 2012) LNAI 7278, Springer,pp. 96-111.
"""

import bisect
from collections import defaultdict

import tqdm


class FormalConcept(object):
    """
    A formal concept is comprised of an extent and and intent.
    Furthermore, intentIndexes is an ordered list of attribute indexes for
    lectic ordering. Also contains sets of introduced attibutes and objects
    and lectically ordered lists of upper and lower neighbours.
    """

    def __init__(self, extent=frozenset(), intent=frozenset(),
                 intentIndexes=[]):
        """ intent/extent are a frozensets because they need to be hashable."""
        self.cnum = 0
        self.extent = extent
        self.intent = intent
        self.introducedAttributes = set()
        self.introducedObjects = set()
        self.intentIndexes = intentIndexes
        self.upperNeighbours = []
        self.lowerNeighbours = []
        self.visited = False  # for lattice traversal

        # attributes that were introduced closest in upwards direction
        # useful for naming a concept that introduces no attributes.
        # recompute after pruning!
        self.closestIntroducedAttributes = []
        # all attributes that are introduced in the downset of this concept;
        # useful for building search list.
        self.downsetAttributes = set()

    def __eq__(self, other):
        """lectic order on intentIndexes."""
        return self.intentIndexes == other.intentIndexes

    def __lt__(self, other):
        """lectic order on intentIndexes."""
        if self.intentIndexes == other.intentIndexes:
            return -1
        i1 = 0
        i2len = len(other.intentIndexes)
        for a1 in self.intentIndexes:
            if i1 >= i2len:
                return -1
            a2 = other.intentIndexes[i1]
            if a1 > a2:
                return 1
            elif a1 < a2:
                return -1
            i1 += 1
        return 1

    def __repr__(self):
        """ print the concept."""
        strrep = "concept no:" + str(self.cnum) + "\n"
        strrep += "extent:" + repr(self.extent) + "\n"
        strrep += "intent:" + repr(self.intent) + "\n"
        strrep += "introduced objects:" + repr(self.introducedObjects) + "\n"
        strrep += "introduced attributes:" + repr(
            self.introducedAttributes) + "\n"
        if hasattr(self, "stability"):
            strrep += "stability: {0:1.4f}".format(
                self.stability) + "\n"
        strrep += "upper neighbours: "
        for un in self.upperNeighbours:
            strrep += str(un.cnum) + ", "
        strrep += "\n"
        strrep += "lower neighbours: "
        for ln in self.lowerNeighbours:
            strrep += str(ln.cnum) + ", "
        strrep += "\n"
        return strrep


class FormalContext(object):
    """
    The formal context.
    Builds dictionaries object=>attributes and vice versa for faster closure
    computation.
    Set of objects and attributes are kept in lists rather than sets for lectic
    ordering of concepts.
    """

    def __init__(self, relation, objects=None, attributes=None):
        """
        'relation' has to be an iterable container of tuples.
        If objects or attributes are not supplied, determine from relation.
        """
        # map from object=> set of attributes of this object
        self.objectsToAttributes = defaultdict(set)
        # map from attributes => set of objects of this attribute
        self.attributesToObjects = defaultdict(set)
        # objects and attributes are kept in lists rather than sets for lectic
        # ordering of concepts.
        self.objects = set()
        self.attributes = set()
        if objects is not None:
            self.objects.update(objects)
        if attributes is not None:
            self.attributes.update(attributes)

        for obj, att in relation:
            self.objects.add(obj)
            self.attributes.add(att)
            self.objectsToAttributes[obj].add(att)
            self.attributesToObjects[att].add(obj)

        self.attributes = sorted(self.attributes, reverse=True)

    def objectsPrime(self, objectSet):
        """return a frozenset of all attributes which are shared by members
        of objectSet. """
        if len(objectSet) == 0:
            return frozenset(self.attributes)
        oiter = iter(objectSet)
        opr = self.objectsToAttributes[next(oiter)].copy()
        for obj in oiter:
            opr.intersection_update(self.objectsToAttributes[obj])
        return frozenset(opr)

    def attributesPrime(self, attributeSet):
        """return a set of all objects which have all attributes in
        attribute set. """
        if len(attributeSet) == 0:
            return frozenset(self.objects)
        aiter = iter(attributeSet)
        apr = self.attributesToObjects[next(aiter)].copy()
        for att in aiter:
            apr.intersection_update(self.attributesToObjects[att])
        return frozenset(apr)

    def updateIntent(self, intent, object):
        """return intersection of intent and all attributes of object."""
        return frozenset(intent.intersection(self.objectsToAttributes[object]))

    def indexList(self, attributeSet):
        """return ordered list of attribute indexes. For lectic ordering of
        concepts. """
        ilist = sorted(map(self.attributes.index, attributeSet))
        return ilist


class FormalConcepts(object):
    """ Computes set of concepts from a binary relation by an algorithm
    similar to C. Lindig's Fast Concept Analysis (2002).
    """

    def __init__(self, relation, objects=None, attributes=None, verbose=False):
        """ 'relation' has to be an iterable container of tuples. If objects
        or attributes are not supplied, determine from relation. """
        self.context = FormalContext(relation, objects, attributes)
        self.concepts = []  # a lectically ordered list of concepts"
        self.intentToConceptDict = dict()
        self.verbose = verbose

    def computeUpperNeighbours(self, concept):
        """
        This version of upperNeighbours runs fast enough in Python to be
        useful.
        Based on a theorem from C. Lindig's (1999) PhD thesis.
        Returns list of upper neighbours of concept."""
        # The set of all objects g which are not in concept's extent G and
        # might therefore be used to create upper neighbours via ((G u g)'',(G
        # u g)')
        upperNeighbourGeneratingObjects = self.context.objects.difference(
            concept.extent)
        # dictionary of intent => set of generating objects
        upperNeighbourCandidates = defaultdict(set)
        for g in upperNeighbourGeneratingObjects:
            # an intent of a concept >= concept. Computed by intersecting i(g)
            # with concept.intent,
            # where i(g) is the set of all attributes of g.
            intent = self.context.updateIntent(concept.intent, g)
            # self.intentToConceptDict is a dictionary of all concepts computed
            # so far.
            if intent not in self.intentToConceptDict:
                # Store every concept in self.conceptDict, because it will
                # eventually be used and the closure is expensive to compute
                extent = self.context.attributesPrime(intent)
                curConcept = FormalConcept(extent, intent,
                                           self.context.indexList(intent))
                self.intentToConceptDict[intent] = curConcept

            # remember which g generated what concept
            upperNeighbourCandidates[intent].add(g)

        neighbours = []
        # find all upper neighbours by Lindig's theorem:
        # a concept C=((G u g)'',(G u g)') is an upper neighbour of (G,I) if
        # and only if (G u g)'' \ G = set of all g which generated C.
        for intent, generatingObjects in upperNeighbourCandidates.items():
            extraObjects = self.intentToConceptDict[intent].extent.difference(
                concept.extent)
            if extraObjects == generatingObjects:
                neighbours.append(self.intentToConceptDict[intent])
        return neighbours

    def numberConceptsAndComputeIntroduced(self):
        """ Numbers concepts and computes introduced objects and attributes"""
        for curConNum, curConcept in enumerate(self.concepts):
            curConcept.cnum = curConNum
            curConcept.introducedObjects = set(curConcept.extent)
            for ln in curConcept.lowerNeighbours:
                curConcept.introducedObjects.difference_update(ln.extent)
            curConcept.introducedAttributes = set(curConcept.intent)
            for un in curConcept.upperNeighbours:
                curConcept.introducedAttributes.difference_update(un.intent)

    def computeLattice(self):
        """ Computes concepts and lattice.
        self.concepts contains lectically ordered list of concepts after
        completion."""
        intent = self.context.objectsPrime(set())
        extent = self.context.attributesPrime(intent)
        curConcept = FormalConcept(extent, intent,
                                   self.context.indexList(intent))
        self.concepts = [curConcept]
        self.intentToConceptDict[curConcept.intent] = curConcept
        curConceptIndex = 0
        progress_bar = tqdm.tqdm(disable=not self.verbose,
                                 desc="computeLattice")
        while curConceptIndex >= 0:
            upperNeighbours = self.computeUpperNeighbours(curConcept)
            for upperNeighbour in upperNeighbours:
                upperNeighbourIndex = bisect.bisect(self.concepts,
                                                    upperNeighbour)
                if upperNeighbourIndex == 0 or self.concepts[
                        upperNeighbourIndex - 1] != upperNeighbour:
                    self.concepts.insert(upperNeighbourIndex, upperNeighbour)
                    curConceptIndex += 1

                curConcept.upperNeighbours.append(upperNeighbour)
                upperNeighbour.lowerNeighbours.append(curConcept)

            curConceptIndex -= 1
            curConcept = self.concepts[curConceptIndex]
            progress_bar.update()

        self.numberConceptsAndComputeIntroduced()
