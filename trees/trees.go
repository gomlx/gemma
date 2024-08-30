package trees

import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/pkg/errors"
	"golang.org/x/exp/slices"
	"iter"
	"strings"
)

// Tree represent both a root of a tree, and a tree node.
//
// It can either be a Value or a Map of its children -- but not both.
type Tree[T any] struct {
	// Value is set for leaf nodes only.
	Value T

	// Map is set for non-leaf nodes (and nil in leaf nodes).
	Map map[string]*Tree[T]
}

func (n *Tree[T]) IsLeaf() bool { return n.Map == nil }

// Path is usually used as the path from the root node.
type Path []string

// New creates a new empty tree.
func New[T any]() *Tree[T] {
	return &Tree[T]{
		Map: make(map[string]*Tree[T]),
	}
}

// NewLeaf creates a new leaf node with the given value.
func NewLeaf[T any](value T) *Tree[T] {
	return &Tree[T]{Value: value}
}

// DefaultTreePath is used whenever an empty treePath is given.
var DefaultTreePath = []string{"#root"}

// Set value in treePath, populating intermediary nodes where needed.
//
// Empty values in treePath are not used.
//
// It returns an error if one is trying to set the value to an existing non-leaf node: nodes can either
// be a leaf or a Map (non-leaf), but not both.
func (tree *Tree[T]) Set(treePath Path, value T) error {
	// Remove empty ("") path components -- clone the slice, not to modify caller's slice.
	if slices.Index(treePath, "") > 0 {
		treePath = slices.DeleteFunc(slices.Clone(treePath),
			func(s string) bool {
				return s == ""
			})
	}
	remainingPath := treePath
	pathCount := 0
	node := tree
	for len(remainingPath) > 0 {
		pathElement := remainingPath[0]
		remainingPath = remainingPath[1:]
		if node.IsLeaf() {
			var t T
			return errors.Errorf("trees.Tree[%T].Set(%q) trying to create a path using an existing leaf node (%q) as a non-leaf node",
				t, treePath, treePath[:pathCount])
		}
		newNode := node.Map[pathElement]
		if newNode == nil {
			if len(remainingPath) == 0 {
				newNode = NewLeaf[T](value)
			} else {
				newNode = New[T]()
			}
			node.Map[pathElement] = newNode
		}
		node = newNode
		pathCount++
	}
	if !node.IsLeaf() {
		var t T
		return errors.Errorf("trees.Tree[%T].Set(%q) trying to set the value to a non-leaf node -- each node can either be a leaf node, or be a structural map of the tree",
			t, treePath)
	}
	node.Value = value
	return nil
}

// String implements fmt.String
func (tree *Tree[T]) String() string {
	var parts []string
	parts = nodeToString(parts, "/", tree, 0)
	return strings.Join(parts, "\n") + "\n"
}

func nodeToString[T any](parts []string, name string, subTree *Tree[T], indent int) []string {
	indentSpaces := strings.Repeat("  ", indent)
	indent++
	if len(subTree.Map) == 0 {
		// Leaf node.
		var valueAny any
		valueAny = subTree.Value
		if valueStr, ok := valueAny.(fmt.Stringer); ok {
			// T is a stringer:
			return append(parts, fmt.Sprintf("%s%q: %s", indentSpaces, name, valueStr))
		}
		// If not a stringer, use %v.
		return append(parts, fmt.Sprintf("%s%q: %v", indentSpaces, name, subTree.Value))
	}
	parts = append(parts, fmt.Sprintf("%s%q: {", indentSpaces, name))

	for _, key := range xslices.SortedKeys(subTree.Map) {
		parts = nodeToString(parts, key, subTree.Map[key], indent)
	}
	parts = append(parts, fmt.Sprintf("%s}", indentSpaces))
	return parts
}

// Map converts a Tree[T1] to a Tree[T2] by calling mapFn at every element.
func Map[T1, T2 any](tree1 *Tree[T1], mapFn func(Path, T1) T2) *Tree[T2] {
	if tree1.IsLeaf() {
		// Input tree1 is just a leaf node:
		return NewLeaf[T2](mapFn(nil, tree1.Value))
	}

	tree2 := New[T2]()
	for p, t1 := range tree1.Leaves() {
		err := tree2.Set(p, mapFn(p, t1))
		if err != nil {
			// Should never happen, since there can be no errors duplicating the structure of an existing valid tree.
			panic(err)
		}
	}
	return tree2
}

// Leaves returns an iterator that goes over all the leaf nodes of the Tree.
// The key is a Path, and value is T.
func (tree *Tree[T]) Leaves() iter.Seq2[Path, T] {
	return func(yield func(Path, T) bool) {
		recursiveLeaves(nil, tree, false, yield)
	}
}

// NumLeaves traverses the trees and returns the number of leaf nodes.
func (tree *Tree[T]) NumLeaves() int {
	var count int
	for _, _ = range tree.Leaves() {
		count++
	}
	return count
}

// OrderedLeaves returns an iterator that goes over all the leaf nodes of the Tree in alphabetical order of the
// tree nodes (depth-first).
//
// The key is a Path, and value is T.
func (tree *Tree[T]) OrderedLeaves() iter.Seq2[Path, T] {
	return func(yield func(Path, T) bool) {
		recursiveLeaves(nil, tree, true, yield)
	}
}

func recursiveLeaves[T any](treePath Path, node *Tree[T], ordered bool, yield func(Path, T) bool) bool {
	if node.IsLeaf() {
		return yield(slices.Clone(treePath), node.Value)
	}
	if ordered {
		// Extract keys and sort first.
		for _, key := range xslices.SortedKeys(node.Map) {
			subNode := node.Map[key]
			ok := recursiveLeaves[T](append(treePath, key), subNode, ordered, yield)
			if !ok {
				return false
			}
		}
	} else {
		// Usual range over map, non-deterministic.
		for key, subNode := range node.Map {
			ok := recursiveLeaves(append(treePath, key), subNode, ordered, yield)
			if !ok {
				return false
			}
		}
	}
	return true
}

// ValuesAsList extracts the leaf values of Tree into a list.
//
// It's generated in alphabetical order -- see OrderedLeaves to see or generate the order.
func ValuesAsList[T any](tree *Tree[T]) []T {
	results := make([]T, 0, tree.NumLeaves())
	for _, values := range tree.OrderedLeaves() {
		results = append(results, values)
	}
	return results
}

// FromValuesAndTree creates a Tree[T1] with the given values, but borrowing the structure from the given tree (but
// ignoring the tree's values).
func FromValuesAndTree[T1, T2 any](values []T1, tree *Tree[T2]) *Tree[T1] {
	numLeaves := tree.NumLeaves()
	if len(values) != numLeaves {
		exceptions.Panicf("%d values given, but the tree to be built has %d leaves.", len(values), numLeaves)
	}
	newTree := New[T1]()
	var idx int
	for treePath, _ := range tree.OrderedLeaves() {
		err := newTree.Set(treePath, values[idx])
		if err != nil {
			// Should never happen, since there can be no errors duplicating the structure of an existing valid tree.
			panic(err)
		}
		idx++
	}
	return newTree
}
