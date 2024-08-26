package trees

import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/xslices"
	"golang.org/x/exp/slices"
	"iter"
	"strings"
)

// Node is either a Value or a Map of its children -- but not both.
type Node[T any] struct {
	// Value is set for leaf nodes only.
	Value T

	// Map is set for non-leaf nodes (and nil in leaf nodes).
	Map map[string]*Node[T]
}

func (n *Node[T]) IsLeaf() bool { return n.Map == nil }

// Tree holds the root node for a Tree-like structure, as parallel to the PyTree structure.
// It also provides several convenience methods of access.
//
// T is the type of the leaf nodes.
type Tree[T any] struct {
	Root *Node[T]
}

// Path is usually used as the path from the root node.
type Path []string

func New[T any](root *Node[T]) *Tree[T] {
	return &Tree[T]{Root: root}
}

func NewMap[T any]() *Node[T] {
	return &Node[T]{Map: make(map[string]*Node[T])}
}

func NewLeaf[T any](value T) *Node[T] {
	return &Node[T]{Value: value}
}

// Set value in treePath, populating intermediary nodes where needed.
func (tree *Tree[T]) Set(treePath Path, value T) {
	node := tree.Root
	for len(treePath) > 0 {
		pathElement := treePath[0]
		treePath = treePath[1:]
		if pathElement == "" {
			// Skip empty path components
			continue
		}
		if node.IsLeaf() {
			node.Map = make(map[string]*Node[T])
		}
		newNode := node.Map[pathElement]
		if newNode == nil {
			newNode = &Node[T]{}
			node.Map[pathElement] = newNode
		}
		node = newNode
	}
	node.Value = value
}

// String implements fmt.String
func (tree *Tree[T]) String() string {
	var parts []string
	parts = nodeToString(parts, "/", tree.Root, 0)
	return strings.Join(parts, "\n") + "\n"
}

func nodeToString[T any](parts []string, name string, subTree *Node[T], indent int) []string {
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
	if tree1.Root.IsLeaf() {
		return New(NewLeaf(mapFn(nil, tree1.Root.Value)))
	}
	tree2 := New(NewMap[T2]())
	for p, t1 := range tree1.Leaves() {
		tree2.Set(p, mapFn(p, t1))
	}
	return tree2
}

// Leaves returns an iterator that goes over all the leaf nodes of the Tree.
// The key is a Path, and value is T.
func (tree *Tree[T]) Leaves() iter.Seq2[Path, T] {
	return func(yield func(Path, T) bool) {
		recursiveLeaves(nil, tree.Root, false, yield)
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
		recursiveLeaves(nil, tree.Root, true, yield)
	}
}

func recursiveLeaves[T any](treePath Path, node *Node[T], ordered bool, yield func(Path, T) bool) bool {
	if node.Map == nil {
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
	newTree := &Tree[T1]{Root: &Node[T1]{}}
	var idx int
	for treePath, _ := range tree.OrderedLeaves() {
		newTree.Set(treePath, values[idx])
		idx++
	}
	return newTree
}
