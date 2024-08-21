package tree

import (
	"fmt"
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

// Insert value in treePath, creating intermediary nodes where needed.
func (tree *Tree[T]) Insert(treePath Path, value T) {
	node := tree.Root
	for len(treePath) > 0 {
		pathElement := treePath[0]
		treePath = treePath[1:]
		if pathElement == "" {
			// Skip empty path components
			continue
		}
		if node.Map == nil {
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
func Map[T1, T2 any](tree *Tree[T1], mapFn func(T1) T2) *Tree[T2] {
	return nil
}

// Leaves returns an iterator that goes over all the leaf nodes of the Tree.
// The key is a Path, and value is T.
func (tree *Tree[T]) Leaves() iter.Seq2[Path, T] {
	return func(yield func(Path, T) bool) {
		recursiveLeaves(nil, tree.Root, yield)
	}
}

func recursiveLeaves[T any](treePath Path, node *Node[T], yield func(Path, T) bool) bool {
	if node.Map == nil {
		return yield(slices.Clone(treePath), node.Value)
	}
	for key, subNode := range node.Map {
		ok := recursiveLeaves(append(treePath, key), subNode, yield)
		if !ok {
			return false
		}
	}
	return true
}
