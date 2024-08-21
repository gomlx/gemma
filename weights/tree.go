package weights

import (
	"fmt"
	"github.com/gomlx/gomlx/types/xslices"
	"strings"
)

// TreeNode is either a Value or a Map of its children -- but not both.
type TreeNode[T any] struct {
	// Value is set for leaf nodes only.
	Value T

	// Map is set for non-leaf nodes (and nil in leaf nodes).
	Map map[string]*TreeNode[T]
}

// Tree holds the root node for a Tree-like structure, as parallel to the PyTree structure.
// It also provides several convenience methods of access.
//
// T is the type of the leaf nodes.
type Tree[T any] struct {
	Root *TreeNode[T]
}

// TreePath is usually used as the path from the root node.
type TreePath []string

func NewTree[T any](root *TreeNode[T]) *Tree[T] {
	return &Tree[T]{Root: root}
}

func NewMapNode[T any]() *TreeNode[T] {
	return &TreeNode[T]{Map: make(map[string]*TreeNode[T])}
}

func NewLeaf[T any](value T) *TreeNode[T] {
	return &TreeNode[T]{Value: value}
}

// Insert value in treePath, creating intermediary nodes where needed.
func (tree *Tree[T]) Insert(treePath TreePath, value T) {
	node := tree.Root
	for len(treePath) > 0 {
		pathElement := treePath[0]
		treePath = treePath[1:]
		if pathElement == "" {
			// Skip empty path components
			continue
		}
		if node.Map == nil {
			node.Map = make(map[string]*TreeNode[T])
		}
		newNode := node.Map[pathElement]
		if newNode == nil {
			newNode = &TreeNode[T]{}
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

func nodeToString[T any](parts []string, name string, subTree *TreeNode[T], indent int) []string {
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

// EnumerateLeaves calls fn for all leaf nodes, providing the path from the root node to the leaf node,
// and the leaf node value (of type T).
func (tree *Tree[T]) EnumerateLeaves(fn func(treePath TreePath, leaf T)) {
	return
}
