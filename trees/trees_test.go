package trees

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"testing"
)

type expectedTreeValueType[T any] struct {
	p Path
	v T
}

func verifyTreeValues[T any](t *testing.T, tree *Tree[T], wantValues []expectedTreeValueType[T]) {
	count := 0
	for p, v := range tree.OrderedLeaves() {
		if count >= len(wantValues) {
			t.Fatalf("tree ranged over more leaves than the %d expected", len(wantValues))
		}
		require.Equalf(t, wantValues[count].p, p, "Unexpected path %q -- maybe out-of-order?", p)
		require.Equalf(t, wantValues[count].v, v, "Unexpected value for path %q", p)
		count++
	}
	if count != len(wantValues) {
		t.Fatalf("tree only ranged over %d leaf-values, but we expected %d values", count, len(wantValues))
	}
}

func createTestTree(t *testing.T) *Tree[int] {
	tree := New[int]()
	require.NoError(t, tree.Set([]string{"a"}, 1))
	require.NoError(t, tree.Set([]string{"b", "y"}, 3))
	require.NoError(t, tree.Set([]string{"b", "x"}, 2))
	return tree
}

func TestNewAndSet(t *testing.T) {
	tree := createTestTree(t)
	fmt.Printf("Tree:\n%v\n", tree)

	require.Equal(t, 1, tree.Map["a"].Value)
	require.Equal(t, 2, tree.Map["b"].Map["x"].Value)
	require.Equal(t, 3, tree.Map["b"].Map["y"].Value)

	err := tree.Set([]string{"b"}, 4)
	fmt.Printf("\texpected error trying to set non-leaf node: %v\n", err)
	require.ErrorContains(t, err, "trying to set the value to a non-leaf node")

	err = tree.Set([]string{"b", "x", "0"}, 5)
	fmt.Printf("\texpected error trying to use leaf node as structure: %v\n", err)
	require.ErrorContains(t, err, "trying to create a path using an existing leaf node")

	tree2 := NewLeaf(float32(7))
	fmt.Printf("Tree:\n%v\n", tree2)
	require.NoError(t, tree2.Set(nil, float32(11)))
	require.Equal(t, float32(11), tree2.Value)
}

func TestOrderedLeaves(t *testing.T) {
	tree := createTestTree(t)
	fmt.Printf("Tree:\n%v\n", tree)
	// Test OrderedLeaves traversal and that the contents of the tree match.
	verifyTreeValues(t, tree, []expectedTreeValueType[int]{
		{Path{"a"}, 1},
		{Path{"b", "x"}, 2},
		{Path{"b", "y"}, 3},
	})
}

func TestMap(t *testing.T) {
	tree := createTestTree(t)
	fmt.Printf("Tree:\n%v\n", tree)
	treeFloat := Map(tree, func(_ Path, v int) float32 { return float32(v) })
	verifyTreeValues(t, treeFloat, []expectedTreeValueType[float32]{
		{Path{"a"}, 1},
		{Path{"b", "x"}, 2},
		{Path{"b", "y"}, 3},
	})

	tree2 := NewLeaf(float32(7))
	fmt.Printf("Tree:\n%v\n", tree2)
	tree2Int := Map(tree2, func(_ Path, v float32) int { return int(v) })
	verifyTreeValues(t, tree2Int, []expectedTreeValueType[int]{
		{nil, 7},
	})
}

func TestValuesAsList(t *testing.T) {
	tree := createTestTree(t)
	fmt.Printf("Tree:\n%v\n", tree)
	require.Equal(t, []int{1, 2, 3}, ValuesAsList(tree))

	tree2 := NewLeaf(float32(7))
	fmt.Printf("Tree:\n%v\n", tree2)
	require.Equal(t, []float32{7}, ValuesAsList(tree2))
}

func TestFromValuesAndTree(t *testing.T) {
	tree := createTestTree(t)
	newValues := []float64{1.01, 2.02, 3.03}
	newTree := FromValuesAndTree(newValues, tree)
	fmt.Printf("New Tree:\n%v\n", newTree)
	verifyTreeValues(t, newTree, []expectedTreeValueType[float64]{
		{Path{"a"}, 1.01},
		{Path{"b", "x"}, 2.02},
		{Path{"b", "y"}, 3.03},
	})
}
