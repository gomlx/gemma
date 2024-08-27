package transformers

import (
	"github.com/gomlx/gemma/trees"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
)

// createAttentionCache creates the attention cache for the attention layer under treePath.
func createAttentionCache(data *trees.Tree[*tensors.Tensor], treePath trees.Path, dtype dtypes.DType,
	batchSize, maxCacheLength, numHeads, headDim int) {
	// Value cache:
	must.M(data.Set(append(treePath, "v"),
		tensors.FromShape(shapes.Make(dtype, batchSize, maxCacheLength, numHeads, headDim))))

	// Keys cache:
	must.M(data.Set(append(treePath, "k"),
		tensors.FromShape(shapes.Make(dtype, batchSize, maxCacheLength, numHeads, headDim))))

	// Index where to insert new values, in a rotating cache.
	must.M(data.Set(append(treePath, "end_index"), tensors.FromScalar(int32(0))))
}
