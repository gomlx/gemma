package transformers

import (
	"github.com/gomlx/gemma/trees"
	"github.com/gomlx/gomlx/types/tensors"
)

// Cache is a state cache of a (batch of) sequence being encoded/decoded.
//
// It has a fixed size (so typical cached values with be prefixed with the dimensions [BatchSize, Cache.Size],
// and current position (where each decode step is stored) is rotating: CurrentStep = (CurrentStep+1)%Cache.Size).
//
// It's stored as a trees.Tree[*tensor.Tensor].
//
// For the Gemma2 model, the first level of the tree being the layer names,
// and the second level hold the "keys" and "values" embedding caches for each transformer layer.
type Cache struct {
	// Config of the model.
	Config *Config

	// Size (in number of steps) of the cache. The cache itself is rotating on this size.
	Size int

	// Data holds the cached data, organized as a trees.Tree[*tensors.Tensor].
	Data *trees.Tree[*tensors.Tensor]
}

func NewCache(config *Config, size int) *Cache {
	return &Cache{
		Config: config,
		Size:   size,
		Data:   trees.New(trees.NewMap[*tensors.Tensor]()),
	}
}
