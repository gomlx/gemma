package transformers

import (
	"fmt"
	"github.com/gomlx/gemma/trees"
	"github.com/gomlx/gomlx/types/tensors"
)

// Cache is a state cache of a (batch of) sequence being encoded/decoded.
//
// It has a fixed size (so typical cached values with be prefixed with the dimensions [BatchSize, Cache.Size],
// and current position (where each decode step is stored) is rotating: CurrentStep = (CurrentStep+1)%Cache.Length).
//
// It's stored as a trees.Tree[*tensor.Tensor].
//
// For the Gemma2 model, the first level of the tree being the layer names,
// and the second level hold the "keys" and "values" embedding caches for each transformer layer.
type Cache struct {
	// Config of the model.
	Config *Config

	// BatchSize for this cache.
	BatchSize int

	// Length (in number of steps) of the cache. The cache itself is rotating on this size.
	// It comes from config.MaxCacheLength.
	Length int

	// Data holds the cached data, organized as a trees.Tree[*tensors.Tensor].
	Data *trees.Tree[*tensors.Tensor]
}

func NewCache(config *Config, batchSize int) (*Cache, error) {
	c := &Cache{
		Config:    config,
		BatchSize: batchSize,
		Length:    config.MaxCacheLength,
		Data:      trees.New[*tensors.Tensor](),
	}

	for layerIdx := range config.NumLayers {
		treePath := []string{fmt.Sprintf("layer_%d", layerIdx)}
		err := createAttentionCache(c.Data, treePath, config.DType, batchSize, config.MaxCacheLength,
			config.NumKVHeads, config.HeadDim)
		if err != nil {
			return nil, err
		}
	}
	return c, nil
}
