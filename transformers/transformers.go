// Package transformers implements the various Gema models.
// It is based on https://github.com/google-deepmind/gemma/blob/main/gemma/transformer.py
package transformers

import (
	"fmt"
	"github.com/gomlx/gemma/trees"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// GemmaWithCache creates a forward path on a Gemma model for one decoding step,
// using the weights in Config to initialize the variables.
//
// It takes as input the current token to decode currentTokens (shape [batchSize, 1] in a sequence along
// with currentPosition (shape [batchSize, 1]) and the current cache of the key/values for each transformer
// layer (see Cache), whose elements are generally shaped [batchSize, MaxCacheLength,...].
//
// It updates the Cache with the new step in-place, and returns the logits (shape [batchSize, <num_tokens>])
// of the prediction of the next token.
func GemmaWithCache(ctx *context.Context, config *Config,
	currentTokens, currentPositions *Node, cache *trees.Tree[*Node], cacheAttentionMask *Node) *Node {
	layerIdx := -1 // One before next.
	nextLayerIdx := func() string {
		layerIdx++
		return fmt.Sprintf("%03d_", layerIdx)
	}

	// Embed.
	x := EmbedTokens(ctx.In(nextLayerIdx()+"_embedder"), config, currentTokens)

	// Run through numLayers blocks.
	for blockIdx := range config.NumLayers {
		blockName := fmt.Sprintf("layer_%d", blockIdx)
		blockCtx := ctx.In(nextLayerIdx() + blockName)
		blockCache := cache.Map[blockName]
		x = Block(blockCtx, config, blockName, x, currentPositions, blockCache, cacheAttentionMask)
		x.SetLogged(blockName)
		if true {
			break
		}
	}
	_ = x
	return nil
}

// EmbedTokens using weights in Config.
func EmbedTokens(ctx *context.Context, config *Config, currentTokens *Node) *Node {
	g := currentTokens.Graph()
	treePath := []string{"transformer", "embedder", "input_embedding"}
	var embedTableVar *context.Variable
	if config.Weights != nil {
		// Initialize variables from config weights:
		embedTableT, err := config.Weights.Get(treePath)
		if err != nil {
			panic(errors.Wrapf(err, "tranformer model missing embedding table weights in path %q", treePath))
		}
		embedTableVar = ctx.VariableWithValue("embeddings", embedTableT)
	} else {
		// Default variable initialization (likely read from checkpoint)
		embedTableVar = ctx.VariableWithShape("embeddings", shapes.Make(dtypes.BFloat16, config.VocabularySize, config.EmbedDim))
	}
	embedTable := embedTableVar.ValueGraph(g)
	embeddings := Gather(embedTable, currentTokens)
	embeddings = Mul(embeddings, Sqrt(Scalar(g, embeddings.DType(), float64(config.EmbedDim))))
	return embeddings
}

// Block implements one transformer block for the Gemma model.
func Block(ctx *context.Context, config *Config, blockName string, x, positions *Node, cache *trees.Tree[*Node], cacheAttentionMask *Node) *Node {
	blockWeights := config.Weights.Map["transformer"].Map[blockName]
	_ = blockWeights
	return x
}
