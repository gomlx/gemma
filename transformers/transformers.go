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

	// Embed.
	x := EmbedTokens(ctx.In("embedder"), config, currentTokens)

	// Run through numLayers blocks.
	for blockIdx := range config.NumLayers {
		blockName := fmt.Sprintf("layer_%d", blockIdx)
		blockCtx := ctx.In(blockName)
		blockCache := cache.Map[blockName]
		x = Block(blockCtx, config, blockIdx, x, currentPositions, blockCache, cacheAttentionMask)
		if true {
			break
		}
	}
	_ = x
	return nil
}

func debugContext(ctx *context.Context) {
	fmt.Printf("ctx: scope=%q\n", ctx.Scope())
	ctx.EnumerateVariablesInScope(func(v *context.Variable) {
		fmt.Printf("\t> %q\n", v.ScopeAndName())
	})
}

// EmbedTokens using weights in Config.
// Input: currentTokens: [batchSize, sequenceSize]
// Output: embeddings: [batchSize, sequenceSize, config.EmbedDim]
func EmbedTokens(ctx *context.Context, config *Config, currentTokens *Node) *Node {
	g := currentTokens.Graph()
	embedTableVar := ctx.VariableWithShape("input_embedding", shapes.Make(dtypes.BFloat16, config.VocabularySize, config.EmbedDim))
	embeddings := Gather(embedTableVar.ValueGraph(g), ExpandDims(currentTokens, -1))
	embeddings = Mul(embeddings, Sqrt(Scalar(g, embeddings.DType(), float64(config.EmbedDim))))
	return embeddings
}

// Block implements one transformer block for the Gemma model. x is shaped [batchSize, sequenceSize], and if
// using cache (cache != nil), x will only contain the current token, shaped [batchSize, 1].
//
// The attentionIdx indexes attention configuration (in config) parameters, like config.AttentionTypes.
//
// If cache is given, attentionMask is relative to the cache. Otherwise, attentionMask is relative to the operand x.
func Block(ctx *context.Context, config *Config, attentionIdx int, x, positions *Node, cache *trees.Tree[*Node], attentionMask *Node) *Node {
	x.SetLogged("block input")
	normalizedX := RMSNorm(ctx.In("pre_attention_norm"), x)
	normalizedX.SetLogged("Block::pre_attention_norm")

	// Attention
	attentionOut := Attention(ctx.In("attn"), config, attentionIdx, normalizedX, positions, cache, attentionMask)
	if true {
		return nil
	}
	attentionOut.SetLogged("Block::attentionOut")
	if config.UsePostAttentionNorm {
		attentionOut = RMSNorm(ctx.In("post_attention_norm"), attentionOut)
	}

	// Residual (or skip) connection.
	attentionOut = Add(attentionOut, x)

	// One feed-forward ("ffw") layer.
	output := RMSNorm(ctx.In("pre_ffw_norm"), attentionOut)

	//...ffw
	if config.UsePostFFWNorm {
		output = RMSNorm(ctx.In("post_ffw_norm"), output)
	}
	return output
}
