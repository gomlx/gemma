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
	batchSize := currentTokens.Shape().Dim(0)
	seqLength := currentTokens.Shape().Dim(1)

	// Embed.
	x := EmbedTokens(ctx.In("embedder"), config, currentTokens)

	// Run through numLayers blocks.
	for blockIdx := range config.NumLayers {
		blockName := fmt.Sprintf("layer_%d", blockIdx)
		blockCtx := ctx.In(blockName)
		blockCache := cache.Map[blockName]
		x = Block(blockCtx, config, blockIdx, x, currentPositions, blockCache, cacheAttentionMask)
		//x.SetLogged(fmt.Sprintf("GemmaWithCache::x(%s)", blockName))
		x = Identity(x)
	}

	x = RMSNorm(ctx.In("final_norm"), x)
	logits := DecodeTokens(ctx.Reuse().In("embedder"), config, x)
	logits = SoftCap(logits, config.FinalLogitSoftCap)
	logits.AssertDims(batchSize, seqLength, config.VocabularySize)
	return logits
}

// EmbedTokens using weights in Config.
// Input: currentTokens: [batchSize, sequenceLength]
// Output: embeddings: [batchSize, sequenceLength, config.EmbedDim]
func EmbedTokens(ctx *context.Context, config *Config, currentTokens *Node) *Node {
	g := currentTokens.Graph()
	embedTableVar := ctx.VariableWithShape("input_embedding", shapes.Make(dtypes.BFloat16, config.VocabularySize, config.EmbedDim))
	embeddings := Gather(embedTableVar.ValueGraph(g), ExpandDims(currentTokens, -1))
	embeddings = Mul(embeddings, Sqrt(Scalar(g, embeddings.DType(), config.EmbedDim)))
	return embeddings
}

// DecodeTokens use the same table as EmbedTokens to convert embedding back to the tokens -- or to token logits.
// Input: current embeddings: [batchSize, sequenceLength, embedDim]
// Output: logits for each token: [batchSize, sequenceLength, vocabularySize]
func DecodeTokens(ctx *context.Context, config *Config, x *Node) *Node {
	g := x.Graph()
	embedTableVar := ctx.VariableWithShape("input_embedding", shapes.Make(dtypes.BFloat16, config.VocabularySize, config.EmbedDim))
	embedTable := embedTableVar.ValueGraph(g)
	return DotGeneral(x, []int{-1}, nil, embedTable, []int{-1}, nil)
}

// Block implements one transformer block for the Gemma model. x is shaped [batchSize, sequenceLength], and if
// using cache (cache != nil), x will only contain the current token, shaped [batchSize, 1].
//
// The attentionIdx indexes attention configuration (in config) parameters, like config.AttentionTypes.
//
// If cache is given, attentionMask is relative to the cache. Otherwise, attentionMask is relative to the operand x.
func Block(ctx *context.Context, config *Config, attentionIdx int, x, positions *Node, cache *trees.Tree[*Node], attentionMask *Node) *Node {
	normalizedX := RMSNorm(ctx.In("pre_attention_norm"), x)

	// Attention
	attentionOut := Attention(ctx.In("attn"), config, attentionIdx, normalizedX, positions, cache, attentionMask)
	if config.UsePostAttentionNorm {
		attentionOut = RMSNorm(ctx.In("post_attention_norm"), attentionOut)
	}

	// Residual (or skip) connection.
	attentionOut = Add(attentionOut, x)

	// GatedFeedForward ("ffw") layer: 2 layers, with a gate.
	output := RMSNorm(ctx.In("pre_ffw_norm"), attentionOut)
	if config.HuggingFaceVersion {
		output = HuggingFaceGatedFeedForward(ctx.In("mlp"), output, config.HiddenDim, config.TransposeGatingEinsum)
	} else {
		output = GatedFeedForward(ctx.In("mlp"), output, config.HiddenDim, config.TransposeGatingEinsum)
	}
	if config.UsePostFFWNorm {
		output = RMSNorm(ctx.In("post_ffw_norm"), output)
	}

	// Residual to attentionOut.
	output = Add(output, attentionOut)
	return output
}
