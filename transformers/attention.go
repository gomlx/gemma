package transformers

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gemma/trees"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// createAttentionCache creates the attention cache for the attention layer under treePath.
func createAttentionCache(data *trees.Tree[*tensors.Tensor], treePath trees.Path, dtype dtypes.DType,
	batchSize, maxCacheLength, numHeads, headDim int) error {
	// Value cache:
	err := data.Set(append(treePath, "v"),
		tensors.FromShape(shapes.Make(dtype, batchSize, maxCacheLength, numHeads, headDim)))
	if err != nil {
		return errors.WithMessage(err, "in createAttentionCache()")
	}

	// Keys cache:
	err = data.Set(append(treePath, "k"),
		tensors.FromShape(shapes.Make(dtype, batchSize, maxCacheLength, numHeads, headDim)))
	if err != nil {
		return errors.WithMessage(err, "in createAttentionCache()")
	}

	// Index where to insert new values, in a rotating cache.
	err = data.Set(append(treePath, "end_index"), tensors.FromScalar(int32(0)))
	if err != nil {
		return errors.WithMessage(err, "in createAttentionCache()")
	}
	return nil
}

// Must panics if the error is not nil.
func Must(err error) {
	if err != nil {
		panic(err)
	}
}

// Must1 panics in case of error, otherwise returns the one return value.
func Must1[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

// Attention builds an attention layer, optionally using cache to store a limited amount of context.
//
//   - attentionIdx indexes attention configuration (in config) parameters, like config.AttentionTypes.
//   - x is the operand shaped [batchSize, sequenceLength, embedDim]. If using cache, typically the sequenceLength will be 1.
//   - positions are the positions of the sequence in x, shaped int32[batchSize, sequenceLength].
//   - cache: if set, x is only used for the current token (so sequenceLength will be 1), and the x's key and value projections
//     are set in the cache. After that, cache is used instead of x for the attention.
//   - attentionMask: shaped bool[batchSize, sequenceLength, sequenceLength] (if cache is nil) or bool[batchSize, sequenceLength==1, config.MaxCacheLength] if
//     cache is being used.
func Attention(ctx *context.Context, config *Config, attentionIdx int, x, positions *Node, cache *trees.Tree[*Node], attentionMask *Node) *Node {
	g := x.Graph()
	dtype := x.DType()

	// Calculates projections used in the attention.
	var queryProjection, keyProjection, valueProjection *Node

	// Glossary of keys for einsum:
	// B = batchSize
	// T = sequenceLength
	// D = config.EmbedDim
	// N = config.NumHeads
	// H = config.HeadDim
	// K = config.NumKVHeads
	if config.HuggingFaceVersion {
		// HuggingFace version has separate variables per projection.
		keyProjectionWeights := ctx.In("hf").
			VariableWithShape("k_proj", shapes.Make(dtype, config.NumKVHeads*config.HeadDim, config.EmbedDim)).
			ValueGraph(g)
		keyProjectionWeights = Reshape(keyProjectionWeights, config.NumKVHeads, config.HeadDim, config.EmbedDim)
		keyProjection = Einsum("BSD,KHD->BSKH", x, keyProjectionWeights)

		valueProjectionWeights := ctx.In("hf").
			VariableWithShape("v_proj", shapes.Make(dtype, config.NumKVHeads*config.HeadDim, config.EmbedDim)).
			ValueGraph(g)
		valueProjectionWeights = Reshape(valueProjectionWeights, config.NumKVHeads, config.HeadDim, config.EmbedDim)
		valueProjection = Einsum("BSD,KHD->BSKH", x, valueProjectionWeights)

		queryProjectionWeights := ctx.In("hf").
			VariableWithShape("q_proj", shapes.Make(dtype, config.NumHeads*config.HeadDim, config.EmbedDim)).
			ValueGraph(g)
		queryProjectionWeights = Reshape(queryProjectionWeights, config.NumHeads, config.HeadDim, config.EmbedDim)
		queryProjection = Einsum("BSD,NHD->BSNH", x, queryProjectionWeights)

	} else if config.UseQKV {
		// S = 3, one extra dimensions for query, key, value projections
		qkvProjections := KernelEinsum(ctx.In("qkv_einsum"), "BTD,SNDH->SBTNH", x,
			shapes.Make(dtype /* k, q, v = 3 */, 3, config.NumHeads, config.EmbedDim, config.HeadDim))
		queryProjection = Squeeze(Slice(qkvProjections, AxisElem(0)), 0)
		keyProjection = Squeeze(Slice(qkvProjections, AxisElem(1)), 0)
		valueProjection = Squeeze(Slice(qkvProjections, AxisElem(2)), 0)
	} else {
		queryProjection = KernelEinsum(ctx.In("q_einsum"), "BTD,NDH->BTNH", x,
			shapes.Make(dtype, config.NumHeads, config.EmbedDim, config.HeadDim))
		// C = 2, one dimension for key, the other for value.
		kvProjections := KernelEinsum(ctx.In("kv_einsum"), "BSD,CKDH->CBSKH", x,
			shapes.Make(dtype, 2, config.NumKVHeads, config.EmbedDim, config.HeadDim))
		keyProjection = Squeeze(Slice(kvProjections, AxisElem(0)), 0)
		valueProjection = Squeeze(Slice(kvProjections, AxisElem(1)), 0)
	}

	queryProjection = ApplyRotaryPositionEncoding(queryProjection, positions, RoPEDefaultMaxWaveLength)
	queryScaled := MulScalar(queryProjection, config.QueryPreAttentionScalar())
	keyProjection = ApplyRotaryPositionEncoding(keyProjection, positions, RoPEDefaultMaxWaveLength)

	// If cache is set, update it with the projections of the slice of the sequence given, and then take the
	// projections of the whole cache.
	if cache != nil {
		// Insert calculated projections in cache: cached projections are shaped [batchSize, maxCacheLength, numHeads, headDim]
		endIndex, err := cache.Get("end_index")
		if err != nil {
			panic(err)
		}
		zeroIdx := ScalarZero(g, dtypes.Int32)
		cacheSequencePosition := Mod(endIndex, Scalar(g, endIndex.DType(), config.MaxCacheLength))
		updateSliceIndices := []*Node{zeroIdx, cacheSequencePosition, zeroIdx, zeroIdx}

		valueProjection = DynamicUpdateSlice(Must1(cache.Get("v")), valueProjection, updateSliceIndices)
		keyProjection = DynamicUpdateSlice(Must1(cache.Get("k")), keyProjection, updateSliceIndices)
		Must(cache.Set(trees.Path{"v"}, valueProjection))
		Must(cache.Set(trees.Path{"k"}, keyProjection))
		// Bump end_index the length of tokens provided at this step: typically, this will be only 1. If > 1
		// this will probably not work if the cache wraps around.
		Must(cache.Set(trees.Path{"end_index"}, AddScalar(endIndex, positions.Shape().Dim(-1))))
	}

	batchSize := queryScaled.Shape().Dim(0)               // B
	seqLength := queryScaled.Shape().Dim(1)               // T
	numQueryHeads := queryScaled.Shape().Dim(2)           // N
	headDim := queryScaled.Shape().Dim(3)                 // H
	numKVHeads := config.NumKVHeads                       // K
	attentionTargetLength := keyProjection.Shape().Dim(1) // S = config.MaxCacheLength if cache != nil, or seqLength.

	var logits *Node
	if config.UseGroupQueryAttention {
		// There are fewer key (and value) projections than query projections,
		// reshape matrices accordingly and adjust Einsum.
		queryPerKVHeads := numQueryHeads / numKVHeads // G
		queryScaled = Reshape(queryScaled, batchSize, seqLength, numKVHeads, queryPerKVHeads, headDim)
		logits = Einsum("BTKGH,BSKH->BTKGS", queryScaled, keyProjection)
		logits = Reshape(logits, batchSize, seqLength, numQueryHeads, attentionTargetLength)
	} else {
		// Same number of query/key projections.
		// N = numQueryHeads == numKVHeads.
		logits = Einsum("BTNH,BSNH->BTNS", queryScaled, keyProjection)
	}
	logits.AssertDims(batchSize, seqLength, numQueryHeads, config.MaxCacheLength)
	logits = SoftCap(logits, config.AttentionLogitsSoftCap) // No-op if config.AttentionLogitsSoftCap is 0.

	if config.AttentionTypes[attentionIdx] == AttentionTypeLocalSliding {
		// Create a sliding mask: a mask that has a band (2*config.SlidingWindowSize) around the diagonal.
		// Issue: this will not work when using cache, and the cache loops around its config.MaxCacheLength, since
		//        the sliding mask "band" won't wrap around.
		if config.SlidingWindowSize <= 0 {
			exceptions.Panicf("Config.SlidingWindowSize must be set for AttentionTypeLocalSliding")
		}
		allOnes := OnesLike(attentionMask)
		slidingMask := And(
			TakeUpperTriangular(allOnes, 1-config.SlidingWindowSize),
			TakeLowerTriangular(allOnes, config.SlidingWindowSize-1),
		)
		attentionMask = And(attentionMask, slidingMask)
	}

	// Calculate attention weights.
	const logitsMask = -2.3819763e38
	paddedLogits := Where(
		BroadcastToShape(ExpandDims(attentionMask, -2), logits.Shape()),
		logits,
		Scalar(g, logits.DType(), logitsMask),
	)
	attentionWeights := Softmax(paddedLogits, -1)

	// Weighted sum of the values:
	var encoded *Node
	if config.UseGroupQueryAttention {
		// Reshape matrices to enable Einsums over groups of queries.
		queryPerKVHeads := numQueryHeads / numKVHeads // G
		attentionWeights = Reshape(attentionWeights, batchSize, seqLength, numKVHeads, queryPerKVHeads, attentionTargetLength)
		encoded = Einsum("BTKGS,BSKH->BTKGH", attentionWeights, valueProjection)
		encoded = Reshape(encoded, batchSize, seqLength, numQueryHeads, headDim)
	} else {
		// Plain attention: same number of query, keys and values projections.
		encoded = Einsum("BTNS,BSNH->BTNH", attentionWeights, valueProjection)
		encoded.AssertDims(batchSize, seqLength, numQueryHeads, headDim)
	}

	// Finally, a linear transformation on the result, merging all the heads.
	var output *Node
	if config.HuggingFaceVersion {
		outputProjectionWeights := ctx.In("hf").
			VariableWithShape("o_proj", shapes.Make(dtype, config.EmbedDim, config.NumHeads*config.HeadDim)).
			ValueGraph(g)
		outputProjectionWeights = Reshape(outputProjectionWeights, config.EmbedDim, numQueryHeads, config.HeadDim)
		output = Einsum("BTNH,DNH->BTD", encoded, outputProjectionWeights)

	} else {
		output = KernelEinsum(ctx.In("attn_vec_einsum"), "BTNH,NHD->BTD",
			encoded,
			shapes.Make(encoded.DType(), numQueryHeads, config.HeadDim, config.EmbedDim))
	}
	return output
}
