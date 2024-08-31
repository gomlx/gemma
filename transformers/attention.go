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
	panic(err)
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
// The attentionIdx indexes attention configuration (in config) parameters, like config.AttentionTypes.
func Attention(ctx *context.Context, config *Config, attentionIdx int, x, positions *Node, cache *trees.Tree[*Node], attentionMask *Node) *Node {
	g := x.Graph()

	// Calculates projections used in the attention.
	var queryProjection, keyProjection, valueProjection *Node
	if config.UseQKV {
		// B = batchSize
		// T = sequenceLength
		// D = config.EmbedDim
		// S = 3, one extra dimensions for query, key, value projections
		// N = config.NumHeads
		// H = config.HeadDim
		qkvProjections := KernelEinsum(ctx.In("qkv_einsum"), "BTD,SNDH->SBTNH", x,
			shapes.Make(x.DType() /* k, q, v = 3 */, 3, config.NumHeads, config.EmbedDim, config.HeadDim))
		queryProjection = Squeeze(Slice(qkvProjections, AxisElem(0)), 0)
		keyProjection = Squeeze(Slice(qkvProjections, AxisElem(1)), 0)
		valueProjection = Squeeze(Slice(qkvProjections, AxisElem(2)), 0)
	} else {
		queryProjection = KernelEinsum(ctx.In("q_einsum"), "BTD,NDH->BTNH", x,
			shapes.Make(x.DType(), config.NumHeads, config.EmbedDim, config.HeadDim))
		// C = 2, one dimension for key, the other for value.
		// K =
		kvProjections := KernelEinsum(ctx.In("kv_einsum"), "BSD,CKDH->CBSKH", x,
			shapes.Make(x.DType(), 2, config.NumKVHeads, config.EmbedDim, config.HeadDim))
		keyProjection = Squeeze(Slice(kvProjections, AxisElem(0)), 0)
		valueProjection = Squeeze(Slice(kvProjections, AxisElem(1)), 0)
	}

	queryProjection = ApplyRotaryPositionEncoding(queryProjection, positions, RoPEDefaultMaxWaveLength)
	queryScaled := MulScalar(queryProjection, config.QueryPreAttentionScalar())
	keyProjection = ApplyRotaryPositionEncoding(keyProjection, positions, RoPEDefaultMaxWaveLength)

	queryScaled.SetLogged("Attention::queryScaledProjection(pre-cache)")
	keyProjection.SetLogged("Attention::keyProjection(pre-cache)")
	valueProjection.SetLogged("Attention::valueProjection(pre-cache)")

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
	}

	batchSize := queryScaled.Shape().Dim(0) // B
	seqLength := queryScaled.Shape().Dim(1) // T
	numQueryHeads := queryScaled.Shape().Dim(2)
	headDim := queryScaled.Shape().Dim(3) // H

	var logits *Node
	if config.UseGroupQueryAttention {
		// There are fewer key (and value) projections than query projections,
		// reshape matrices accordingly and adjust Einsum.
		numKVHeads := config.NumKVHeads               // K
		queryPerKVHeads := numQueryHeads / numKVHeads // G
		// S = config.MaxCacheLength
		queryScaled = Reshape(queryScaled, batchSize, seqLength, numKVHeads, queryPerKVHeads, headDim)
		logits = Einsum("BTKGH,BSKH->BTKGS", queryScaled, keyProjection)
		logits = Reshape(logits, batchSize, seqLength, numQueryHeads, config.MaxCacheLength)
	} else {
		// Same number of query/key projections.
		// N = numQueryHeads == numKVHeads.
		logits = Einsum("BTNH,BSNH->BTNS", queryScaled, keyProjection)
	}

	if config.AttentionLogitsSoftCap > 0 {
		logits = Tanh(DivScalar(logits, config.AttentionLogitsSoftCap))
		logits = MulScalar(logits, config.AttentionLogitsSoftCap)
	}

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
	//for k, node := range cache.Map {
	//	node.Value.SetLogged(k)
	//}
	return x
}
