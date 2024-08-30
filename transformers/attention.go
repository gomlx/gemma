package transformers

import (
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

func Attention(ctx *context.Context, config *Config, x, positions *Node, cache *trees.Tree[*Node], attentionMask *Node) *Node {
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

	queryProjection.SetLogged("Attention::queryProjection")
	keyProjection.SetLogged("Attention::keyProjection")
	valueProjection.SetLogged("Attention::valueProjection")

	//for k, node := range cache.Map {
	//	node.Value.SetLogged(k)
	//}
	return x
}
