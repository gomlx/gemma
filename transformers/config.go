package transformers

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"math"
)

type GemmaType int

const (
	UnknownGemmaType GemmaType = iota
	Gemma_2B
	Gemma_7B
	Gemma2_2B
	Gemma2_9B
	Gemma2_27B
)

//go:generate enumer -type=GemmaType -transform=snake -values -text -json -yaml config.go

var numLayersToGemmaClass = map[int]GemmaType{
	18: Gemma_2B,
	28: Gemma_7B,
	26: Gemma2_2B,
	42: Gemma2_9B,
	46: Gemma2_27B,
}

type AttentionType int

//go:generate enumer -type=AttentionType -trimprefix=AttentionType -transform=snake -values -text -json -yaml config.go

const (
	AttentionTypeUnknown AttentionType = iota
	AttentionTypeGlobal
	AttentionTypeLocalSliding
)

// QueryPreAttentionNormalisationType defines how to normalize query before attention.
type QueryPreAttentionNormalisationType int

//go:generate enumer -type=QueryPreAttentionNormalisationType -trimprefix=QueryNormType -transform=snake -values -text -json -yaml config.go

const (
	// QueryNormTypeByOneOverSqrtHeadDim indicates whether to scale the query by 1/sqrt(head_dim)
	QueryNormTypeByOneOverSqrtHeadDim QueryPreAttentionNormalisationType = iota

	// QueryNormTypeByEmbedDimDivNumHeads indicates whether to scale the query by `embed_dim // num_heads`
	QueryNormTypeByEmbedDimDivNumHeads

	// QueryNormTypeByOneOverSqrtEmbedDimDivNumHeads indicates whether to scale the query by `1/sqrt(embed_dim // num_heads)`
	QueryNormTypeByOneOverSqrtEmbedDimDivNumHeads
)

// Config Gemma transformer model.
type Config struct {
	Type                GemmaType
	DType               dtypes.DType
	VocabularySize      int
	NumLayers, NumEmbed int

	// HuggingFaceVersion has different shapes for some of the variables.
	HuggingFaceVersion bool

	// EmbedDim is also called "features" in the original code. It is the representation size (last dimension) of the output of the attention layers.
	EmbedDim                             int
	NumHeads, HeadDim                    int
	HiddenDim                            int
	NumKVHeads                           int
	FinalLogitSoftCap                    float64
	UseQKV, UseGroupQueryAttention       bool
	UsePostAttentionNorm, UsePostFFWNorm bool

	AttentionTypes        []AttentionType
	MaxCacheLength        int
	QueryPreAttentionNorm QueryPreAttentionNormalisationType

	// AttentionLogitsSoftCap limits the attention logits (logits = AttentionLogitsSoftCap * tanh(logits/AttentionLogitsSoftCap)).
	// Enabled if > 0.
	AttentionLogitsSoftCap float64
	SlidingWindowSize      int
	TransposeGatingEinsum  bool
}

// NewConfigFromContext creates a transformers config model, based on the structure of the variables in the given context -- the scope
// has to be set directly to the model variables.
func NewConfigFromContext(ctx *context.Context) (*Config, error) {
	c := &Config{
		MaxCacheLength:        1024,
		QueryPreAttentionNorm: QueryNormTypeByOneOverSqrtHeadDim,
	}

	embedTable := ctx.In("embedder").GetVariable("input_embedding")
	if embedTable == nil {
		return nil, errors.New("context given doesn't have an embedding table defined in \"embedder/input_embedding\"")
	}

	c.DType = embedTable.Shape().DType
	c.VocabularySize = embedTable.Shape().Dim(0)
	c.HuggingFaceVersion = c.VocabularySize == 256000 // Kaggle version is 256128.

	// Find number of layers.
	for {
		v := ctx.Inf("layer_%d", c.NumLayers).In("pre_attention_norm").GetVariable("scale")
		if v == nil {
			break
		}
		c.NumLayers++
	}
	if t, found := numLayersToGemmaClass[c.NumLayers]; found {
		c.Type = t
	}

	switch c.Type {
	case Gemma2_2B:
		c.setGemma2_2B()
	default:
		return nil, errors.Errorf("unknown or not implemented for Gemma model type %q", c.Type)
	}

	c.UseQKV = c.NumKVHeads == c.NumHeads
	c.UseGroupQueryAttention = (c.NumKVHeads != c.NumHeads) && c.NumKVHeads > 1
	return c, nil
}

func (c *Config) setGemma2_2B() {
	c.NumLayers = 26
	c.NumEmbed = 256128
	c.EmbedDim = 2304
	c.HiddenDim = 9216
	c.NumHeads = 8
	c.HeadDim = 256
	c.NumKVHeads = 4
	c.FinalLogitSoftCap = 30.0
	c.AttentionTypes = nil
	for _ = range c.NumLayers / 2 {
		c.AttentionTypes = append(c.AttentionTypes, []AttentionType{AttentionTypeLocalSliding, AttentionTypeGlobal}...)
	}
	c.UsePostAttentionNorm = true
	c.UsePostFFWNorm = true
	c.QueryPreAttentionNorm = QueryNormTypeByOneOverSqrtHeadDim
	c.AttentionLogitsSoftCap = 50.0
	c.SlidingWindowSize = 4096
}

// QueryPreAttentionScalar is a multiplier to the query projections.
func (c *Config) QueryPreAttentionScalar() float64 {
	switch c.QueryPreAttentionNorm {
	case QueryNormTypeByEmbedDimDivNumHeads:
		return float64(c.EmbedDim / c.NumHeads)
	case QueryNormTypeByOneOverSqrtEmbedDimDivNumHeads:
		return 1.0 / math.Sqrt(float64(c.EmbedDim/c.NumHeads))
	case QueryNormTypeByOneOverSqrtHeadDim:
		return 1.0 / math.Sqrt(float64(c.HeadDim))
	default:
		exceptions.Panicf("invalid value of QueryPreAttentionNorm = %d, expected one of the valid enum values", c.QueryPreAttentionNorm)
		panic(nil) // Quiet lint.
	}
}
