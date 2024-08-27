package transformers

import (
	"github.com/gomlx/gemma/trees"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"maps"
	"strings"
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
	Type                                 GemmaType
	DType                                dtypes.DType
	NumLayers                            int
	NumEmbed, EmbedDim                   int
	NumHeads, HeadDim                    int
	HiddenDim                            int
	NumKVHeads                           int
	FinalLogitSoftCap                    float64
	UsePostAttentionNorm, UsePostFFWNorm bool

	AttentionTypes        []AttentionType
	MaxCacheLength        int
	QueryPreAttentionNorm QueryPreAttentionNormalisationType

	AttentionLogitsSoftCap float64
	SlidingWindowSize      int
	TransposeGatingEinsum  bool
}

// NewConfigFromWeights creates a transformers config model, based on the structure of the loaded model weights.
func NewConfigFromWeights(weights *trees.Tree[*tensors.Tensor]) (*Config, error) {
	c := &Config{
		MaxCacheLength:        1024,
		QueryPreAttentionNorm: QueryNormTypeByOneOverSqrtHeadDim,
	}

	for _, w := range weights.Leaves() {
		if c.DType == dtypes.InvalidDType {
			c.DType = w.DType()
			continue
		}
		if c.DType != w.DType() {
			return nil, errors.New("can't infer dtype, different parameters have different dtypes")
		}
	}

	// Find number of layers:
	for key := range maps.Keys(weights.Root.Map["transformer"].Map) {
		if strings.Index(key, "layer") != -1 {
			c.NumLayers++
		}
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
