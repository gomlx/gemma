// Package samplers uses a transformer model to generate senteces based on prompts.
package samplers

import (
	"fmt"
	"github.com/gomlx/gemma/transformers"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"slices"
)

type Vocabulary interface {
	EncodeAsIds(text string) []int
	DecodeIds([]int) string

	// The methods below define the special ids for the model.

	BeginningOfSentenceId() int
	EndOfSentenceId() int
	UnknownId() int
	PadId() int
}

// Sampler has a transformer (LLM) model and a vocabulary (sentencepiece) configured and generates
// sentences based on prompts.
type Sampler struct {
	Backend backends.Backend
	Vocab   Vocabulary
	Model   any

	MaxGeneratedTokens int
}

// New creates a new sampler with the registered vocabulary and model.
func New(backend backends.Backend, vocab Vocabulary, model any, maxGeneratedTokens int) *Sampler {
	return &Sampler{
		Backend:            backend,
		Vocab:              vocab,
		Model:              model,
		MaxGeneratedTokens: maxGeneratedTokens,
	}
}

// Sample the continuation from the given prompts.
func (s *Sampler) Sample(prompts []string) []string {
	return s.SampleMaxTokens(prompts, s.MaxGeneratedTokens)
}

// SampleMaxTokens is like Sample, but instead of using the default MaxGenerateTokens, uses the given maxTokens instead.
func (s *Sampler) SampleMaxTokens(prompts []string, maxTokens int) []string {
	ids := xslices.Map(prompts, s.Vocab.EncodeAsIds)
	lengths := xslices.Map(ids, func(seq []int) int { return len(seq) })
	maxInputLength := slices.Max(lengths)
	totalLength := maxInputLength + maxTokens
	inputIds, inputMask := s.createInputTensors(ids, totalLength)
	positions := transformers.BuildPositionsFromMask(s.Backend, inputMask)
	_ = inputIds
	fmt.Printf("positions=%v\n", positions.Value())
	return nil
}

// createInputTensors creates a tensor shaped int32[batchSize, totalLength+2] padded with the Vocab.PadId filled (left to right)
// with the given promptIds.
//
// It also returns the mask, that is set to true where it is not padding.
//
// It also adds a "bos" (beginning of sentence) token to each prompt.
func (s *Sampler) createInputTensors(promptIds [][]int, totalLength int) (inputIds, inputMask *tensors.Tensor) {
	batchSize := len(promptIds)
	totalLength += 2 // To accommodate for "bos" and "eos".
	inputIds = tensors.FromScalarAndDimensions(int32(s.Vocab.PadId()), batchSize, totalLength)
	inputMask = tensors.FromShape(shapes.Make(dtypes.Bool, batchSize, totalLength))
	bos := int32(s.Vocab.BeginningOfSentenceId())
	pad := s.Vocab.PadId()
	tensors.MutableFlatData(inputIds, func(flatIds []int32) {
		tensors.MutableFlatData(inputMask, func(flatMask []bool) {
			for exampleIdx := range batchSize {
				exampleIds := flatIds[exampleIdx*totalLength : (exampleIdx+1)*totalLength]
				exampleMask := flatMask[exampleIdx*totalLength : (exampleIdx+1)*totalLength]
				exampleIds[0] = bos
				exampleMask[0] = true
				for ii, value := range promptIds[exampleIdx] {
					exampleIds[1+ii] = int32(value)
					exampleMask[1+ii] = value != pad
				}
			}
		})
	})
	return
}
