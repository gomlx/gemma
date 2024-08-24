// Package samplers uses a transformer model to generate senteces based on prompts.
package samplers

import (
	"fmt"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"slices"
)

type Vocabulary interface {
	Encode(text string) []int
	Decode([]int) string

	// The methods below define the special ids for the model.

	BeginningOfSentenceId() int
	EndOfSentenceId() int
	UnknownId() int
	PadId() int
}

// Sampler has a transformer (LLM) model and a vocabulary (sentencepiece) configured and generates
// sentences based on prompts.
type Sampler struct {
	Vocab Vocabulary
	Model any

	MaxGeneratedTokens int
}

// New creates a new sampler with the registered vocabulary and model.
func New(vocab Vocabulary, model any, maxGeneratedTokens int) *Sampler {
	return &Sampler{
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
	ids := xslices.Map(prompts, s.Vocab.Encode)
	lengths := xslices.Map(ids, func(seq []int) int { return len(seq) })
	maxInputLength := slices.Max(lengths)
	totalLength := maxInputLength + maxTokens
	input := s.createInputTensor(ids, totalLength)
	fmt.Printf("Ids: %v\n", input)
	return nil
}

// createInputTensor creates a tensor shaped int32[batchSize, totalLength+2] padded with the Vocab.PadId filled (left to right)
// with the given promptIds.
//
// It also adds a "bos" (beginning of sentence) token to each prompt.
func (s *Sampler) createInputTensor(promptIds [][]int, totalLength int) *tensors.Tensor {
	batchSize := len(promptIds)
	totalLength += 2 // To accommodate for "bos" and "eos".
	input := tensors.FromScalarAndDimensions(int32(s.Vocab.PadId()), batchSize, totalLength)
	bos := int32(s.Vocab.BeginningOfSentenceId())
	tensors.MutableFlatData(input, func(flat []int32) {
		for exampleIdx := range batchSize {
			exampleIds := flat[exampleIdx*totalLength : (exampleIdx+1)*totalLength]
			exampleIds[0] = bos
			for ii, value := range promptIds[exampleIdx] {
				exampleIds[1+ii] = int32(value)
			}
		}
	})
	return input
}
