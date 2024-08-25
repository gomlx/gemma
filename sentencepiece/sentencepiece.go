// Package sentencepiece fills some missing functionality from github.com/eliben/go-sentencepiece
//
// Hopefully it's temporary.
package sentencepiece

import (
	esentencepiece "github.com/eliben/go-sentencepiece"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/pkg/errors"
)

type Processor struct {
	*esentencepiece.Processor
}

func NewFromPath(vocabPath string) (*Processor, error) {
	proc, err := esentencepiece.NewProcessorFromPath(vocabPath)
	if err != nil {
		return nil, errors.Wrapf(err, "can't create sentencepiece")
	}
	return &Processor{
		Processor: proc,
	}, nil
}

type Token = esentencepiece.Token

// EncodeAsIds returns the text encoded into a sequence of ids.
// It implements sampler.Vocabulary.
func (p *Processor) EncodeAsIds(text string) []int {
	tokens := p.Processor.Encode(text)
	return xslices.Map(tokens, func(t Token) int { return t.ID })
}

// DecodeIds returns the text from a sequence of ids.
// It implements sampler.Vocabulary.
func (p *Processor) DecodeIds(ids []int) string {
	return p.Processor.Decode(ids)
}

// BeginningOfSentenceId returns the corresponding token, aka "bos".
//
// TODO: read from tokenizer model instead.
func (p *Processor) BeginningOfSentenceId() int {
	return 2
}

// EndOfSentenceId returns the corresponding token, aka "eos".
//
// TODO: read from tokenizer model instead.
func (p *Processor) EndOfSentenceId() int {
	return 1
}

// UnknownId returns the corresponding token, aka "unk".
//
// TODO: read from tokenizer model instead.
func (p *Processor) UnknownId() int {
	return 3
}

// PadId returns the corresponding token, aka "pad".
//
// TODO: read from tokenizer model instead.
func (p *Processor) PadId() int {
	return 0
}
