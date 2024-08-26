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

// EncodeAsIDs returns the text encoded into a sequence of ids.
// It implements sampler.Vocabulary.
func (p *Processor) EncodeAsIDs(text string) []int {
	tokens := p.Processor.Encode(text)
	return xslices.Map(tokens, func(t Token) int { return t.ID })
}

// DecodeIDs returns the text from a sequence of ids.
// It implements sampler.Vocabulary.
func (p *Processor) DecodeIDs(ids []int) string {
	return p.Processor.Decode(ids)
}
