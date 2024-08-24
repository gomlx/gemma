// Package sentencepiece fills some missing functionality from github.com/eliben/go-sentencepiece
//
// Hopefully it's temporary.
package sentencepiece

import (
	esentencepiece "github.com/eliben/go-sentencepiece"
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

// BeginningOfSentence returns the corresponding token, aka "bos".
func (p *Processor) BeginningOfSentence() Token {
	return Token{
		ID: 2,
	}
}

// EndOfSentence returns the corresponding token, aka "eos".
func (p *Processor) EndOfSentence() Token {
	return Token{
		ID: 1,
	}
}

// Unknown returns the corresponding token, aka "unk".
func (p *Processor) Unknown() Token {
	return Token{
		ID: 3,
	}
}

// Pad returns the corresponding token, aka "pad".
func (p *Processor) Pad() Token {
	return Token{
		ID: 0,
	}
}
