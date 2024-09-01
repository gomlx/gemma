package main

import (
	"flag"
	"github.com/gomlx/gemma/samplers"
	"github.com/gomlx/gemma/sentencepiece"
	weightsPkg "github.com/gomlx/gemma/weights"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/janpfeifer/must"
	"path"

	_ "github.com/gomlx/gomlx/backends/xla"
)

var (
	flagDataDir            = flag.String("data", "~/work/gemma", "Directory to cache downloaded and generated dataset files.")
	flagVocabFile          = flag.String("vocab", "weights/tokenizer.model", "Tokenizer file with vocabulary. Relative to --data directory.")
	flagGemmaWeights       = flag.String("gemma", "weights/gemma2-2b-it", "Gemma weights file. Relative to --data directory.")
	flagMaxGeneratedTokens = flag.Int("max_tokens", 512, "Maximum number of tokens to generate.")
)

// BuildTokenizer from flags --data and --vocab. Panics in case of error.
func BuildTokenizer() *sentencepiece.Processor {
	vocabPath := data.ReplaceTildeInDir(*flagVocabFile)
	if !path.IsAbs(vocabPath) {
		dataDir := data.ReplaceTildeInDir(*flagDataDir)
		vocabPath = path.Join(dataDir, vocabPath)
	}
	return must.M1(sentencepiece.NewFromPath(vocabPath))
}

// GemmaCheckpointDir returns the configured Gemma checkpoint.
func GemmaCheckpointDir() string {
	checkpointPath := data.ReplaceTildeInDir(*flagGemmaWeights)
	if !path.IsAbs(checkpointPath) {
		dataDir := data.ReplaceTildeInDir(*flagDataDir)
		checkpointPath = path.Join(dataDir, checkpointPath)
	}
	return checkpointPath
}

func BuildSampler() *samplers.Sampler {
	vocab := BuildTokenizer()
	weights := must.M1(weightsPkg.ReadConvertedWeights(GemmaCheckpointDir()))
	sampler := must.M1(samplers.New(backends.New(), vocab, weights, *flagMaxGeneratedTokens))
	return sampler
}
