package main

import (
	"flag"
	hfd "github.com/gomlx/gemma/download/huggingface"
	"github.com/gomlx/gemma/samplers"
	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/xla"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/janpfeifer/must"
	"os"
	"path"
)

var (
	flagDataDir            = flag.String("data", "~/work/gemma", "Directory to cache downloaded and generated dataset files.")
	flagModelID            = flag.String("model", "google/gemma-2-2b-it", "HuggingFace Gemma model id")
	flagMaxGeneratedTokens = flag.Int("max_tokens", 1024, "Maximum number of tokens to generate.")
)

func BuildSampler() *samplers.Sampler {
	ctx := context.New()
	vocab := must.M1(hfd.Download(ctx, *flagModelID, os.Getenv("HF_TOKEN"), path.Join(*flagDataDir, "huggingface")))
	return must.M1(samplers.New(backends.New(), ctx, vocab, *flagMaxGeneratedTokens))
}
