
<img align="right" src="https://github.com/gomlx/gomlx/raw/main/docs/gomlx_gopher.jpg" alt="GoMLX Gopher" width="128px"/>

# GoMLX Gemma

GoMLX (for Go) port of Google Deepmind's Gemma GenAI/LLM model.

## 📖 About GoMLX Gemma

An implementation of [Google DeepMind](deepmind.google)'s [Gemma model](https://github.com/google-deepmind/gemma?tab=readme-ov-file)
using [GoMLX, a Machine Learning framework for Go](https://github.com/gomlx/gomlx).

It is very "_fresh from the oven_", so use it at your own risk. 
At the same time, I'm happy to help if you need any specific features, it's a good time for feature requests.

## ✅ **What is done** already:

* **Sampling** / **Generating**: it provides the `samplers.Sampler` object to easily generate text.
  See example below, or `cmd/gemma_demo/generator.go` for an example.
* HuggingFace Weights Version:
  * Download weights from HuggingFace, using provided AuthToken -- a read-only token will suffice.
* Kaggle Version
  * Requires manually downloading weights from Kaggle.
  * Use provided `cmd/convert_checkpoint.py` script to convert Jax weights -- requires Python installation.
* A command-line demo `cmd/gemma_demo`, with a simple [Charm](https://charm.sh/) interface.

## ❌ **Not done** yet:

* **Fine-tuning**: the model is there, and it just needs some wiring together. But there is no sample code yet. 

## ⌨️ Sample Code

This is an example of how a `Sampler` object is created (for the simpler HuggingFace version) -- it requires the
HuggingFace token (read-only) used to download to be set in HF_TOKEN -- go to HuggingFace webpage to generate one for you.

```go
package main

import (
    ...
	
    hfd "github.com/gomlx/gemma/download/huggingface"
    "github.com/gomlx/gemma/samplers"
    "github.com/gomlx/gomlx/backends"
    "github.com/gomlx/gomlx/ml/context"
    
    _ "github.com/gomlx/gomlx/backends/xla"
)

var (
    flagModelID = flag.String("model", "google/gemma-2-2b-it", "HuggingFace Gemma model id")
    flagDataDir = flag.String("data", "~/work/gemma", "Directory to cache downloaded dataset files.")
)

func main() {
    flag.Parse()
    prompts := []string{
        "What is 1+1 ?",
        "What are the planets of the solar system?",
        "```\n// BubbleSort is a Go function that sorts the Bubble Sort algorithm\nfunc BubbleSort[S ~[]E, E cmp.Ordered](x S) {\n",
    }
    ctx := context.New()
    vocab, err := hfd.Download(ctx, *flagModelID, os.Getenv("HF_TOKEN"), path.Join(*flagDataDir, "huggingface"))
    if err != nil {
        log.Fatalf("%+v", err)
    }
    sampler, err := samplers.New(backends.New(), ctx, vocab, 1024)
    if err != nil {
        log.Fatalf("%+v", err)
    }
    
    start := time.Now()
    output, err := sampler.Sample([]string{
        "What is 1+1?",
        "What are the planets of the solar system?",
        // "// BubbleSort is a Go function that sorts the Bubble Sort algorithm\nfunc BubbleSort[S ~[]E, E cmp.Ordered](x S)",
    })
    if err != nil {
        log.Fatalf("%+v", err)
    }
    fmt.Printf("\tElapsed time: %s\n", time.Since(start))
    fmt.Printf("Generated text:\n%s\n", strings.Join(output, "\n\n"))
}
```

## 🔗 Resources

1. [**github.com/google-deepmind/gemma**](https://github.com/google-deepmind/gemma):
   [Gemma](https://ai.google.dev/gemma) is a family of open-weights Large Language Model (LLM) by [Google DeepMind](https://deepmind.google/),
   based on Gemini research and technology.
1. <img src="https://raw.githubusercontent.com/eliben/go-sentencepiece/main/doc/toklogo2.png" width="20"/> [github.com/eliben/go-sentencepiece](https://github.com/eliben/go-sentencepiece):
   This is a pure Go implementation of encoding and decoding text with the [SentencePiece tokenizer](https://github.com/google/sentencepiece).


## 📝 TODO

* Remove special symbols from sampling, like "<end_of_turn>".
* Fine-tuning demo.
* Benchmarking: how does it compare to Jax implementation ? Jax JIT-compile the main sampling loop during generation,
  which could be done with GoMLX, but it would require implementing some new features. Not sure it is needed yet.
  * At least in an old NVidia RTX 2080Ti, it works with GoMLX, and Jax reference implementation fails to sample, 
    because it tries to JIT-compile the full sampling loop.