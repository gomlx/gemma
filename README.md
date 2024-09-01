
<img align="right" src="https://github.com/gomlx/gomlx/raw/main/docs/gomlx_gopher.jpg" alt="GoMLX Gopher" width="128px"/>

# GoMLX Gemma

GoMLX (for Go) port of Google Deepmind's Gemma GenAI/LLM model.

## 📖 About gemma


An implementation of [Google DeepMind](deepmind.google)'s [Gemma model](https://github.com/google-deepmind/gemma?tab=readme-ov-file)
using [GoMLX, a Machine Learning framework for Go](https://github.com/gomlx/gomlx).

✅ **What is done** already:

* **Importing weights from Jax model**: See `cmd/convert_checkpoint.py` script.
* **Sampling** / **Generating**: it provides the `samplers.Sampler` object to easily generate text. 
  See `cmd/gemma_demo/generator.go` for an example.
* A command-line demo `cmd/gemma_demo`, with a simple [Charm](https://charm.sh/) interface.

❌ **Not done** yet:

* **Fine-tuning**: the model is there, and it just needs some wiring together. But there is no sample code yet. 

## ⌨️ Sample Code

This is an example of how a `Sampler` object is created:

```go
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
```

And here is how it is used:

```go
...
    prompts := []string{
      "// BubbleSort is a Go function that sorts the Bubble Sort algorithm\nfunc BubbleSort[S ~[]E, E cmp.Ordered](x S) {\n",
      "What are the planets of the solar system?",	
    }   	
    outputs, err := sampler.Sample(prompts)
	if err != nil {
		return "", err
	}
	for _, output := range outputs {
		fmt.Printf("%s\n\n", output)
    }   
...
```

## 🔗 Resources

1. [**github.com/google-deepmind/gemma**](https://github.com/google-deepmind/gemma):
   [Gemma](https://ai.google.dev/gemma) is a family of open-weights Large Language Model (LLM) by [Google DeepMind](https://deepmind.google/),
   based on Gemini research and technology.
1. <img src="https://raw.githubusercontent.com/eliben/go-sentencepiece/main/doc/toklogo2.png" width="20"/> [github.com/eliben/go-sentencepiece](https://github.com/eliben/go-sentencepiece):
   This is a pure Go implementation of encoding and decoding text with the [SentencePiece tokenizer](https://github.com/google/sentencepiece).


## 📝 TODO

* Prevent up-scaling dtype (from bfloat16 to float32) during ApplyRotaryPositionEncoding: probably save a bit of memory and time.
* Remove special symbols from sampling, like "<end_of_turn>".
* Fine-tuning demo.