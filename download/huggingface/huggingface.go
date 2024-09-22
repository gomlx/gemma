// Package huggingface handles downloading Gemma model weights from HuggingFace.
//
// This has some advantages from downloading it from Google (Kaggle):
//
//   - With a HuggingFace token, the process is automatic.
//   - No need for conversion of the model, the library reads directly from the HuggingFace ".safetensors" format into
//     GoMLX context.
//   - No Python dependency.
//
// Example:
package huggingface

import (
	"fmt"
	"github.com/gomlx/gemma/sentencepiece"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	gomlxhf "github.com/gomlx/gomlx/ml/data/huggingface"
	"github.com/gomlx/gomlx/types/xslices"
	"path"
	"strconv"
	"strings"
)

// Download will download (if needed) the Gemma model identified by hfID (it's a HuggingFace model id, e.g.: "google/gemma-2-2b-it"),
// and save under the cacheDir (for future reuse).
//
// The hfAuthToken is a HuggingFace token -- read-only access -- that needs to be created once in HuggingFace site.
//
// It loads the weights into the given context and creates a sentencepiece tokenizer (vocab) that is returned.
//
// An error is returned if something fails.
func Download(ctx *context.Context, hfID, hfAuthToken, cacheDir string) (vocab *sentencepiece.Tokenizer, err error) {
	cacheDir = data.ReplaceTildeInDir(cacheDir)
	var hfm *gomlxhf.Model
	hfm, err = gomlxhf.New(hfID, hfAuthToken, cacheDir)
	if err != nil {
		return
	}
	err = hfm.Download()
	if err != nil {
		return
	}

	vocab, err = sentencepiece.NewFromPath(path.Join(hfm.BaseDir, "tokenizer.model"))
	if err != nil {
		return
	}

	for entry, err2 := range hfm.EnumerateTensors() {
		if err2 != nil {
			err = err2
			return
		}
		scopeAndName := convertHuggingFaceNameToScopeAndName(entry.Name)
		if len(scopeAndName) == 0 {
			fmt.Printf("Skipping: %s -> %s\n", entry.Name, entry.Tensor.Shape())
		} else {
			ctxTmp := ctx.In("model")
			name, scope := xslices.Pop(scopeAndName)
			for _, p := range scope {
				ctxTmp = ctxTmp.In(p)
			}
			ctxTmp.VariableWithValue(name, entry.Tensor)
		}
	}
	return
}

func convertHuggingFaceNameToScopeAndName(name string) []string {
	if name == "model.embed_tokens.weight" {
		return []string{"embedder", "input_embedding"}
	} else if name == "model.norm.weight" {
		return []string{"final_norm", "scale"}
	}

	// Parse the layer number for the name prefixed as "model.layers.X.<...>"
	if strings.HasPrefix(name, "model.layers.") {
		parts := strings.Split(name, ".")
		if len(parts) < 5 || xslices.Last(parts) != "weight" {
			return nil
		}
		layerNumberStr := parts[2]
		layerNumber, err := strconv.Atoi(layerNumberStr)
		if err != nil {
			return nil
		}
		layerScope := fmt.Sprintf("layer_%d", layerNumber)
		switch parts[3] {
		case "input_layernorm":
			return append([]string{layerScope, "pre_attention_norm", "scale"})
		case "post_attention_layernorm":
			return append([]string{layerScope, "post_attention_norm", "scale"})
		case "post_feedforward_layernorm":
			return append([]string{layerScope, "post_ffw_norm", "scale"})
		case "pre_feedforward_layernorm":
			return append([]string{layerScope, "pre_ffw_norm", "scale"})
		case "mlp":
			// For the MLP (the GatedFeedForwardNetwork), the weights in HuggingFace are transposed/split differently,
			// so they take new variable names not matching those in Kaggle version.
			switch parts[4] {
			case "down_proj":
				return append([]string{layerScope, "mlp", "hf", "down_proj"})
			case "gate_proj":
				return append([]string{layerScope, "mlp", "hf", "gating_proj"})
			case "up_proj":
				return append([]string{layerScope, "mlp", "hf", "up_proj"})
			default:
				return nil
			}
		case "self_attn":
			return append([]string{layerScope, "attn", "hf", parts[4]})
		default:
			return nil
		}
	}
	return nil
}
