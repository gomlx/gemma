// Package transformers implements the various Gema models.
// It is based on https://github.com/google-deepmind/gemma/blob/main/gemma/transformer.py
package transformers

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
)

// BuildPositionsFromMask where inputMask is true for non-padded tokens.
//
// It returns the indices to use for RoPE (Rotary Position Embedding).
//
// Example:
//
//	BuildPositionFromMask([[True, True, False, False],
//						   [True, True, True, False]])
//	> [0, 1, 1, 1], [0, 1, 2, 2]
func BuildPositionsFromMask(backend backends.Backend, inputMask *tensors.Tensor) *tensors.Tensor {
	return NewExec(backend, func(mask *Node) *Node {
		g := mask.Graph()
		positions := CumSum(ConvertDType(mask, dtypes.Int32), -1)
		// Make it 0-based (as opposed to starting with 1), for rows that are not empty (all zeros).
		nonZero := GreaterThan(positions, ScalarZero(g, dtypes.Int32))
		positions = Sub(positions, ConvertDType(nonZero, dtypes.Int32))
		return positions
	}).Call(inputMask)[0]
}
