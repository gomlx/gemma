package transformers

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
)

// RMSNorm normalizes by its root-mean-square (x = x / (mean(sqrt(x), axis=-1) + epsilon)) and applies a learned scale.
//
// If weights is not nil, the scale is taken from there, otherwise it is initialized as default by the context.
func RMSNorm(ctx *context.Context, x *Node, weights *tensors.Tensor) *Node {
	g := x.Graph()
	variance := ReduceAndKeep(Square(x), ReduceMean, -1)
	const epsilon = 1e-6
	normedX := Mul(x, Rsqrt(AddScalar(variance, epsilon)))

	// Now apply a learned scale.
	var scaleVar *context.Variable
	if weights != nil {
		// take scale from loaded weights.
		scaleVar = ctx.VariableWithValue("scale", weights)
	} else {
		// scale is a no-op at 0, so we force a zero-initializer by default.
		scaleVar = ctx.WithInitializer(initializers.Zero).
			VariableWithShape("scale", shapes.Make(x.DType(), x.Shape().Dim(-1)))
	}
	scale := scaleVar.ValueGraph(g)
	scale = OnePlus(scale)                          // Scale centered on 1.0 (no effect).
	scale = ExpandLeftToRank(scale, normedX.Rank()) // Expand rank of scale to match normedX.
	normedX = Mul(x, scale)
	return normedX
}
