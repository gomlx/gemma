package transformers

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/shapes"
)

// RMSNorm normalizes by its root-mean-square (x = x / (mean(sqrt(x), axis=-1) + epsilon)) and applies a learned scale.
func RMSNorm(ctx *context.Context, x *Node) *Node {
	g := x.Graph()
	variance := ReduceAndKeep(Square(x), ReduceMean, -1)
	const epsilon = 1e-6
	normalizedX := Mul(x, Rsqrt(AddScalar(variance, epsilon)))

	// Now apply a learned scale.
	scaleVar := ctx.WithInitializer(initializers.Zero).
		VariableWithShape("scale", shapes.Make(x.DType(), x.Shape().Dim(-1)))
	scale := scaleVar.ValueGraph(g)
	// Scale centered on 1.0 (so 0.0 has no effect).
	scale = OnePlus(scaleVar.ValueGraph(g))
	scale = ExpandLeftToRank(scale, normalizedX.Rank()) // Expand rank of scale to match normalizedX.
	normalizedX = Mul(x, scale)
	return normalizedX
}
