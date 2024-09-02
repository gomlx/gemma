package transformers

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/types/shapes"
)

// SoftCap using Tanh, so values won't go beyond +/- cap. If cap <= 0, it is a no-op.
//
// SoftCap(x) = Tanh(x/cap) * cap
func SoftCap(x *Node, cap float64) *Node {
	if cap <= 0 {
		return x
	}
	return MulScalar(Tanh(DivScalar(x, cap)), cap)
}

// KernelEinsum multiplies the input by a kernel of the given shape, using the given graph.EinSum equation.
func KernelEinsum(ctx *context.Context, equation string, x *Node, kernelShape shapes.Shape) *Node {
	g := x.Graph()
	kernelVar := ctx.VariableWithShape("w", kernelShape)
	kernel := kernelVar.ValueGraph(g)
	return Einsum(equation, x, kernel)

}

// RMSNorm normalizes by its root-mean-square x = x / √(mean(sqrt(x), axis=-1) + epsilon) and applies a learned scale.
func RMSNorm(ctx *context.Context, x *Node) *Node {
	g := x.Graph()
	variance := ReduceAndKeep(Square(x), ReduceMean, -1)
	const epsilon = 1e-6
	normalizedX := Mul(x, Rsqrt(AddScalar(variance, epsilon)))

	// Now apply a learned scale.
	scaleVar := ctx.WithInitializer(initializers.Zero).
		VariableWithShape("scale", shapes.Make(x.DType(), x.Shape().Dim(-1)))
	scale := scaleVar.ValueGraph(g)
	scale = ExpandLeftToRank(scale, normalizedX.Rank()) // Expand rank of scale to match normalizedX.
	// Scale centered on 1.0 (so 0.0 has no effect).
	scale = OnePlus(scale)
	normalizedX = Mul(scale, normalizedX)
	return normalizedX
}

// RoPEDefaultMaxWaveLength is a default value to use for rotary positional encoding.
// See ApplyRotaryPositionEncoding.
const RoPEDefaultMaxWaveLength = 10_000

// ApplyRotaryPositionEncoding (aka. RoPE) applies the positional encoding to the operand, given the positions
// (integer numbers of the position).
//
// - operand: the last axis ("features" or "embedding" axis) must be divisible by 2. The shape usually is [batchSize, sequenceSize, numHeads, headDim].
// - positions: its shape must be a prefix to operand. Typically, it's shaped [batchSize, sequenceSize].
// - maxWaveLength: it uses wave lengths in a power scale, up to maxWaveLength -- see RoPEDefaultMaxWaveLength for a reasonable value.
//
// Reference: https://arxiv.org/abs/2104.09864
func ApplyRotaryPositionEncoding(operand, positions *Node, maxWaveLength int) *Node {
	g := operand.Graph()
	dtype := operand.DType()
	featuresDim := operand.Shape().Dim(-1)
	if featuresDim <= 0 || featuresDim%2 != 0 {
		exceptions.Panicf("ApplyRotaryPositionEncoding(operand=%s, position=%s) requires operand's last "+
			"dimension to be >= 0 and divisible by 2", operand.Shape(), positions.Shape())
	}

	transientDType := dtype
	fraction := Iota(g, shapes.Make(transientDType, featuresDim/2), 0)
	fraction = MulScalar(fraction, 2.0/float64(featuresDim))
	timeScale := Pow(Scalar(g, transientDType, float64(maxWaveLength)), fraction)
	timeScale = ExpandLeftToRank(timeScale, positions.Rank()+1)

	// Angles shape will add a rank to positions: we will take each position at a different wave length (or timeScale).
	angles := ConvertDType(ExpandDims(positions, -1), transientDType)
	angles = Div(angles, timeScale)

	// Insert an axis just before the last until it matches the operand's shape.
	for angles.Rank() < operand.Rank() {
		angles = ExpandDims(angles, -2)
	}
	sines := Sin(angles)
	cosines := Cos(angles)

	// Split first/second half of operands features (the last dimension), and apply rotation at the various wave lengths.
	firstHalf := Slice(operand, AxisRange().Spacer(), AxisRange(0, featuresDim/2))
	secondHalf := Slice(operand, AxisRange().Spacer(), AxisRangeToEnd(featuresDim/2))
	firstHalfUpdate := Sub(
		Mul(firstHalf, cosines),
		Mul(secondHalf, sines),
	)
	secondHalfUpdate := Add(
		Mul(secondHalf, cosines),
		Mul(firstHalf, sines),
	)
	return ConvertDType(Concatenate([]*Node{firstHalfUpdate, secondHalfUpdate}, -1), dtype)
}

// GatedFeedForward layer for Gemma:
// - hiddenDim: one intermediary layer.
// - transposeGatingEinsum: for some versions of Gemma, the gating (hidden) weights have the axes transposed.
// - It uses Gelu as activation function for the gating signal (multiplied by the up-projected values).
func GatedFeedForward(ctx *context.Context, x *Node, hiddenDim int, transposeGatingEinsum bool) *Node {
	g := x.Graph()
	featuresDim := x.Shape().Dim(-1)

	var gatingWeights *Node
	if transposeGatingEinsum {
		// Some versions of Gemma use an alternate parameter ordering that transposes hiddenDim and outputDim.
		gatingVar := ctx.WithInitializer(initializers.Zero).
			VariableWithShape("gating_einsum", shapes.Make(x.DType(), 2, hiddenDim, featuresDim))
		gatingWeights = gatingVar.ValueGraph(g)
		gatingWeights = Transpose(gatingWeights, 1, 2)
	} else {
		// Standard shape of the gating weights.
		gatingVar := ctx.WithInitializer(initializers.Zero).
			VariableWithShape("gating_einsum", shapes.Make(x.DType(), 2, featuresDim, hiddenDim))
		gatingWeights = gatingVar.ValueGraph(g)
	}
	gatingWeights0 := Squeeze(Slice(gatingWeights, AxisElem(0)), 0)
	gatingWeights1 := Squeeze(Slice(gatingWeights, AxisElem(1)), 0)

	gateValue := DotGeneral(x, []int{-1}, nil, gatingWeights0, []int{0}, nil)
	gateValue = activations.Gelu(gateValue)

	upProjection := DotGeneral(x, []int{-1}, nil, gatingWeights1, []int{0}, nil)
	upProjection = Mul(gateValue, upProjection) // Gate upProjection.

	downProjectionVar := ctx.WithInitializer(initializers.Zero).
		VariableWithShape("linear", shapes.Make(x.DType(), hiddenDim, featuresDim))
	downProjectionWeights := downProjectionVar.ValueGraph(g)
	output := DotGeneral(upProjection, []int{-1}, nil, downProjectionWeights, []int{0}, nil)
	return output
}
