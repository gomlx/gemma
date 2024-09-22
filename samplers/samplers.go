// Package samplers uses a transformer model to generate senteces based on prompts.
package samplers

import (
	"github.com/dustin/go-humanize"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gemma/transformers"
	"github.com/gomlx/gemma/trees"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	klog "k8s.io/klog/v2"
	"slices"
	"time"
)

type Vocabulary interface {
	EncodeAsIDs(text string) []int
	DecodeIDs([]int) string

	// The methods below define the special ids for the model.

	BeginningOfSentenceID() int
	EndOfSentenceID() int
	UnknownID() int
	PadID() int
}

// Sampler has a transformer (LLM) model and a vocabulary (sentencepiece) configured and generates
// sentences based on prompts.
type Sampler struct {
	Backend backends.Backend
	Vocab   Vocabulary

	// MaxGeneratedTokens default for Sampler.Sample.
	MaxGeneratedTokens int

	// Context with the model weights, used to execute the model.
	Context *context.Context

	// SampleStep graph computation.
	SampleStep *context.Exec

	// Config of the Gemma model, created from the weights.
	Config *transformers.Config

	// CacheTreeStructure holds the structure of the tree used for caching: the tree structure (paths) is stable
	// across different calls to Sample.
	CacheTreeStructure *trees.Tree[struct{}]
}

// New creates a new sampler with the registered vocabulary and model.
func New(backend backends.Backend, ctx *context.Context, vocab Vocabulary, maxGeneratedTokens int) (*Sampler, error) {
	s := &Sampler{
		Backend:            backend,
		Vocab:              vocab,
		MaxGeneratedTokens: maxGeneratedTokens,
		Context:            ctx,
	}
	var err error
	s.Config, err = transformers.NewConfigFromContext(s.Context.In("model"))
	if err != nil {
		return nil, err
	}
	s.Context = s.Context.Reuse()
	s.SampleStep = context.NewExec(backend, s.Context, s.sampleStepGraphFn())
	return s, nil
}

// Sample the continuation from the given prompts.
func (s *Sampler) Sample(prompts []string) ([]string, error) {
	return s.SampleMaxTokens(prompts, s.MaxGeneratedTokens)
}

// SampleMaxTokens is like Sample, but instead of using the default MaxGenerateTokens, uses the given maxTokens instead.
func (s *Sampler) SampleMaxTokens(prompts []string, maxTokens int) ([]string, error) {
	promptIds := xslices.Map(prompts, s.Vocab.EncodeAsIDs)
	state, err := s.initialState(promptIds, maxTokens)
	if err != nil {
		return nil, err
	}
	err = exceptions.TryCatch[error](func() {
		state = s.sampleLoop(state)
	})
	if err != nil {
		return nil, err
	}
	return s.decode(state), nil
}

// sampleLoop, executes a sampleStep until all examples in the batch are finished.
func (s *Sampler) sampleLoop(state samplingState) samplingState {
	// Prepare inputs as slice:
	// * If you change this order, change the parsing order in sampleStepGraphFn below.
	inputs := []any{
		state.InputBuffer,
		state.StepNum,
		state.Done,
	}
	// * Append cache values.
	cacheValues := trees.ValuesAsList(state.Cache.Data)
	inputs = append(inputs, xslices.Map(cacheValues, func(t *tensors.Tensor) any { return t })...)
	numMutableInputs := len(inputs)

	// Append constant inputs.
	start := time.Now()
	inputs = append(inputs,
		state.Positions,
	)
	var outputs []*tensors.Tensor
	var execTime, inputsPrepTime time.Duration
	var count int
	for {
		inputPrepStart := time.Now()
		// We donate all the inputs, since they are all going to be updated (saves some GPU memory).
		for ii := range numMutableInputs {
			inputs[ii] = DonateTensorBuffer(inputs[ii].(*tensors.Tensor), s.Backend)
		}
		inputsPrepTime += time.Since(inputPrepStart)

		// Execute a step.
		execStart := time.Now()
		outputs = s.SampleStep.Call(inputs...)
		execTime += time.Since(execStart)
		count++

		// Update states (the output has the same order as the input).
		for ii := range numMutableInputs {
			inputs[ii] = outputs[ii]
		}
		extraOutputs := outputs[numMutableInputs:] // Separate the transient outputs.
		done := tensors.ToScalar[bool](extraOutputs[0])

		// End-of-sampling:
		if done {
			break
		}
	}
	if klog.V(1).Enabled() {
		elapsed := time.Since(start)
		klog.Infof("Sample execution time (%d steps): %s", count, elapsed)
		klog.Infof("> Graph execution time: %s", execTime)
		klog.Infof("> Inputs preparation time: %s", inputsPrepTime)
	}
	state.InputBuffer = outputs[0]
	state.Positions = outputs[1]
	state.StepNum = outputs[2]
	state.Done = outputs[3]
	updatedCache := trees.FromValuesAndTree(outputs[4:4+s.CacheTreeStructure.NumLeaves()], s.CacheTreeStructure)
	state.Cache.Data = updatedCache
	return state
}

// buildSampleStepGraphFn returns the computation graph building function for this sampler.
// The returned function can be used by context.NewExec.
func (s *Sampler) sampleStepGraphFn() func(*context.Context, []*Node) []*Node {
	return func(ctx *context.Context, state []*Node) []*Node {
		g := state[0].Graph() // Reference to the underlying graph, it could be from any of the inputs.
		_ = ctx

		// Extract state parts:
		stateFieldsIdx := 0
		nextState := func() *Node {
			field := state[stateFieldsIdx]
			stateFieldsIdx++
			return field
		}
		// This order has to match the order fed in sampleLoop.
		// - Mutable fields, to be updated.
		inputBuffer := nextState()
		stepNum := nextState()
		done := nextState()
		numCacheValues := s.CacheTreeStructure.NumLeaves()
		cache := trees.FromValuesAndTree(state[stateFieldsIdx:stateFieldsIdx+numCacheValues], s.CacheTreeStructure)
		stateFieldsIdx += numCacheValues

		// - Constant fields.
		positions := nextState()

		// Take the current step token for all examples of the batch.
		batchSize := inputBuffer.Shape().Dimensions[0]
		zeroIdx := ScalarZero(g, dtypes.Int32)
		currentTokens := DynamicSlice(inputBuffer, []*Node{zeroIdx, stepNum}, []int{batchSize, 1})
		currentTokens.AssertDims(batchSize, 1)
		currentPositions := DynamicSlice(positions, []*Node{zeroIdx, stepNum}, []int{batchSize, 1})
		currentPositions.AssertDims(batchSize, 1)

		// Attention to all positions < current step number (stepNum).
		// Notice that the cache rotates, so once stepNum > Config.MaxCacheLength, the mask will be
		// true everywhere.
		cacheAttentionMask := Iota(g, shapes.Make(dtypes.Int32, batchSize, 1, s.Config.MaxCacheLength), -1)
		cacheAttentionMask = LessOrEqual(cacheAttentionMask, stepNum)

		logits := transformers.GemmaWithCache(ctx.In("model"), s.Config,
			currentTokens, currentPositions, cache, cacheAttentionMask)
		logits.AssertDims(batchSize, 1, s.Config.VocabularySize)

		nextTokenNum := OnePlus(stepNum)
		nextPredictedTokens := ArgMax(logits, -1)
		nextPredictedTokens.AssertDims(batchSize, 1)
		nextTokenStartIdx := []*Node{zeroIdx, nextTokenNum}
		nextTokens := DynamicSlice(inputBuffer, nextTokenStartIdx, []int{batchSize, 1})
		nextTokens.AssertDims(batchSize, 1)
		nextTokens = Where(
			Or(
				Equal(nextTokens, Const(g, int32(s.Vocab.PadID()))),
				ExpandDims(done, -1),
			),
			nextPredictedTokens,
			nextTokens,
		)
		inputBuffer = DynamicUpdateSlice(inputBuffer, nextTokens, nextTokenStartIdx)
		eosToken := Scalar(g, dtypes.Int32, s.Vocab.EndOfSentenceID())
		nextTokenIsEOS := Squeeze(Equal(nextTokens, eosToken), -1)
		done = Or(done, nextTokenIsEOS)

		// Prepare next step: are we done ?
		stepNum = nextTokenNum
		maxSteps := inputBuffer.Shape().Dimensions[1] - 2
		allDone := Or(
			LogicalAll(done),
			GreaterOrEqual(stepNum, Const(g, int32(maxSteps))),
		)

		// Outputs: updated mutable values first including cache):
		outputs := []*Node{inputBuffer, stepNum, done}
		outputs = append(outputs, trees.ValuesAsList(cache)...)
		// - Other results:
		outputs = append(outputs, allDone)
		return outputs
	}
}

// samplingState holds the state of the sampling loop plus some constants for the loop.
type samplingState struct {
	// BatchSize, MaxTokens, TotalLength are constants for one sampling.
	BatchSize, MaxTokens, TotalLength int

	// InputBuffer holds the ids with prepended <bos> (beginning-of-sentence) and padding (<pad>) and extra space for
	// an <eos> (end-of-sentence).
	InputBuffer *tensors.Tensor

	// NumInputTokens is the number of tokens on the original input per example: shaped int32[batch_size].
	NumInputTokens *tensors.Tensor

	// Positions for each token, see transformers.BuildPositionsFromMask
	Positions *tensors.Tensor

	// StepNum is a scalar counter of the steps sampled (decoded) so far.
	StepNum *tensors.Tensor

	// Done is a vector of the inputs who are done with the generation: shaped bool[batch_size].
	Done *tensors.Tensor

	// Cache used during the sampling.
	Cache *transformers.Cache
}

// initialState creates a tensor shaped int32[batchSize, totalLength+2] padded with the Vocab.PadId filled (left to right)
// with the given promptIds.
//
// It also returns the mask, that is set to true where it is not padding.
//
// It also adds a "bos" (beginning of sentence) token to each prompt.
func (s *Sampler) initialState(promptIds [][]int, maxTokens int) (state samplingState, err error) {
	state.MaxTokens = maxTokens
	state.BatchSize = len(promptIds)
	batchSize := state.BatchSize

	lengths := xslices.Map(promptIds, func(seq []int) int32 { return int32(len(seq)) + 1 }) // +1 for <bos> (beginning-of-sentence) token.
	state.NumInputTokens = tensors.FromValue(lengths)                                       // Shape [batchSize]
	maxInputLength := int(slices.Max(lengths))
	state.TotalLength = maxInputLength + maxTokens + 1 // +1 for <eos>.
	totalLength := state.TotalLength

	state.StepNum = tensors.FromScalar(int32(0))
	state.InputBuffer = tensors.FromScalarAndDimensions(int32(s.Vocab.PadID()), batchSize, totalLength)
	bos := int32(s.Vocab.BeginningOfSentenceID())

	// Copy over "ragged" promptIds to dense InputBuffer (filled with <pad>), prepending <bos>,
	// and set InputMask to true where InputBuffer != <pad>.
	tensors.MutableFlatData(state.InputBuffer, func(flatIDs []int32) {
		for exampleIdx := range batchSize {
			exampleIds := flatIDs[exampleIdx*totalLength : (exampleIdx+1)*totalLength]
			exampleIds[0] = bos
			for ii, value := range promptIds[exampleIdx] {
				exampleIds[1+ii] = int32(value)
			}
		}
	})

	// Notice that the convoluted code in https://github.com/google-deepmind/gemma/blob/main/gemma/sampler.py
	// (see Sampler.init_sample_state()) in the end simply does the same as Iota() -- except if the input
	// has pad symbols (not the padding added by filling the ragged input) inside it -- which is not doable when
	// converting from string.
	//
	// Probably there is a bug in the original code ... it's not documented what they intended to do with it, but
	// we simply take the iota here.
	state.Positions = ExecOnce(s.Backend, func(g *Graph) *Node {
		return Iota(g, shapes.Make(dtypes.Int32, batchSize, totalLength), -1)
	})

	state.Done = tensors.FromShape(shapes.Make(dtypes.Bool, batchSize))

	// Setup cache, and if not yet setup, configure cache structure.
	var start time.Time
	if klog.V(1).Enabled() {
		start = time.Now()
	}
	state.Cache, err = transformers.NewCache(s.Config, batchSize)
	if err != nil {
		return
	}
	if s.CacheTreeStructure == nil {
		s.CacheTreeStructure = trees.Map(state.Cache.Data, func(_ trees.Path, _ *tensors.Tensor) (empty struct{}) { return })
	}

	if klog.V(1).Enabled() {
		elapsed := time.Since(start)
		var cacheMem uintptr
		for _, t := range state.Cache.Data.Leaves() {
			cacheMem += t.Memory()
		}
		klog.Infof("cache: elapsed %s, memory used %s\n", elapsed, humanize.Bytes(uint64(cacheMem)))
	}
	return
}

// decode converts the state's InputBuffer with the sampled tokens to actual text.
func (s *Sampler) decode(state samplingState) []string {
	text := make([]string, state.BatchSize)
	totalLength := state.TotalLength
	tensors.ConstFlatData(state.InputBuffer, func(flatIds []int32) {
		for exampleIdx := range state.BatchSize {
			exampleIds := flatIds[exampleIdx*totalLength : (exampleIdx+1)*totalLength]
			ids := xslices.Map(exampleIds, func(id int32) int { return int(id) })
			nonPad := 0
			for _, id := range ids {
				if id == s.Vocab.EndOfSentenceID() || id == s.Vocab.PadID() {
					break
				}
				nonPad++
			}
			text[exampleIdx] = s.Vocab.DecodeIDs(ids[:nonPad]) // Notice <pad>, <bos> and <eos> are converted to empty strings.
			//fmt.Printf("tokens: %#v\n", ids[:nonPad])
		}
	})
	return text
}

// UpdateCacheAttentionMaskGraph given an inputMask (on the whole batch of example token ids), a currentStep and
// attentionLen (static).
//
// It's based on _compute_attention_mask in https://github.com/google-deepmind/gemma/blob/main/gemma/sampler.py#L32:
// the inputs and outputs here are very cryptic ... my best guess (also with some help from Gemini) what the original
// authors meant to generate is a mask that is False to where they can attend, except if it is in the "future"
// ("future" means positions > currentStep).
//
// - currentStep: scalar with current step.
// - attentionLen: length of the attention mask (it's ok to be larger than inputMask, it will pad the output accordingly)
// - inputMask: mask of valid tokens in the input, shaped [batchSize, inputLen]
func UpdateCacheAttentionMaskGraph(currentStep *Node, attentionLen int, inputMask *Node) *Node {
	return nil
}
