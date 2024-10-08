// Package kaggle loads Gemma weights into tensors along with the matching metadata, after they
// have been downloaded from kaggle and converted using the included cmd/convert_checkpoint.py.
package kaggle

import (
	"github.com/dustin/go-humanize"
	"github.com/gomlx/gemma/trees"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
	"github.com/pkg/errors"
	"github.com/vmihailenco/msgpack"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	AggregateFileName = "checkpoint"

	// OCDBTManifestFileName indicates usage of "Orbax Consistent Distributed Backend Tree" (OCDBT).
	OCDBTManifestFileName = "manifest.ocdbt"
)

// ReadConvertedWeights from checkpointDir (under the "raw/" subdirectory).
// It will read the weights and shape converted by the `convert_checkpoint.py` script
// (see github.com/gomlx/gemma repository, under cmd/convert_checkpoint.py) and set them
// in the given context, under its current scope.
func ReadConvertedWeights(ctx *context.Context, checkpointDir string) error {
	weights, err := ReadConvertedWeightsToTree(checkpointDir)
	if err != nil {
		return err
	}
	UploadWeightsToContext(ctx.In("model"), weights)
	return nil
}

// ReadConvertedWeightsToTree from checkpointDir (under the "raw/" subdirectory).
// It will read the weights and shape converted by the `convert_checkpoint.py` script
// (see github.com/gomlx/gemma repository, under cmd/convert_checkpoint.py)
//
// It returns a tree of tensors, with the path matching those of the original Jax checkpoint.
func ReadConvertedWeightsToTree(checkpointDir string) (tree *trees.Tree[*tensors.Tensor], err error) {
	rawDir := path.Join(checkpointDir, "raw")
	if !data.FileExists(rawDir) {
		err = errors.Errorf(
			"ReadConvertedWeights(%q), the given directory doesn't have a subdirectory 'raw/' with the converted files",
			checkpointDir)
		return
	}
	tree = trees.New[*tensors.Tensor]()
	err = fs.WalkDir(os.DirFS(rawDir), ".", func(filePath string, entry fs.DirEntry, err error) error {
		if err != nil {
			return errors.Wrapf(err, "failed to traverse %q", rawDir)
		}
		if entry.IsDir() {
			return nil
		}
		ext := filepath.Ext(filePath)
		if ext != ".raw" {
			return nil
		}

		// Here we have teh pair of files ".shape" and ".raw":
		base := strings.TrimSuffix(filePath, ext)
		basePath := path.Join(rawDir, base)
		shapeFilePath := basePath + ".shape"
		if !data.FileExists(shapeFilePath) {
			return nil
		}
		shapeBytes, err := os.ReadFile(shapeFilePath)
		if err != nil {
			return errors.Wrapf(err, "failed to read shape from %q", shapeFilePath)
		}
		shapeParts := strings.Split(string(shapeBytes), ",")
		dtype, err := dtypes.DTypeString(shapeParts[0])
		if err != nil {
			return errors.Wrapf(err, "unknown dtype read from %q", shapeFilePath)
		}
		shapeDims := xslices.Map(shapeParts[1:], func(s string) (v int) {
			if err != nil {
				return 0
			}
			v, err = strconv.Atoi(s)
			return
		})
		if err != nil {
			return errors.Wrapf(err, "failed to convert %q to a dimension, read from %q", shapeBytes, basePath+".shape")
		}
		shape := shapes.Make(dtype, shapeDims...)

		rawFilePath := basePath + ".raw"
		info, err := entry.Info()
		if err != nil {
			return errors.Wrapf(err, "failed to get info from %q", rawFilePath)
		}
		if info.Size() != int64(shape.Memory()) {
			return errors.Errorf("file %q has %d bytes, but shape %s (read from %q) requires %d bytes, something went wrong in the conversion",
				rawFilePath, info.Size(), shape, shapeFilePath, shape.Memory())
		}

		treePath := strings.Split(base, "/")
		//fmt.Printf("%q -> %s\n", treePath, shape)

		tensor := tensors.FromShape(shape)
		f, err := os.Open(rawFilePath)
		if err != nil {
			return errors.Wrapf(err, "failed to open raw data from %q", rawFilePath)
		}
		var n int
		tensor.MutableBytes(func(data []byte) {
			n, err = io.ReadFull(f, data)
		})
		_ = f.Close()
		if err != nil {
			return errors.Wrapf(err, "failed to read raw data from %q", rawFilePath)
		}
		if n != int(shape.Memory()) {
			return errors.Errorf("read %d bytes from %q, expected %d", n, rawFilePath, shape.Memory())
		}
		err = tree.Set(treePath, tensor)
		if err != nil {
			return errors.WithMessagef(err, "failed to set variable with %s", humanize.Bytes(uint64(n)))
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return
}

// PyReadAggregate of Python checkpoint. Not used by Gemma v2.
func PyReadAggregate(checkpointDir string) (results any, err error) {
	checkpointDir = data.ReplaceTildeInDir(checkpointDir)
	aggregatePath := path.Join(checkpointDir, AggregateFileName)
	var f *os.File
	f, err = os.Open(aggregatePath)
	if err != nil {
		err = errors.Wrapf(err, "failed to read aggregate checkpoint file from %q", aggregatePath)
		return
	}

	dec := msgpack.NewDecoder(f)
	results, err = dec.DecodeMap()
	defer func() { _ = f.Close() }()
	return
}

func isOCDBT(checkpointDir string) bool {
	checkpointDir = data.ReplaceTildeInDir(checkpointDir)
	ocdbtPath := path.Join(checkpointDir, OCDBTManifestFileName)
	return data.FileExists(ocdbtPath)
}

type PyParamInfo struct {
	Name, Path      string
	SkipDeserialize bool
}

func PyReadParamInfo(checkpointDir string) *trees.Tree[*PyParamInfo] {
	checkpointDir = data.ReplaceTildeInDir(checkpointDir)
	metadata := must.M1(ReadMetadata(checkpointDir))
	return trees.Map(metadata, func(p trees.Path, meta *Metadata) *PyParamInfo {
		name := strings.Join(p, ".")
		pi := &PyParamInfo{
			Name:            name,
			Path:            path.Join(checkpointDir, name),
			SkipDeserialize: meta.SkipDeserialize,
		}
		return pi
	})
}

// UploadWeightsToContext creates variables corresponding to the weights.
// It returns the ctx given, with the variables set.
//
// It's tightly coupled with the model building functions in this package.
// Meaning the modeling must match the naming here.
func UploadWeightsToContext(ctx *context.Context, weights *trees.Tree[*tensors.Tensor]) {
	weights = weights.Map["transformer"]
	for treePath, tensor := range weights.Leaves() {
		scopedCtx := ctx
		scopeParts := treePath[:len(treePath)-1]
		for _, p := range scopeParts {
			scopedCtx = scopedCtx.In(p)
		}
		varName := treePath[len(treePath)-1]
		_ = scopedCtx.VariableWithValue(varName, tensor)
	}
}
