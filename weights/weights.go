// Package weights loads Gemma weights into tensors along with the matching metadata.
package weights

import (
	"github.com/gomlx/gemma/trees"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
	"github.com/pkg/errors"
	"github.com/vmihailenco/msgpack"
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
// (see github.com/gomlx/gemma repository, under cmd/convert_checkpoint.py)
//
// It returns a tree of tensors, with the path matching those of the original Jax checkpoint.
func ReadConvertedWeights(checkpointDir string) (tree *trees.Tree[*tensors.Tensor], err error) {
	rawDir := path.Join(checkpointDir, "raw")
	if !data.FileExists(rawDir) {
		err = errors.Errorf(
			"ReadConvertedWeights(%q), the given directory doesn't have a subdirectory 'raw/' with the converted files",
			checkpointDir)
		return
	}
	tree = trees.New(trees.NewMap[*tensors.Tensor]())
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
		base := path.Join(rawDir, strings.TrimSuffix(filePath, ext))
		shapeFilePath := base + ".shape"
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
			return errors.Wrapf(err, "failed to convert %q to a dimension, read from %q", shapeBytes, base+".shape")
		}
		shape := shapes.Make(dtype, shapeDims...)

		rawFilePath := base + ".raw"
		info, err := entry.Info()
		if err != nil {
			return errors.Wrapf(err, "failed to get info from %q", rawFilePath)
		}
		if info.Size() != int64(shape.Memory()) {
			return errors.Errorf("file %q has %d bytes, but shape %s (read from %q) requires %d bytes, something went wrong in the conversion",
				rawFilePath, info.Size(), shape, shapeFilePath, shape.Memory())
		}

		tensor := tensors.FromShape(shape)
		f, err := os.Open(rawFilePath)
		if err != nil {
			return errors.Wrapf(err, "failed to open raw data from %q", rawFilePath)
		}
		var n int
		tensor.MutableBytes(func(data []byte) {
			n, err = f.Read(data)
		})
		_ = f.Close()
		if err != nil {
			return errors.Wrapf(err, "failed to read raw data from %q", rawFilePath)
		}
		if n != int(shape.Memory()) {
			return errors.Wrapf(err, "read %d bytes from %q, expected %d", n, rawFilePath, shape.Memory())
		}
		treePath := strings.Split(filePath, "/")
		treePath = treePath[:len(treePath)-1]
		tree.Insert(treePath, tensor)
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
