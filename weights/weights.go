// Package weights loads Gemma weights into tensors along with the matching metadata.
package weights

import (
	"github.com/gomlx/gemma/trees"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/janpfeifer/must"
	"github.com/pkg/errors"
	"github.com/vmihailenco/msgpack"
	"os"
	"path"
	"strings"
)

const (
	AggregateFileName = "checkpoint"

	// OCDBTManifestFileName indicates usage of "Orbax Consistent Distributed Backend Tree" (OCDBT).
	OCDBTManifestFileName = "manifest.ocdbt"
)

// ReadAggregate of checkpoint. Not used by Gemma v2.
func ReadAggregate(checkpointDir string) (results any, err error) {
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

type ParamInfo struct {
	Name, Path      string
	SkipDeserialize bool
}

func ReadParamInfo(checkpointDir string) *trees.Tree[*ParamInfo] {
	checkpointDir = data.ReplaceTildeInDir(checkpointDir)
	metadata := must.M1(ReadMetadata(checkpointDir))
	return trees.Map(metadata, func(p trees.Path, meta *Metadata) *ParamInfo {
		name := strings.Join(p, ".")
		pi := &ParamInfo{
			Name:            name,
			Path:            path.Join(checkpointDir, name),
			SkipDeserialize: meta.SkipDeserialize,
		}
		return pi
	})
}
