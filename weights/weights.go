// Package weights loads Gemma weights into tensors along with the matching metadata.
package weights

import (
	"github.com/gomlx/gomlx/ml/data"
	"github.com/pkg/errors"
	"github.com/vmihailenco/msgpack"
	"os"
	"path"
)

const (
	AggregateFileName = "checkpoint"
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
