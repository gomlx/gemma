package weights

import (
	"encoding/json"
	"fmt"
	"github.com/gomlx/gemma/trees"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/pkg/errors"
	"os"
	"path"
	"regexp"
	"strings"
)

const (
	MetadataFileName = "_METADATA"
	KeyUseZarr3      = "use_zarr3"
	KeyTreeMetadata  = "tree_metadata"
	KeyKeyMetadata   = "key_metadata"
	KeyValueMetadata = "value_metadata"
)

// ReadMetadata returns the metadata loaded from the given directory in the form of a tree.
func ReadMetadata(checkpointDir string) (tree *trees.Tree[*Metadata], err error) {
	checkpointDir = data.ReplaceTildeInDir(checkpointDir)
	metadataPath := path.Join(checkpointDir, MetadataFileName)
	var f *os.File
	f, err = os.Open(metadataPath)
	if err != nil {
		err = errors.Wrapf(err, "failed to read aggregate checkpoint file from %q", metadataPath)
		return
	}
	defer func() { _ = f.Close() }()

	dec := json.NewDecoder(f)
	var jsonTree any
	err = dec.Decode(&jsonTree)
	if err != nil {
		return
	}
	tree, err = fromJsonTreeMetaData(jsonTree)
	return
}

// Metadata of one checkpoint entry (usually weights or embedding tables of model)
type Metadata struct {
	KeyPath []string

	KeyToKeyType    map[string]MetadataKeyType
	ValueType       string
	SkipDeserialize bool
}

// String implements fmt.Stringer.
func (m *Metadata) String() string {
	deserialize := ""
	if m.SkipDeserialize {
		deserialize = " [*]"
	}
	return fmt.Sprintf("%s%s", m.ValueType, deserialize)
}

type MetadataKeyType int

const (
	KeyTypeSequence MetadataKeyType = 1
	KeyTypeDict                     = 2
)

func fromJsonTreeMetaData(jsonTree any) (tree *trees.Tree[*Metadata], err error) {
	mapAny, ok := jsonTree.(map[string]any)
	if !ok {
		err = errors.Errorf("expected json to be a map of strings, got %T instead", jsonTree)
		return
	}
	_ = mapAny
	tree = trees.New[*Metadata]()
	for key, value := range mapAny {
		switch key {
		case KeyUseZarr3:
			// Check for Zarr3: not supported.
			zarr3, ok := value.(bool)
			if !ok {
				err = errors.Errorf("metadata json value for key %q is not a bool, got %T instead", key, value)
				return
			}
			if zarr3 {
				err = errors.Errorf("%q set to true, but Zarr3 is not supported by this library", key)
				return
			}
		case KeyTreeMetadata:
			entries, ok := value.(map[string]any)
			if !ok {
				err = errors.Errorf("metadata json value for key %q is not a map[string]any, got %T instead", key, value)
				return
			}
			for keyPath, jsonEntryAny := range entries {
				jsonEntry, ok := jsonEntryAny.(map[string]any)
				if !ok {
					err = errors.Errorf("metadata json value for key %q/%q is not a map[string]any, got %T instead", key, keyPath, jsonEntryAny)
					return
				}
				err = parseJsonMetadataEntry(tree, keyPath, jsonEntry)
				if err != nil {
					err = errors.WithMessagef(err, "metadata json value for key %q/%q", key, keyPath)
					return
				}
			}

		default:
			err = errors.Errorf("metadata json key %q unknown, don't know how to proceed", key)
			return
		}

	}
	return
}

func parseJsonMetadataEntry(tree *trees.Tree[*Metadata], keyPath string, jsonEntry map[string]any) error {
	entry := &Metadata{}
	if err := parseKeyPath(entry, keyPath); err != nil {
		return err
	}

	keyMetadataJsonAny, found := jsonEntry[KeyKeyMetadata]
	if !found {
		return errors.Errorf("missing KeyMetadata (key %q)", KeyKeyMetadata)
	}
	keyMetadataJson, ok := keyMetadataJsonAny.([]any)
	if !ok {
		return errors.Errorf("invalid KeyMetadata (key %q) type %T, expected []any", KeyKeyMetadata, keyMetadataJsonAny)
	}
	parseKeyMetadata(entry, keyMetadataJson)

	valueMetadataJsonAny, found := jsonEntry[KeyValueMetadata]
	if !found {
		return errors.Errorf("missing ValueMetadata (key %q)", KeyValueMetadata)
	}
	valueMetadataJson, ok := valueMetadataJsonAny.(map[string]any)
	if !ok {
		return errors.Errorf("invalid ValueMetadata (key %q) type %T, expected map[string]any", KeyValueMetadata, valueMetadataJsonAny)
	}
	parseValueMetadata(entry, valueMetadataJson)
	tree.Set(entry.KeyPath, entry)
	return nil
}

var reParseKeyPath = regexp.MustCompile(`'(.*?)'\s*[,)]`)

func parseKeyPath(metadata *Metadata, keyPathStr string) error {
	matches := reParseKeyPath.FindAllStringSubmatch(keyPathStr, -1)
	if len(matches) == 0 {
		return errors.Errorf("can't parse keypath from %q", keyPathStr)
	}
	for _, match := range matches {
		metadata.KeyPath = append(metadata.KeyPath, match[1])
	}
	return nil
}

func parseKeyMetadata(metadata *Metadata, keyMetadataJson []any) {
	if metadata.KeyToKeyType == nil {
		metadata.KeyToKeyType = make(map[string]MetadataKeyType)
	}
	for _, entryAny := range keyMetadataJson {
		entry := entryAny.(map[string]any)
		metadata.KeyToKeyType[entry["key"].(string)] = MetadataKeyType(entry["key_type"].(float64))
	}
}

func parseValueMetadata(metadata *Metadata, valueMetadataJson map[string]any) {
	metadata.ValueType = valueMetadataJson["value_type"].(string)
	metadata.SkipDeserialize = valueMetadataJson["skip_deserialize"].(bool)
}

// ParamNames convert metadata to the paramNames (?? not sure where in Gemma this is sued)
func ParamNames(metadata *trees.Tree[*Metadata]) *trees.Tree[string] {
	return trees.Map(metadata, func(treePath trees.Path, metadata *Metadata) string {
		return strings.Join(treePath, ".")
	})
}
