package weights

import "github.com/gomlx/gomlx/types/shapes"

// MetadataNode holds a metadata sub-tree.
// Each node either has a Map or an Entry, but not both.
type MetadataNode struct {
	Map   map[string]*MetadataNode
	Entry *MetadataEntry
}

// MetadataEntry holds information about one tensor.
type MetadataEntry struct {
	Name  string
	Shape shapes.Shape
}

// Metadata points to the root node of a metadata tree.
type Metadata struct {
	Root *MetadataNode

	// Directory where tha data is located.
	Directory string
}

// LoadMetadata returns the metadata loaded from the given directory in the form of a tree.
func LoadMetadata(dir string) (metadata *Metadata, err error) {
	
}
