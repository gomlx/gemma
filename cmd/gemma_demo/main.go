// gemma_demo uses Gemma for GoMLX to generate text given a prompt.
//
// It also uses github.com/charmbracelet libraries to make for a pretty command-line UI.
package main

import (
	"flag"
	"fmt"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/gomlx/exceptions"
	"os"
)

func main() {
	flag.Parse()

	var p *tea.Program
	err := exceptions.TryCatch[error](func() { p = tea.NewProgram(newUIModel()) })
	if err != nil {
		fmt.Fprintf(os.Stderr, "Alas, there's been an error: %+v", err)
		os.Exit(1)
	}
	_, err = p.Run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Alas, there's been an error: %+v", err)
		os.Exit(1)
	}
}
