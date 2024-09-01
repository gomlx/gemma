package main

import (
	"fmt"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/gemma/samplers"
)

type uiModel struct {
	textarea  textarea.Model
	viewport  viewport.Model
	submitted bool
	sampler   *samplers.Sampler
	err       error
}

func newUIModel() *uiModel {
	ta := textarea.New()
	ta.Placeholder = "Gemma Prompt:"
	ta.Focus()

	vp := viewport.New(0, 0)
	vp.Style = lipgloss.NewStyle().Margin(1, 2).
		Border(lipgloss.NormalBorder()).BorderForeground(lipgloss.Color("99"))

	return &uiModel{
		textarea: ta,
		viewport: vp,
		sampler:  BuildSampler(),
	}
}

func (m *uiModel) Init() tea.Cmd {
	return textarea.Blink
}

func (m *uiModel) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	var (
		taCmd  tea.Cmd
		vpCmd  tea.Cmd
		cmds   []tea.Cmd
		resize bool
	)

	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch {
		case msg.Type == tea.KeyCtrlC || msg.Type == tea.KeyEsc:
			return m, tea.Quit
		case msg.Type == tea.KeyCtrlL:
			m.textarea.Reset()

		case msg.Type == tea.KeyCtrlD && !m.submitted: // Ctrl+Enter to submit
			m.submitted = true
			generatedContent, err := m.Generate()
			if err != nil {
				m.err = err
				return m, tea.Quit
			}
			m.viewport.SetContent(generatedContent)
			m.textarea.Blur()
			m.textarea.SetValue(generatedContent)

		case m.submitted && msg.Type == tea.KeyEnter: // Enter while submitted to edit
			m.submitted = false
			m.textarea.Focus()
		}

	case tea.WindowSizeMsg:
		resize = true
		m.viewport.Width = msg.Width
		m.viewport.Height = msg.Height - 3 // Account for textarea and margins
		m.textarea.SetWidth(msg.Width - 4) // Account for textarea margins
		m.textarea.SetHeight(msg.Height - 8)
	}

	m.textarea, taCmd = m.textarea.Update(msg)
	m.viewport, vpCmd = m.viewport.Update(msg)

	if resize {
		cmds = append(cmds, vpCmd)
	}

	return m, tea.Batch(append(cmds, taCmd)...)
}

func (m *uiModel) Generate() (string, error) {
	outputs, err := m.sampler.Sample([]string{m.textarea.Value()})
	if err != nil {
		return "", err
	}
	return outputs[0], nil
}

func (m *uiModel) View() string {
	if m.submitted {
		return fmt.Sprintf("\n%s\n\nPress Enter to edit...", m.viewport.View())
	}

	return fmt.Sprintf(
		"\n%s\n\n"+
			"\t\u2022 Ctrl+C or ESC to quit;\n"+
			"\t• Ctrl+D to submit;\n"+
			"\t• Ctrl+L to clear the prompt.\n",
		m.textarea.View(),
	)
}
