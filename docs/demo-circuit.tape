Output "/tmp/neuronscope-demo/docs/demo-circuit.gif"
Set FontSize 16
Set Width 760
Set Height 260
Set Theme "Dracula"
Set WindowBar Colorful
Set WindowBarSize 36

Type "export PS1='$ '"
Enter
Sleep 300ms
Type "export HOME=/tmp/neuronscope-hf-home"
Enter
Sleep 300ms

Type "source /tmp/neuronscope-demo/.venv/bin/activate"
Enter
Sleep 500ms

Type `neuronscope circuit gpt2 "The capital of France is Paris. The capital of Japan is" --top-k 3`
Enter
Sleep 6s
