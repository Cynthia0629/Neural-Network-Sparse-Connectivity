#!/bin/bash
tmux kill-session

# Configure the first window for git
tmux new-session -d -s dev -n myWindow
tmux send-keys -t dev "cd work/epilepsy" Enter
tmux send-keys -t dev "module load emacs" Enter
tmux send-keys -t dev "module load git" Enter
tmux send-keys -t dev "zsh" Enter

# Split pane 1 vertical by 20%, load zsh and cd to epilepsy
tmux splitw -v -p 20
tmux send-keys "zsh" Enter
tmux send-keys "cd work/epilepsy" Enter
tmux send-keys "module load emacs" Enter

# Split pane 1 horizontal by 50%, load zsh
tmux selectp -t 0
tmux splitw -h -p 50
tmux send-keys "zsh" Enter
tmux send-keys "cd work/epilepsy" Enter
tmux send-keys "module load emacs" Enter


#Create a new window called scratch
#tmux new-window -t dev:1 -n scratch
#tmux send-keys -t dev:scratch "cd work/epilepsy" Enter
#tmux send-keys -t dev:scratch "zsh" Enter
#tmux send-keys -t dev:scratch "module load emacs" Enter

tmux attach -t dev
