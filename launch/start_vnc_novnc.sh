#!/bin/bash

# Settings
VNC_DISPLAY_NUM=:1
SCREEN_RES="1280x720x24"
NOVNC_PORT=6080
VNC_PORT=5900
PASSWORD_FILE="$HOME/.vnc/passwd"
SESSION="vnc_novnc"

echo "🧹 Cleaning up old VNC, Xvfb, websockify, and ROS 2 processes..."

# Kill previous processes
pkill -f Xvfb
pkill -f x11vnc
pkill -f websockify
pkill -f ros2
pkill -f spawn_iris_gazebo.launch.py

# Kill old tmux session if it exists
tmux kill-session -t $SESSION 2>/dev/null
sleep 1

echo "🚀 Starting new tmux session: $SESSION"
tmux new-session -d -s $SESSION

# Pane 0: Xvfb (virtual display)
tmux send-keys -t $SESSION "Xvfb $VNC_DISPLAY_NUM -screen 0 $SCREEN_RES" C-m
sleep 1
tmux send-keys -t $SESSION "export DISPLAY=$VNC_DISPLAY_NUM" C-m

# Pane 1: x11vnc with password auth
tmux split-window -h -t $SESSION
if [ ! -f "$PASSWORD_FILE" ]; then
    mkdir -p ~/.vnc
    echo "🔐 Set a password for VNC access:"
    x11vnc -storepasswd $PASSWORD_FILE
fi
tmux send-keys -t $SESSION.1 "export DISPLAY=$VNC_DISPLAY_NUM && x11vnc -display $VNC_DISPLAY_NUM -rfbauth $PASSWORD_FILE -nopw -forever -shared" C-m

# Pane 2: websockify (noVNC server)
tmux split-window -v -t $SESSION.1
tmux send-keys -t $SESSION.2 "websockify --web=/usr/share/novnc/ $NOVNC_PORT localhost:$VNC_PORT" C-m

# Pane 3: ROS 2 launch
tmux split-window -v -t $SESSION.0
tmux send-keys -t $SESSION.3 "export DISPLAY=$VNC_DISPLAY_NUM && source /opt/ros/humble/setup.bash && cd /root/flymind_ws/launch && ros2 launch spawn_iris_gazebo.launch.py" C-m

# Attach to session
tmux select-pane -t $SESSION.3
tmux attach-session -t $SESSION

