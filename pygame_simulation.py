#!/usr/bin/env python3
"""
EEG Drone 3D Simulator
This script integrates EEG-based command prediction with a PyBullet drone simulation
and a Pygame user interface. 
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from math import sin, cos, atan2, radians

import numpy as np
import pygame
import pybullet as p
import pybullet_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Paths and simulation bounds
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ENV_BOUND = 5
ALTITUDE_MAX = 10
ALTITUDE_MIN = 0.5

# Attempt to import EEG preprocessing
try:
    from training.scripts.MindReaderModel.preprocess_data import load_and_preprocess_subject_data
except ImportError as e:
    logger.error(f"Could not import EEG preprocessing: {e}")
    sys.exit(1)

# Initialise predictor if available
predictor = None
predictor_available = False
try:
    from MindReaderService.mind_reader_predictor import MindReaderPredictor
    inst = MindReaderPredictor(project_root_path=str(HERE))
    if inst.is_ready:
        predictor = inst
        predictor_available = True
        logger.info("MindReaderPredictor is ready.")
    else:
        logger.warning("Predictor initialised but not ready; using ground truth.")
except Exception as e:
    logger.warning(f"Predictor unavailable: {e}")


def load_eeg_data_and_commands(subject_id=1, session_type='T'):
    """
    Load EEG epochs for a given subject and session, map events to drone commands.

    Returns:
        X_data (np.ndarray): Preprocessed EEG features.
        commands (list): Sequence of mapped commands.
        epochs: MNE Epochs object for timing information.
    """
    data_dir = HERE / 'training' / 'data' / 'MindReaderData' / 'BCICIV_2a_gdf'
    sess = session_type.upper()
    logger.info(f"Loading EEG data: Subject {subject_id}, Session {sess}")

    try:
        epochs = load_and_preprocess_subject_data(subject_id, sess, data_dir)
    except Exception as e:
        logger.error(f"Failed to load/preprocess data: {e}")
        sys.exit(1)

    if epochs is None:
        logger.error("No EEG epochs returned.")
        sys.exit(1)

    X_data = epochs.get_data()
    labels = epochs.events[:, 2]
    event_map = epochs.event_id

    # Map event labels to indices
    markers = sorted(event_map.values())
    index_map = {m: i for i, m in enumerate(markers)}

    class_names = [None] * len(markers)
    for name, marker in event_map.items():
        class_names[index_map[marker]] = name.lower()

    class_indices = np.array([index_map[lbl] for lbl in labels])

    # Define action mapping
    key_map = {
        'left_hand': 'TURN_LEFT', 'right_hand': 'TURN_RIGHT',
        'feet': 'FORWARD', 'tongue': 'UP'
    }
    action_map = {}
    for idx, name in enumerate(class_names):
        action = 'HOVER'
        for part in name.split('/'):
            if part in key_map:
                action = key_map[part]
                break
            elif 'left' in part:
                action = 'TURN_LEFT'
            elif 'right' in part:
                action = 'TURN_RIGHT'
            elif 'feet' in part or 'foot' in part:
                action = 'FORWARD'
            elif 'tongue' in part:
                action = 'UP'
        action_map[idx] = action

    commands = [action_map[i] for i in class_indices]
    logger.info(f"Loaded {len(commands)} commands; event map: {event_map}")
    return X_data, commands, epochs


class BulletDroneSim:
    """
    PyBullet-based drone simulation for visualisation and control.
    """

    def __init__(self, main_w, main_h, onboard_w, onboard_h):
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        p.setTimeStep(1/240, physicsClientId=self.client)

        self._create_plane_and_grid()
        self.drone = self._load_drone()

        self.x = self.y = 0.0
        self.z = 1.0
        self.yaw = 0.0
        p.resetBasePositionAndOrientation(
            self.drone,
            [self.x, self.y, self.z],
            p.getQuaternionFromEuler([0, 0, self.yaw]),
            physicsClientId=self.client
        )

        # Target state and motion parameters
        self.target = np.array([self.x, self.y, self.z], dtype=float)
        self.target_yaw = self.yaw
        self.yaw_rate = radians(60)
        self.speed_forward = 1.5
        self.speed_alt = 1.0
        self.smooth_pos = 4.0
        self.smooth_yaw = 4.0

        self.main_size = (main_w, main_h)
        self.onboard_size = (onboard_w, onboard_h)
        logger.info("BulletDroneSim initialised.")

    def _create_plane_and_grid(self):
        p.loadURDF('plane.urdf', physicsClientId=self.client)
        colour = [0.2] * 3
        for i in range(-ENV_BOUND, ENV_BOUND + 1):
            p.addUserDebugLine([i, -ENV_BOUND, 0.01], [i, ENV_BOUND, 0.01], colour, 1, physicsClientId=self.client)
            p.addUserDebugLine([-ENV_BOUND, i, 0.01], [ENV_BOUND, i, 0.01], colour, 1, physicsClientId=self.client)

    def _load_drone(self):
        """
        Attempt to load a quadrotor URDF; fallback to a simple box.
        """
        try:
            path = Path(pybullet_data.getDataPath()) / 'quadrotor.urdf'
            drone_id = p.loadURDF(str(path), [0, 0, 1], physicsClientId=self.client)
            logger.info(f"Loaded drone model: {path}")
            return drone_id
        except Exception as e:
            logger.warning(f"URDF load failed: {e}; using box model.")
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2,0.2,0.05], physicsClientId=self.client)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2,0.2,0.05], rgbaColor=[0.7,0.7,0.1,1], physicsClientId=self.client)
            return p.createMultiBody(0.5, col, vis, [0,0,1], physicsClientId=self.client)

    def reset(self):
        """Reset drone position and orientation to start state."""
        self.x, self.y, self.z, self.yaw = 0.0, 0.0, 1.0, 0.0
        self.target = np.array([self.x, self.y, self.z], dtype=float)
        self.target_yaw = self.yaw
        p.resetBasePositionAndOrientation(
            self.drone,
            [self.x, self.y, self.z],
            p.getQuaternionFromEuler([0, 0, self.yaw]),
            physicsClientId=self.client
        )
        logger.info("Simulation reset.")

    def apply_command(self, command, dt):
        """
        Update target state based on the issued command and time delta.

        command: one of 'TURN_LEFT', 'TURN_RIGHT', 'FORWARD', 'UP', 'DOWN', 'HOVER'
        dt: duration in seconds
        """
        tx, ty, tz = self.target
        tyaw = self.target_yaw

        if command == 'TURN_LEFT':
            tyaw += self.yaw_rate * dt
        elif command == 'TURN_RIGHT':
            tyaw -= self.yaw_rate * dt
        elif command == 'FORWARD':
            tx += self.speed_forward * dt * cos(tyaw)
            ty += self.speed_forward * dt * sin(tyaw)
        elif command == 'UP':
            tz += self.speed_alt * dt
        elif command == 'DOWN':
            tz -= self.speed_alt * dt

        # Clamp within bounds
        self.target = np.clip([tx, ty, tz], -ENV_BOUND, ENV_BOUND)
        self.target[2] = np.clip(self.target[2], ALTITUDE_MIN, ALTITUDE_MAX)
        self.target_yaw = tyaw % (2 * np.pi)

    def step(self, frame_dt):
        """Interpolate current state towards target and advance simulation."""
        # Position smoothing
        delta = self.target - np.array([self.x, self.y, self.z], dtype=float)
        self.x += delta[0] * self.smooth_pos * frame_dt
        self.y += delta[1] * self.smooth_pos * frame_dt
        self.z += delta[2] * self.smooth_pos * frame_dt

        # Yaw smoothing
        yaw_diff = ((self.target_yaw - self.yaw + np.pi) % (2*np.pi)) - np.pi
        self.yaw += yaw_diff * self.smooth_yaw * frame_dt
        self.yaw %= 2 * np.pi

        # Update in PyBullet
        quat = p.getQuaternionFromEuler([0, 0, self.yaw])
        p.resetBasePositionAndOrientation(
            self.drone,
            [self.x, self.y, self.z],
            quat,
            physicsClientId=self.client
        )
        p.stepSimulation(physicsClientId=self.client)

    def _render_view(self, size, fov, eye, target):
        """Helper to capture camera image from given eye and target positions."""
        view = p.computeViewMatrix(eye, target, [0, 0, 1], physicsClientId=self.client)
        proj = p.computeProjectionMatrixFOV(
            fov, size[0]/size[1], 0.1, 100,
            physicsClientId=self.client
        )
        _, _, rgba, _, _ = p.getCameraImage(
            size[0], size[1], view, proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.client
        )
        img = np.reshape(rgba, (size[1], size[0], 4))[:, :, :3]
        return img

    def get_third_person(self):
        """Return third-person view image as RGB array."""
        dist, height = 5, 3
        eye = [self.x - dist*cos(self.yaw), self.y - dist*sin(self.yaw), self.z + height]
        target = [self.x, self.y, self.z]
        return self._render_view(self.main_size, 60, eye, target)

    def get_onboard(self):
        """Return onboard (first-person) view image as RGB array."""
        eye = [self.x, self.y, self.z + 0.1]
        tgt = [
            self.x + cos(self.yaw),
            self.y + sin(self.yaw),
            self.z + 0.1
        ]
        return self._render_view(self.onboard_size, 75, eye, tgt)

    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect(physicsClientId=self.client)
        logger.info("PyBullet disconnected.")


# Pygame UI colours
COLOURS = {
    'bg': (32, 33, 36),
    'panel': (48, 49, 52),
    'text': (228, 229, 232),
    'secondary': (154, 160, 166),
    'highlight': (138, 180, 248),
    'button': (60, 64, 67),
    'hover': (74, 78, 82),
    'green': (95, 180, 100),
    'red': (217, 79, 72),
    'yellow': (253, 213, 87),
    'border': (94, 98, 102)
}


def draw_button(surface, rect, text, font, base_col, hover_col, txt_col, enabled=True):
    """Draw a button, return True if hovered and enabled."""
    mouse = pygame.mouse.get_pos()
    hovered = rect.collidepoint(mouse) and enabled
    colour = hover_col if hovered else base_col
    # Dim if disabled
    if not enabled:
        panel = COLOURS['panel']
        colour = tuple((pygame.Color(*base_col).lerp(panel, 0.5)))
        txt_col = tuple((pygame.Color(*txt_col).lerp(panel, 0.5)))

    pygame.draw.rect(surface, colour, rect, border_radius=5)
    pygame.draw.rect(surface, COLOURS['border'], rect, width=1, border_radius=5)
    txt_surf = font.render(text, True, txt_col)
    txt_rect = txt_surf.get_rect(center=rect.center)
    surface.blit(txt_surf, txt_rect)
    return hovered


def draw_text(surface, text, pos, font, col, centre=False, bg=None):
    """Draw text; return y-coordinate of bottom of rendered text."""
    surf = font.render(text, True, col, bg)
    rect = surf.get_rect()
    if centre:
        rect.center = pos
    else:
        rect.topleft = pos
    surface.blit(surf, rect)
    return rect.bottom


def main():
    """Entry point: initialise Pygame, load data, run simulation and UI loop."""
    parser = argparse.ArgumentParser(description="EEG Drone Simulator")
    parser.add_argument('--subject_id', type=int, default=1)
    parser.add_argument('--session_type', type=str, default='T', choices=['T','E','t','e'])
    args = parser.parse_args()
    args.session_type = args.session_type.upper()

    pygame.init()
    pygame.font.init()

    WIN_W, WIN_H = 1600, 900
    MAIN_W, MAIN_H = int(WIN_W*2/3), WIN_H
    ON_W, ON_H = 240, 180
    PANEL_W = WIN_W - MAIN_W

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"EEG Drone Sim - S{args.subject_id}{args.session_type}")
    clock = pygame.time.Clock()

    # Fonts with fallbacks
    try:
        font_sm = pygame.font.SysFont("Roboto", 14)
        font_md = pygame.font.SysFont("Roboto", 17)
        font_lg = pygame.font.SysFont("Roboto", 20)
        font_xl = pygame.font.SysFont("Roboto", 24, bold=True)
    except pygame.error:
        font_sm = pygame.font.SysFont(None, 20)
        font_md = pygame.font.SysFont(None, 22)
        font_lg = pygame.font.SysFont(None, 26)
        font_xl = pygame.font.SysFont(None, 30, bold=True)
        logger.warning("Fallback fonts used.")

    X_data, commands, epochs = load_eeg_data_and_commands(args.subject_id, args.session_type)
    interval = epochs.tmax - epochs.tmin
    sim = BulletDroneSim(MAIN_W, MAIN_H, ON_W, ON_H)

    idx = 0
    acc_time = 0.0
    running = False
    history = []
    correct = total = 0
    use_pred = predictor_available
    last_conf = None

    # Button rectangles (updated each frame)
    run_rect = reset_rect = toggle_rect = pygame.Rect(0,0,0,0)

    while True:
        dt = clock.tick(60)/1000.0
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                sim.close(); pygame.quit(); logger.info("Exited."); return
            if evt.type == pygame.MOUSEBUTTONDOWN and evt.button == 1:
                if run_rect.collidepoint(evt.pos):
                    running = not running
                if reset_rect.collidepoint(evt.pos):
                    sim.reset(); idx=0; acc_time=0; history.clear(); correct=total=0; last_conf=None
                if toggle_rect.collidepoint(evt.pos) and predictor_available:
                    use_pred = not use_pred

        if running:
            acc_time += dt
            if acc_time >= interval:
                acc_time -= interval
                idx = (idx + 1) % len(commands)
                eeg = X_data[idx]
                true_cmd = commands[idx]
                cmd = true_cmd
                predicted = False

                if predictor_available and use_pred and predictor:
                    pred, conf = predictor.predict_command(eeg)
                    predicted = True
                    total += 1
                    if pred and pred != true_cmd:
                        cmd = pred; last_conf = conf
                    elif pred == true_cmd:
                        cmd = pred; correct += 1; last_conf = conf
                    else:
                        last_conf = 0.0
                history.append((cmd, predicted, true_cmd == cmd if predicted else None))
                if len(history) > 15:
                    history.pop(0)
                sim.apply_command(cmd, interval)
            sim.step(dt)

        # Render views
        screen.fill(COLOURS['bg'])
        tp = pygame.surfarray.make_surface(sim.get_third_person().swapaxes(0,1))
        screen.blit(tp, (0,0))
        ob = pygame.surfarray.make_surface(sim.get_onboard().swapaxes(0,1))
        ob_rect = pygame.Rect(MAIN_W-ON_W-10, 10, ON_W, ON_H)
        screen.blit(ob, ob_rect.topleft)
        pygame.draw.rect(screen, COLOURS['highlight'], ob_rect, 2, border_radius=3)

        # UI panel
        panel = pygame.Rect(MAIN_W, 0, PANEL_W, WIN_H)
        pygame.draw.rect(screen, COLOURS['panel'], panel)
        x0, y0 = MAIN_W+20, 20

        y0 = draw_text(screen, "EEG Drone Simulator", (x0, y0), font_xl, COLOURS['highlight']) + 5
        y0 = draw_text(screen, f"Subject {args.subject_id} | Session {args.session_type}", (x0, y0), font_sm, COLOURS['secondary']) + 15

        # Control buttons
        texts = ["RUN" if not running else "PAUSE", "RESET", "Use Predictor" if not use_pred else "Use GroundTruth"]
        rects = []
        for i, txt in enumerate(texts):
            w = 120 if i < 2 else PANEL_W - (120+10)*2 - 20
            r = pygame.Rect(x0 + (120+10)*i, y0, w, 32)
            rects.append(r)
            hovered = draw_button(
                screen, r, txt, font_md,
                COLOURS['green'] if (i==0 and not running) else COLOURS['red'] if (i==0 and running) else COLOURS['button'],
                COLOURS['hover'], COLOURS['text'], enabled=(i!=2 or predictor_available)
            )
        run_rect, reset_rect, toggle_rect = rects
        y0 += 32 + 15

        # Display stats
        mode = "Predictor" if use_pred and predictor_available else "Ground Truth"
        y0 = draw_text(screen, f"Control Mode: {mode}", (x0, y0), font_md, COLOURS['highlight']) + 10
        y0 = draw_text(screen, "Performance", (x0, y0), font_lg, COLOURS['text']) + 5
        y0 = draw_text(screen, f"Correct: {correct}", (x0+10, y0), font_md, COLOURS['secondary']) + 20
        y0 = draw_text(screen, f"Total: {total}", (x0+10, y0), font_md, COLOURS['secondary']) + 20
        acc = (correct/total*100) if total else 0.0
        col = COLOURS['green'] if acc>70 else COLOURS['yellow'] if acc>40 else COLOURS['red']
        y0 = draw_text(screen, f"Accuracy: {acc:.1f}%", (x0+10, y0), font_md, col) + 25

        # Upcoming commands
        y0 = draw_text(screen, "Upcoming", (x0, y0), font_lg, COLOURS['text']) + 5
        for i in range(5):
            ci = (idx + i) % len(commands)
            txt = f"{ci+1}. {commands[ci]}"
            col = COLOURS['highlight'] if i==0 else COLOURS['secondary']
            y0 = draw_text(screen, txt, (x0+10, y0), font_sm, col) + 18
        y0 += 10

        # History
        y0 = draw_text(screen, "History", (x0, y0), font_lg, COLOURS['text']) + 5
        for i, (cmd, pred, correct_flag) in enumerate(reversed(history[-5:])):
            txt = f"{len(history)-i}. {cmd}"
            col = COLOURS['secondary']
            if pred:
                col = COLOURS['green'] if correct_flag else COLOURS['red']
            y0 = draw_text(screen, txt, (x0+10, y0), font_sm, col) + 18

        pygame.display.flip()


if __name__ == '__main__':
    main()
