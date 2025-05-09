#!/usr/bin/env python3
import os
import sys
import pygame
import numpy as np
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("EEGDroneSim")

# Add project root directory to sys.path to import local modules
dir_here = Path(__file__).resolve().parent
sys.path.insert(0, str(dir_here))

# EEG data preprocessing utility
try:
    from training.scripts.MindReaderModel.preprocess_data import load_and_preprocess_subject_data
except ImportError as e:
    logger.error(f"Error importing preprocess_data: {e}")
    sys.exit(1)

# Predictor
predictor = None
use_predictor = False
try:
    from MindReaderService.mind_reader_predictor import MindReaderPredictor
    predictor = MindReaderPredictor(project_root_path=str(dir_here))
    if predictor.is_ready:
        use_predictor = True
        logger.info("Using MindReaderPredictor for live inference.")
    else:
        predictor = None
        logger.warning("MindReaderPredictor not ready; using ground-truth labels.")
except Exception as e:
    logger.warning(f"Predictor import failed: {e}. Falling back to labels.")


def load_data(subject_id=8, session_type='T'):
    data_path = dir_here / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
    epochs = load_and_preprocess_subject_data(
        subject_id=subject_id,
        session_type=session_type,
        data_path=data_path
    )
    if epochs is None:
        logger.error("No EEG epochs loaded. Check data path.")
        sys.exit(1)

    X = epochs.get_data()
    y_mne = epochs.events[:, 2]
    event_id_map = epochs.event_id

    # Map event IDs to labels
    mids = sorted(event_id_map.values())
    idx_map = {mid: i for i, mid in enumerate(mids)}
    y_idx = np.array([idx_map[mid] for mid in y_mne])
    class_names = [None] * len(mids)
    for name, mid in event_id_map.items():
        class_names[idx_map[mid]] = name

    # EEG to drone action
    action_map = {}
    for i, name in enumerate(class_names):
        ln = name.lower()
        if 'left' in ln:
            action_map[i] = 'TURN_LEFT'
        elif 'right' in ln:
            action_map[i] = 'TURN_RIGHT'
        elif 'feet' in ln:
            action_map[i] = 'FORWARD'
        elif 'tongue' in ln:
            action_map[i] = 'UP'
        else:
            action_map[i] = 'HOVER'
    commands = [action_map[i] for i in y_idx]
    return X, commands, epochs

class DroneSimulator:
    def __init__(self, surf, x, y, w, h):
        self.surface = surf
        self.rect = pygame.Rect(x, y, w, h)
        self.x = x + w // 2
        self.y = y + h // 2
        self.altitude = 0              # in meters
        self.max_altitude = 10         # sim max height in meters
        self.speed = 200               # pixels per second
        self.size = 40
        # safety walls relative to sim rect
        self.walls = [
            pygame.Rect(x + 100, y + 75, w - 200, 20),
            pygame.Rect(x + 100, y + h - 95, w - 200, 20),
            pygame.Rect(x + 100, y + 75, 20, h - 150),
            pygame.Rect(x + w - 120, y + 75, 20, h - 150)
        ]

    def draw(self, cmd, conf=None):
        # Clear sim area
        pygame.draw.rect(self.surface, (30, 30, 30), self.rect)
        # Draw walls
        for w in self.walls:
            pygame.draw.rect(self.surface, (150, 30, 30), w)
        # Draw drone
        d = pygame.Rect(
            self.x - self.size / 2,
            self.y - self.size / 2,
            self.size, self.size
        )
        pygame.draw.rect(self.surface, (200, 200, 50), d)
        # Display status text
        font = pygame.font.SysFont(None, 24)
        # Command
        text = f"Cmd: {cmd}"
        if conf is not None:
            text += f" ({conf:.2f})"
        self.surface.blit(font.render(text, True, (255, 255, 255)), (self.rect.x + 10, self.rect.y + 10))
        # Altitude
        alt_text = f"Alt: {self.altitude} m"
        self.surface.blit(font.render(alt_text, True, (255, 255, 255)), (self.rect.x + 10, self.rect.y + 35))

    def safety_check(self, nx, ny, na=None):
        # Prevent collisions with walls or leaving bounds
        r = pygame.Rect(nx - self.size / 2, ny - self.size / 2, self.size, self.size)
        for w in self.walls:
            if r.colliderect(w):
                return False
        if not self.rect.collidepoint(nx, ny):
            return False
        # Altitude safety
        if na is not None and (na < 0 or na > self.max_altitude):
            return False
        return True

    def update(self, cmd, dt):
        dx = dy = 0
        na = self.altitude
        if cmd == 'TURN_LEFT':
            dx = -self.speed * dt
        elif cmd == 'TURN_RIGHT':
            dx = self.speed * dt
        elif cmd == 'FORWARD':
            dy = -self.speed * dt
        elif cmd == 'UP':
            na = self.altitude + 1  # 1 meter per command event
        nx, ny = self.x + dx, self.y + dy
        # Check movement commands
        if cmd in ('TURN_LEFT', 'TURN_RIGHT', 'FORWARD'):
            if not self.safety_check(nx, ny):
                logger.warning(f"Unsafe '{cmd}' blocked")
                return
            self.x, self.y = nx, ny
        # Check altitude command separately
        elif cmd == 'UP':
            if not self.safety_check(self.x, self.y, na):
                logger.warning("Unsafe 'UP' blocked: altitude limit reached")
                return
            self.altitude = na


def main():
    pygame.init()
    W, H = 1200, 800
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("EEG Drone Simulator")
    clock = pygame.time.Clock()

    # Layout: simulation and control panel
    sim_rect = pygame.Rect(0, 0, W * 2 // 3, H)
    panel_rect = pygame.Rect(W * 2 // 3, 0, W // 3, H)
    run_btn = pygame.Rect(panel_rect.x + 20, panel_rect.y + 20, 100, 40)

    X, plan, epochs = load_data()
    dur = epochs.tmax - epochs.tmin
    sim = DroneSimulator(screen, *sim_rect)

    idx = 0
    acc = 0.0
    running = False
    executed = []
    last_conf = None
    font = pygame.font.SysFont(None, 20)

    while True:
        dt = clock.tick(30) / 1000.0
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return
            if ev.type == pygame.MOUSEBUTTONDOWN and run_btn.collidepoint(ev.pos):
                running = not running

        if running:
            acc += dt
            if acc >= dur:
                acc -= dur
                idx = (idx + 1) % len(plan)
                epoch = X[idx]
                if use_predictor:
                    cmd, conf = predictor.predict_command(epoch)
                    if cmd is None:
                        cmd, conf = 'HOVER', 0.0
                    logger.info(f"Predicted '{cmd}' ({conf:.2f})")
                    last_conf = conf
                else:
                    cmd = plan[idx]
                    last_conf = None
                executed.append(cmd)
                sim.update(cmd, dt)

        # Display chosen command/conf
        if running and executed:
            disp_cmd = executed[-1]
            disp_conf = last_conf
        else:
            disp_cmd = plan[idx]
            disp_conf = None

        # Draw simulation and UI
        sim.draw(disp_cmd, disp_conf)
        pygame.draw.rect(screen, (50, 50, 50), panel_rect)
        # Run/Pause button
        clr = (50, 200, 50) if running else (200, 50, 50)
        pygame.draw.rect(screen, clr, run_btn)
        screen.blit(font.render('RUN' if not running else 'PAUSE', True, (0, 0, 0)), (run_btn.x + 20, run_btn.y + 10))
        # Flight plan
        y0 = run_btn.bottom + 20
        screen.blit(font.render('Flight Plan:', True, (255, 255, 255)), (panel_rect.x + 20, y0))
        for i, c in enumerate(plan[:15]):
            col = (255, 255, 255)
            if i == idx:
                col = (0, 255, 0)
            screen.blit(font.render(f"{i + 1}. {c}", True, col), (panel_rect.x + 20, y0 + 25 + i * 20))
        # Predictions
        y1 = y0 + 25 + 15 * 20 + 20
        screen.blit(font.render('Predictions:', True, (255, 255, 255)), (panel_rect.x + 20, y1))
        for i, c in enumerate(executed[-15:]):
            idx0 = len(executed) - len(executed[-15:]) + i + 1
            screen.blit(font.render(f"{idx0}. {c}", True, (200, 200, 0)), (panel_rect.x + 20, y1 + 25 + i * 20))

        pygame.display.flip()

if __name__ == '__main__':
    main()
