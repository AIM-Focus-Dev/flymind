#!/usr/bin/env python3
import os
import sys
import pygame
import numpy as np
import pybullet as p
import pybullet_data
import logging
import argparse
from pathlib import Path
from math import sin, cos, atan2, degrees, radians

# --- Configuration ---
# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("EEGDrone3DSim")

# Project Path
DIR_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR_HERE))

# Simulation Parameters
ENVIRONMENT_BOUND = 5
TARGET_ALTITUDE_MAX = 10
TARGET_ALTITUDE_MIN = 0.5

# --- EEG Data and Predictor ---
try:
    from training.scripts.MindReaderModel.preprocess_data import load_and_preprocess_subject_data
except ImportError as e:
    logger.error(f"Error importing preprocess_data: {e}. Please ensure the module is in the Python path.")
    sys.exit(1)

predictor = None
predictor_available = False # Tracks if predictor was successfully loaded initially
try:
    from MindReaderService.mind_reader_predictor import MindReaderPredictor
    predictor_instance = MindReaderPredictor(project_root_path=str(DIR_HERE))
    if predictor_instance.is_ready:
        predictor = predictor_instance
        predictor_available = True
        logger.info("MindReaderPredictor is ready and available for use.")
    else:
        logger.warning("MindReaderPredictor loaded but is not ready. Will use ground-truth labels.")
except Exception as e:
    logger.warning(f"MindReaderPredictor import or initialization failed: {e}. Falling back to ground-truth labels.")

def load_eeg_data_and_commands(subject_id=1, session_type='T'):
    """Loads EEG epochs and maps them to drone commands based on subject and session."""
    data_path = DIR_HERE / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
    logger.info(f"Loading EEG data for Subject ID: {subject_id}, Session Type: {session_type}")
    try:
        epochs = load_and_preprocess_subject_data(subject_id, session_type.upper(), data_path)
    except Exception as e:
        logger.error(f"Failed to load or preprocess subject data for S{subject_id}{session_type.upper()}: {e}")
        sys.exit(1)

    if epochs is None:
        logger.error(f"No EEG epochs loaded for S{subject_id}{session_type.upper()}. Exiting.")
        sys.exit(1)

    X_data = epochs.get_data()
    y_labels = epochs.events[:, 2]
    event_map = epochs.event_id
    
    marker_values = sorted(event_map.values())
    index_map = {marker: i for i, marker in enumerate(marker_values)}
    
    class_names = [None] * len(marker_values)
    for name, marker in event_map.items():
        class_names[index_map[marker]] = name
        
    class_indices = np.array([index_map[val] for val in y_labels])
    
    action_map = {}
    expected_event_keys = {
        'left_hand': 'TURN_LEFT', 'right_hand': 'TURN_RIGHT',
        'feet': 'FORWARD', 'tongue': 'UP'
    }
    for i, name in enumerate(class_names):
        lower_name_parts = name.lower().split('/')
        mapped_action = 'HOVER'
        for part in lower_name_parts:
            if part in expected_event_keys:
                mapped_action = expected_event_keys[part]
                break
            elif 'left' in part: mapped_action = 'TURN_LEFT'
            elif 'right' in part: mapped_action = 'TURN_RIGHT'
            elif 'feet' in part or 'foot' in part: mapped_action = 'FORWARD'
            elif 'tongue' in part: mapped_action = 'UP'
        action_map[i] = mapped_action
            
    commands = [action_map[idx] for idx in class_indices]
    logger.info(f"Loaded {len(commands)} commands for S{subject_id}{session_type.upper()}. Event map: {event_map}, Action map: {action_map}")
    return X_data, commands, epochs

# --- PyBullet Drone Simulator ---
class BulletDroneSim:
    def __init__(self, main_view_width, main_view_height, onboard_view_width, onboard_view_height):
        self.physics_client_id = p.connect(p.DIRECT) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id)
        p.setTimeStep(1.0/240.0, physicsClientId=self.physics_client_id) 
        
        self.setup_environment()
        self.drone_id = self.load_drone_model()
        
        self.x, self.y, self.z = 0.0, 0.0, 1.0 
        self.yaw = 0.0 
        p.resetBasePositionAndOrientation(self.drone_id, [self.x, self.y, self.z],
                                          p.getQuaternionFromEuler([0, 0, self.yaw]),
                                          physicsClientId=self.physics_client_id)

        self.target_x, self.target_y, self.target_z = self.x, self.y, self.z
        self.target_yaw = self.yaw
        
        self.yaw_rate_dps = 60.0
        self.forward_speed_mps = 1.5
        self.altitude_speed_mps = 1.0
        self.position_smooth_factor = 4.0
        self.yaw_smooth_factor = 4.0
        
        self.main_view_width, self.main_view_height = main_view_width, main_view_height
        self.onboard_view_width, self.onboard_view_height = onboard_view_width, onboard_view_height
        logger.info("BulletDroneSim initialized.")

    def setup_environment(self):
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client_id)
        grid_color = [0.2, 0.2, 0.2]
        for i in range(-ENVIRONMENT_BOUND, ENVIRONMENT_BOUND + 1):
            p.addUserDebugLine([i, -ENVIRONMENT_BOUND, 0.01], [i, ENVIRONMENT_BOUND, 0.01], grid_color, 1, physicsClientId=self.physics_client_id)
            p.addUserDebugLine([-ENVIRONMENT_BOUND, i, 0.01], [ENVIRONMENT_BOUND, i, 0.01], grid_color, 1, physicsClientId=self.physics_client_id)

    def load_drone_model(self):
        try:
            drone_urdf_path = str(Path(pybullet_data.getDataPath()) / "quadrotor.urdf")
            drone = p.loadURDF(drone_urdf_path, [0, 0, 1], physicsClientId=self.physics_client_id)
            logger.info(f"Loaded drone model from: {drone_urdf_path}")
        except Exception as e:
            logger.warning(f"Failed to load quadrotor.urdf: {e}. Using a fallback box shape.")
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], physicsClientId=self.physics_client_id)
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], rgbaColor=[0.7, 0.7, 0.1, 1], physicsClientId=self.physics_client_id) 
            drone = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=collision_shape,
                                      baseVisualShapeIndex=visual_shape, basePosition=[0, 0, 1],
                                      physicsClientId=self.physics_client_id)
        return drone

    def reset_simulation(self):
        self.x, self.y, self.z = 0.0, 0.0, 1.0
        self.yaw = 0.0
        self.target_x, self.target_y, self.target_z = self.x, self.y, self.z
        self.target_yaw = self.yaw
        p.resetBasePositionAndOrientation(self.drone_id, [self.x, self.y, self.z],
                                          p.getQuaternionFromEuler([0, 0, self.yaw]),
                                          physicsClientId=self.physics_client_id)
        logger.info("Drone simulation reset.")

    def apply_command(self, command, command_delta_time):
        # Applies the given command to update the drone's target state.
        # command_delta_time is the duration this command influences target change (typically epoch duration).
        current_target_x, current_target_y, current_target_z = self.target_x, self.target_y, self.target_z
        current_target_yaw = self.target_yaw

        if command == 'TURN_LEFT': current_target_yaw += radians(self.yaw_rate_dps) * command_delta_time
        elif command == 'TURN_RIGHT': current_target_yaw -= radians(self.yaw_rate_dps) * command_delta_time
        elif command == 'FORWARD':
            current_target_x += self.forward_speed_mps * command_delta_time * cos(self.target_yaw)
            current_target_y += self.forward_speed_mps * command_delta_time * sin(self.target_yaw)
        elif command == 'UP': current_target_z += self.altitude_speed_mps * command_delta_time
        elif command == 'DOWN': current_target_z -= self.altitude_speed_mps * command_delta_time
        
        self.target_x = np.clip(current_target_x, -ENVIRONMENT_BOUND, ENVIRONMENT_BOUND)
        self.target_y = np.clip(current_target_y, -ENVIRONMENT_BOUND, ENVIRONMENT_BOUND)
        self.target_z = np.clip(current_target_z, TARGET_ALTITUDE_MIN, TARGET_ALTITUDE_MAX)
        self.target_yaw = current_target_yaw % (2 * np.pi)

    def step(self, frame_delta_time):
        # Smoothly interpolates drone's current state towards target and updates PyBullet.
        # frame_delta_time is the visual frame time.
        self.x += (self.target_x - self.x) * self.position_smooth_factor * frame_delta_time
        self.y += (self.target_y - self.y) * self.position_smooth_factor * frame_delta_time
        self.z += (self.target_z - self.z) * self.position_smooth_factor * frame_delta_time

        yaw_diff = (self.target_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi
        self.yaw += yaw_diff * self.yaw_smooth_factor * frame_delta_time
        self.yaw = self.yaw % (2 * np.pi)

        new_orientation_quat = p.getQuaternionFromEuler([0, 0, self.yaw])
        p.resetBasePositionAndOrientation(self.drone_id, [self.x, self.y, self.z], new_orientation_quat,
                                          physicsClientId=self.physics_client_id)
        p.stepSimulation(physicsClientId=self.physics_client_id)

    def get_third_person_view_image(self):
        camera_distance, camera_height_offset = 5, 3
        eye_x = self.x - camera_distance * cos(self.yaw)
        eye_y = self.y - camera_distance * sin(self.yaw)
        eye_z = self.z + camera_height_offset
        target_position = [self.x, self.y, self.z]
        
        view_matrix = p.computeViewMatrix([eye_x, eye_y, eye_z], target_position, [0,0,1], physicsClientId=self.physics_client_id)
        projection_matrix = p.computeProjectionMatrixFOV(60, self.main_view_width/self.main_view_height, 0.1, 100, physicsClientId=self.physics_client_id)
        _, _, rgba, _, _ = p.getCameraImage(self.main_view_width, self.main_view_height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physics_client_id)
        return np.reshape(rgba, (self.main_view_height, self.main_view_width, 4))[:, :, :3]

    def get_onboard_view_image(self):
        eye_position = [self.x, self.y, self.z + 0.1] 
        look_distance = 1.0 
        target_position = [self.x + look_distance * cos(self.yaw), self.y + look_distance * sin(self.yaw), self.z + 0.1]
        view_matrix = p.computeViewMatrix(eye_position, target_position, [0,0,1], physicsClientId=self.physics_client_id)
        projection_matrix = p.computeProjectionMatrixFOV(75, self.onboard_view_width/self.onboard_view_height, 0.05, 50, physicsClientId=self.physics_client_id)
        _, _, rgba, _, _ = p.getCameraImage(self.onboard_view_width, self.onboard_view_height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physics_client_id)
        return np.reshape(rgba, (self.onboard_view_height, self.onboard_view_width, 4))[:, :, :3]

    def close(self):
        p.disconnect(physicsClientId=self.physics_client_id)
        logger.info("Disconnected from PyBullet simulation.")

# --- Pygame UI (Google Dark Mode Inspired) ---
COLOR_BACKGROUND = (32, 33, 36)
COLOR_PANEL = (48, 49, 52)
COLOR_TEXT = (228, 229, 232)
COLOR_TEXT_SECONDARY = (154, 160, 166)
COLOR_TEXT_HIGHLIGHT = (138, 180, 248) # Google Blue
COLOR_BUTTON_NORMAL = (60, 64, 67)
COLOR_BUTTON_HOVER = (74, 78, 82)
COLOR_BUTTON_TEXT = (228, 229, 232)
COLOR_ACCENT_GREEN = (95, 180, 100)
COLOR_ACCENT_RED = (217, 79, 72)
COLOR_ACCENT_YELLOW = (253, 213, 87)
COLOR_BORDER = (94, 98, 102)
COLOR_ONBOARD_BORDER = COLOR_TEXT_HIGHLIGHT

def draw_button(surface, rect, text, font, base_color, hover_color, text_color, corner_radius=5, border_thickness=1, border_color=COLOR_BORDER, enabled=True):
    mouse_pos = pygame.mouse.get_pos()
    is_hovered = rect.collidepoint(mouse_pos) and enabled
    
    current_base_color = base_color if enabled else pygame.Color(base_color).lerp(COLOR_PANEL, 0.5) # Dim if disabled
    current_text_color = text_color if enabled else pygame.Color(text_color).lerp(COLOR_PANEL, 0.5) # Dim text if disabled
    
    final_color = hover_color if is_hovered else current_base_color

    # Ensure rect has positive width and height before drawing
    # Pygame's draw.rect can handle negative width/height by normalizing, but it's good practice.
    # However, if width/height are 0 or too small, text might not fit or look good.
    # For this fix, we assume the rect passed in is already sensible due to prior calculations.
    if rect.width > 0 and rect.height > 0:
        pygame.draw.rect(surface, final_color, rect, border_radius=corner_radius)
        if border_thickness > 0 and border_color:
            current_border_color = border_color if enabled else pygame.Color(border_color).lerp(COLOR_PANEL, 0.5)
            pygame.draw.rect(surface, current_border_color, rect, border_radius=corner_radius, width=border_thickness)
            
        text_surf = font.render(text, True, current_text_color)
        text_rect = text_surf.get_rect(center=rect.center)
        surface.blit(text_surf, text_rect)
    return is_hovered # Only true if enabled and hovered

def draw_text(surface, text, position, font, color=COLOR_TEXT, center_aligned=False, background_color=None):
    text_surf = font.render(text, True, color, background_color)
    text_rect = text_surf.get_rect()
    if center_aligned: text_rect.center = position
    else: text_rect.topleft = position
    surface.blit(text_surf, text_rect)
    return text_rect.bottom

def main(args):
    pygame.init()
    pygame.font.init()

    WIN_W, WIN_H = 1600, 900
    MAIN_VIEW_W, MAIN_VIEW_H = int(WIN_W * 2 / 3), WIN_H # Approx 1066
    ONBOARD_VIEW_W, ONBOARD_VIEW_H = 240, 180
    UI_PANEL_W = WIN_W - MAIN_VIEW_W # Approx 534

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"EEG Drone Sim - S{args.subject_id}{args.session_type.upper()}")
    clock = pygame.time.Clock()
    
    try:
        font_ui_normal = pygame.font.SysFont("Roboto", 17) 
        font_ui_large = pygame.font.SysFont("Roboto", 20) 
        font_ui_small = pygame.font.SysFont("Roboto", 14)
        font_ui_title = pygame.font.SysFont("Roboto", 24, bold=True)
    except pygame.error:
        logger.warning("Roboto font not found, using fallback.")
        font_ui_normal = pygame.font.SysFont(None, 22)
        font_ui_large = pygame.font.SysFont(None, 26)
        font_ui_small = pygame.font.SysFont(None, 20)
        font_ui_title = pygame.font.SysFont(None, 30, bold=True)


    eeg_data_X, flight_plan_commands, eeg_epochs = load_eeg_data_and_commands(args.subject_id, args.session_type)
    command_interval_duration = eeg_epochs.tmax - eeg_epochs.tmin 
    
    drone_sim = BulletDroneSim(MAIN_VIEW_W, MAIN_VIEW_H, ONBOARD_VIEW_W, ONBOARD_VIEW_H)

    # Simulation state
    current_command_idx = 0
    time_accumulator = 0.0
    simulation_running = False
    executed_commands_history = []
    last_prediction_confidence = None
    
    # Scoring state
    correct_predictions = 0
    total_predictions_made = 0

    # Control mode state
    global predictor_available 
    user_wants_predictor_mode = predictor_available 

    ui_panel_rect = pygame.Rect(MAIN_VIEW_W, 0, UI_PANEL_W, WIN_H)
    
    # Define button rects here so they are available for event handling
    # Their positions are relative to ui_x_start and current_y which are set before drawing
    # This means they will use the previous frame's positions for event checks if current_y changes,
    # but for this row of buttons, current_y is fixed after the title.
    # For more dynamic UIs, rects might need to be updated right before event check or use a different system.
    button_width, button_height = 120, 32
    # Placeholder rects, will be accurately defined in the drawing loop before use by event handler
    # This is a common Pygame pattern: define rects in draw, check in event (uses prev frame's rect)
    # For this specific issue, the problem was the *calculation* of one rect's width.
    run_pause_button_rect = pygame.Rect(0,0,0,0) 
    reset_button_rect = pygame.Rect(0,0,0,0)
    predictor_toggle_button_rect = pygame.Rect(0,0,0,0)

    is_active = True
    while is_active:
        frame_dt = clock.tick(60) / 1000.0

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: is_active = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: 
                # Button rects are defined in the drawing section below.
                # This check uses the rects from the *previous frame's* drawing.
                # This is generally fine if the button positions are stable.
                if run_pause_button_rect.collidepoint(event.pos):
                    simulation_running = not simulation_running
                    logger.info(f"Simulation {'Resumed' if simulation_running else 'Paused'}")
                elif reset_button_rect.collidepoint(event.pos):
                    logger.info("Reset button clicked.")
                    drone_sim.reset_simulation()
                    current_command_idx = 0
                    time_accumulator = 0.0
                    executed_commands_history = []
                    last_prediction_confidence = None
                    correct_predictions = 0 
                    total_predictions_made = 0 
                elif predictor_toggle_button_rect.collidepoint(event.pos) and predictor_available:
                    user_wants_predictor_mode = not user_wants_predictor_mode
                    logger.info(f"Control mode switched to: {'Predictor' if user_wants_predictor_mode else 'Ground Truth'}")


        # --- Simulation Logic ---
        if simulation_running:
            time_accumulator += frame_dt
            if time_accumulator >= command_interval_duration:
                time_accumulator -= command_interval_duration
                current_command_idx = (current_command_idx + 1) % len(flight_plan_commands)
                
                current_eeg_epoch_data = eeg_data_X[current_command_idx]
                actual_command = flight_plan_commands[current_command_idx]
                
                final_command = actual_command 
                is_prediction = False

                global predictor 
                if predictor_available and user_wants_predictor_mode and predictor:
                    predicted_cmd, confidence = predictor.predict_command(current_eeg_epoch_data)
                    is_prediction = True
                    total_predictions_made += 1
                    if predicted_cmd is None: 
                        final_command = actual_command 
                        last_prediction_confidence = 0.0
                        logger.warning(f"Predictor returned None. Using ground truth: '{actual_command}'")
                    else:
                        final_command = predicted_cmd
                        last_prediction_confidence = confidence
                        if final_command == actual_command:
                            correct_predictions += 1
                        logger.info(f"Epoch {current_command_idx}: Pred: '{final_command}' (Conf: {confidence:.2f}), Actual: '{actual_command}' -> {'CORRECT' if final_command == actual_command else 'WRONG'}")
                else: 
                    final_command = actual_command
                    last_prediction_confidence = None 
                    logger.info(f"Epoch {current_command_idx}: Using ground truth: '{final_command}' (Mode: {'GroundTruth' if not user_wants_predictor_mode else 'PredictorNA'})")
                
                executed_commands_history.append({
                    "cmd": final_command, 
                    "source": "Predictor" if is_prediction and user_wants_predictor_mode else "GroundTruth",
                    "correct": (final_command == actual_command) if is_prediction and user_wants_predictor_mode else None
                })
                if len(executed_commands_history) > 15: executed_commands_history.pop(0)
                
                drone_sim.apply_command(final_command, command_interval_duration) 
            
            drone_sim.step(frame_dt)

        # --- Rendering ---
        screen.fill(COLOR_BACKGROUND)
        tp_image_data = drone_sim.get_third_person_view_image()
        tp_surface = pygame.surfarray.make_surface(tp_image_data.swapaxes(0, 1))
        screen.blit(tp_surface, (0, 0))

        onboard_image_data = drone_sim.get_onboard_view_image()
        onboard_surface = pygame.surfarray.make_surface(onboard_image_data.swapaxes(0, 1))
        onboard_rect = pygame.Rect(MAIN_VIEW_W - ONBOARD_VIEW_W - 10, 10, ONBOARD_VIEW_W, ONBOARD_VIEW_H)
        screen.blit(onboard_surface, onboard_rect.topleft)
        pygame.draw.rect(screen, COLOR_ONBOARD_BORDER, onboard_rect, 2, border_radius=3)

        pygame.draw.rect(screen, COLOR_PANEL, ui_panel_rect)
        ui_x_start = MAIN_VIEW_W + 20 
        current_y = 20 # Y cursor for laying out UI elements

        # Main Title
        current_y = draw_text(screen, "EEG Drone Simulator", (ui_x_start, current_y), font_ui_title, COLOR_TEXT_HIGHLIGHT) + 5
        current_y = draw_text(screen, f"Subject {args.subject_id} | Session {args.session_type.upper()}", (ui_x_start, current_y), font_ui_small, COLOR_TEXT_SECONDARY) + 15
        
        # Control Buttons - Rects are defined here for the current frame
        # button_width, button_height are already defined before the loop
        run_pause_button_text = "PAUSE" if simulation_running else "RUN"
        run_pause_button_color = COLOR_ACCENT_RED if simulation_running else COLOR_ACCENT_GREEN
        run_pause_button_rect = pygame.Rect(ui_x_start, current_y, button_width, button_height) # Redefined for current frame
        draw_button(screen, run_pause_button_rect, run_pause_button_text, font_ui_normal, run_pause_button_color, 
                    pygame.Color(run_pause_button_color).lerp(COLOR_TEXT, 0.2), COLOR_BUTTON_TEXT, 
                    border_color=pygame.Color(run_pause_button_color).lerp(COLOR_BACKGROUND, 0.3))

        reset_button_rect = pygame.Rect(ui_x_start + button_width + 10, current_y, button_width, button_height) # Redefined for current frame
        draw_button(screen, reset_button_rect, "RESET", font_ui_normal, COLOR_BUTTON_NORMAL, COLOR_BUTTON_HOVER, COLOR_BUTTON_TEXT)
        
        # Corrected width calculation for the predictor toggle button
        x_pos_predictor_toggle = ui_x_start + (button_width + 10) * 2
        # Width is from its start to the right edge of the panel, minus right padding
        width_predictor_toggle = (MAIN_VIEW_W + UI_PANEL_W - 20) - x_pos_predictor_toggle 
        if width_predictor_toggle < 10: width_predictor_toggle = 10 # Ensure minimum width
        
        predictor_toggle_button_rect = pygame.Rect(x_pos_predictor_toggle, current_y, width_predictor_toggle, button_height) # Redefined for current frame
        toggle_text = "Use GroundTruth" if user_wants_predictor_mode else "Use Predictor"
        draw_button(screen, predictor_toggle_button_rect, toggle_text, font_ui_normal, COLOR_BUTTON_NORMAL, COLOR_BUTTON_HOVER, COLOR_BUTTON_TEXT, enabled=predictor_available)
        current_y += button_height + 15

        # Active Control Mode Display
        active_mode_text = "Predictor" if user_wants_predictor_mode and predictor_available else "Ground Truth"
        if not predictor_available: active_mode_text += " (Predictor N/A)"
        current_y = draw_text(screen, f"Control Mode: {active_mode_text}", (ui_x_start, current_y), font_ui_normal, COLOR_TEXT_HIGHLIGHT if user_wants_predictor_mode and predictor_available else COLOR_ACCENT_YELLOW ) + 10

        # Drone Status
        current_y = draw_text(screen, "Drone Status", (ui_x_start, current_y), font_ui_large, COLOR_TEXT) + 5
        draw_text(screen, f"Altitude: {drone_sim.z:.2f} m", (ui_x_start + 10, current_y), font_ui_normal, COLOR_TEXT_SECONDARY)
        current_y += 20
        current_cmd_display = executed_commands_history[-1]["cmd"] if executed_commands_history else "N/A"
        draw_text(screen, f"Last Command: {current_cmd_display}", (ui_x_start + 10, current_y), font_ui_normal, COLOR_TEXT_SECONDARY)
        current_y += 20
        if last_prediction_confidence is not None and user_wants_predictor_mode and predictor_available:
            conf_color = COLOR_ACCENT_GREEN if last_prediction_confidence > 0.7 else (COLOR_ACCENT_YELLOW if last_prediction_confidence > 0.4 else COLOR_ACCENT_RED)
            draw_text(screen, f"Confidence: {last_prediction_confidence:.2f}", (ui_x_start + 10, current_y), font_ui_normal, conf_color)
            current_y += 20
        current_y += 5

        # Scoring Display
        current_y = draw_text(screen, "Performance Score", (ui_x_start, current_y), font_ui_large, COLOR_TEXT) + 5
        draw_text(screen, f"Correct Predictions: {correct_predictions}", (ui_x_start + 10, current_y), font_ui_normal, COLOR_TEXT_SECONDARY)
        current_y += 20
        draw_text(screen, f"Total Predictions: {total_predictions_made}", (ui_x_start + 10, current_y), font_ui_normal, COLOR_TEXT_SECONDARY)
        current_y += 20
        accuracy = (correct_predictions / total_predictions_made * 100) if total_predictions_made > 0 else 0
        acc_color = COLOR_ACCENT_GREEN if accuracy > 70 else (COLOR_ACCENT_YELLOW if accuracy > 40 else COLOR_ACCENT_RED)
        draw_text(screen, f"Accuracy: {accuracy:.1f}%", (ui_x_start + 10, current_y), font_ui_normal, acc_color)
        current_y += 25
        
        # Upcoming Commands
        current_y = draw_text(screen, "Upcoming Plan", (ui_x_start, current_y), font_ui_large, COLOR_TEXT) + 5
        max_plan_display = 5
        for i in range(max_plan_display):
            idx_in_plan = (current_command_idx + i) % len(flight_plan_commands)
            cmd_text = f"{idx_in_plan + 1}. {flight_plan_commands[idx_in_plan]}"
            text_color = COLOR_TEXT_HIGHLIGHT if i == 0 else COLOR_TEXT_SECONDARY
            draw_text(screen, cmd_text, (ui_x_start + 10, current_y + i * 18), font_ui_small, text_color)
        current_y += max_plan_display * 18 + 10

        # Executed History
        current_y = draw_text(screen, "Executed History", (ui_x_start, current_y), font_ui_large, COLOR_TEXT) + 5
        for i, item in enumerate(reversed(executed_commands_history[-max_plan_display:])):
            cmd_text = f"{len(executed_commands_history) - i}. {item['cmd']} ({item['source']})"
            item_color = COLOR_TEXT_SECONDARY
            if item["source"] == "Predictor":
                if item["correct"] is True: item_color = COLOR_ACCENT_GREEN
                elif item["correct"] is False: item_color = COLOR_ACCENT_RED
            draw_text(screen, cmd_text, (ui_x_start + 10, current_y + i * 18), font_ui_small, item_color)
        
        pygame.display.flip()

    drone_sim.close()
    pygame.quit()
    logger.info("EEG Drone Simulator exited gracefully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EEG Drone 3D Simulator with Scoring")
    parser.add_argument('--subject_id', type=int, default=1, help='Subject ID (default: 1)')
    parser.add_argument('--session_type', type=str, default='T', choices=['T', 'E', 't', 'e'], help="Session type: 'T' or 'E' (default: 'T')")
    cli_args = parser.parse_args()
    cli_args.session_type = cli_args.session_type.upper()
    main(cli_args)
