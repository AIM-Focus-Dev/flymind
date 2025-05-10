#!/usr/bin/env python3
# Specifies the interpreter for the script.

import os
import sys
import pygame
import numpy as np
import pybullet as p  # Physics simulation library.
import pybullet_data  # Provides access to PyBullet's data files (e.g., URDFs).
import logging
import argparse     # For command-line argument parsing.
from pathlib import Path  # For object-oriented path manipulation.
from math import sin, cos, atan2, degrees, radians # Standard mathematical functions.

# --- Configuration ---
# Logging Setup: Configures basic logging for the application.
logging.basicConfig(
    level=logging.INFO,  # Sets the minimum logging level to INFO.
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s', # Defines log message format.
    datefmt='%Y-%m-%d %H:%M:%S' # Defines the date/time format for log messages.
)
logger = logging.getLogger("EEGDrone3DSim") # Creates a logger instance for this module.

# Project Path Setup: Ensures modules within the project can be imported.
DIR_HERE = Path(__file__).resolve().parent # Gets the directory where this script is located.
# Adds the script's directory to the Python path to facilitate local module imports.
if str(DIR_HERE) not in sys.path:
    sys.path.insert(0, str(DIR_HERE))


# Simulation Parameters
ENVIRONMENT_BOUND = 5       # Defines the half-width/depth of the square ground plane environment.
TARGET_ALTITUDE_MAX = 10    # Maximum permissible target altitude for the drone.
TARGET_ALTITUDE_MIN = 0.5   # Minimum permissible target altitude for the drone.

# --- EEG Data and Predictor Initialisation ---
# Attempt to import the EEG data preprocessing function.
try:
    from training.scripts.MindReaderModel.preprocess_data import load_and_preprocess_subject_data
except ImportError as e:
    logger.error(f"Error importing preprocess_data: {e}. Please ensure the module is in the Python path.")
    sys.exit(1) # Exits if essential data loading utility is missing.

# Initialise placeholder for the MindReaderPredictor.
predictor = None
predictor_available = False # Flag to track if the predictor was successfully loaded.

# Attempt to import and initialise the MindReaderPredictor.
# This allows the simulation to use a trained model for drone control if available.
try:
    from MindReaderService.mind_reader_predictor import MindReaderPredictor
    # Assumes the MindReaderService is in a directory relative to this script.
    # The project_root_path is explicitly passed to ensure correct path resolution within the predictor.
    predictor_instance = MindReaderPredictor(project_root_path=str(DIR_HERE.parent)) # Assuming script is in a subfolder of project root
    if predictor_instance.is_ready:
        predictor = predictor_instance
        predictor_available = True
        logger.info("MindReaderPredictor is ready and available for use.")
    else:
        logger.warning("MindReaderPredictor loaded but is not ready. Simulation will use ground-truth labels for control.")
except Exception as e:
    logger.warning(f"MindReaderPredictor import or initialisation failed: {e}. Falling back to ground-truth labels for control.")

def load_eeg_data_and_commands(subject_id=1, session_type='T'):
    """
    Loads preprocessed EEG epochs for a specified subject and session.
    It also maps the EEG event markers to corresponding drone control commands.

    Args:
        subject_id (int): The identifier for the subject whose data is to be loaded.
        session_type (str): The type of session ('T' for training, 'E' for evaluation).

    Returns:
        tuple: (X_data, commands, epochs)
               - X_data (np.ndarray): The EEG epoch data (epochs, channels, samples).
               - commands (list): A list of drone command strings corresponding to each epoch.
               - epochs (mne.Epochs): The loaded MNE Epochs object for further inspection if needed.
    """
    # Construct the path to the EEG data files.
    data_path = DIR_HERE / "training" / "data" / "MindReaderData" / "BCICIV_2a_gdf"
    logger.info(f"Loading EEG data for Subject ID: {subject_id}, Session Type: {session_type.upper()}")
    try:
        # Utilise the imported function to load and preprocess data.
        epochs = load_and_preprocess_subject_data(subject_id, session_type.upper(), data_path)
    except Exception as e:
        logger.error(f"Failed to load or preprocess subject data for S{subject_id}{session_type.upper()}: {e}")
        sys.exit(1) # Critical failure if data cannot be loaded.

    if epochs is None:
        logger.error(f"No EEG epochs loaded for S{subject_id}{session_type.upper()}. This may indicate missing data or a processing issue. Exiting.")
        sys.exit(1)

    X_data = epochs.get_data()  # Extracts EEG data as a NumPy array.
    y_labels = epochs.events[:, 2] # Extracts event markers from the MNE Epochs object.
    event_map = epochs.event_id   # Dictionary mapping event descriptions (e.g., 'left_hand') to marker values.
    
    # Create a consistent mapping from raw marker values to 0-indexed class indices.
    # This ensures that regardless of the raw marker values, classes are ordered consistently.
    marker_values_sorted = sorted(event_map.values())
    index_map = {marker: i for i, marker in enumerate(marker_values_sorted)}
    
    # Create a list of class names ordered by the new 0-indexed scheme.
    class_names_ordered = [None] * len(marker_values_sorted)
    for name, marker_value in event_map.items():
        class_names_ordered[index_map[marker_value]] = name
        
    # Convert original y_labels (raw event markers) to 0-indexed class indices.
    class_indices = np.array([index_map[val] for val in y_labels])
    
    # Define the mapping from 0-indexed class indices to drone action strings.
    # This allows for flexibility if event names in the data vary slightly.
    action_map = {}
    # Standard BCI command interpretations.
    expected_event_keywords = {
        'left_hand': 'TURN_LEFT', 'right_hand': 'TURN_RIGHT',
        'feet': 'FORWARD', 'tongue': 'UP'
    }
    for i, name in enumerate(class_names_ordered):
        lower_name_parts = name.lower().split('/') # Handle MNE's hierarchical event names.
        mapped_action = 'HOVER' # Default action.
        found_keyword = False
        for part in lower_name_parts:
            if part in expected_event_keywords:
                mapped_action = expected_event_keywords[part]
                found_keyword = True
                break
        if not found_keyword: # Fallback for partial matches if full keyword not found.
            for part in lower_name_parts:
                if 'left' in part: mapped_action = 'TURN_LEFT'; break
                elif 'right' in part: mapped_action = 'TURN_RIGHT'; break
                elif 'feet' in part or 'foot' in part: mapped_action = 'FORWARD'; break
                elif 'tongue' in part: mapped_action = 'UP'; break
        action_map[i] = mapped_action
            
    # Generate the final list of command strings corresponding to each EEG epoch.
    commands = [action_map[idx] for idx in class_indices]
    logger.info(f"Loaded {len(commands)} commands for S{subject_id}{session_type.upper()}. Original event map: {event_map}, Derived action map: {action_map}")
    return X_data, commands, epochs

# --- PyBullet Drone Simulator Class ---
class BulletDroneSim:
    """
    Manages the PyBullet physics simulation for the drone, including its environment,
    movement, and rendering of camera views.
    """
    def __init__(self, main_view_width, main_view_height, onboard_view_width, onboard_view_height):
        """
        Initialises the PyBullet simulation environment and the drone.

        Args:
            main_view_width (int): Width for the main third-person camera view.
            main_view_height (int): Height for the main third-person camera view.
            onboard_view_width (int): Width for the drone's onboard camera view.
            onboard_view_height (int): Height for the drone's onboard camera view.
        """
        # Connect to the PyBullet physics server. p.DIRECT creates a non-graphical simulation.
        # For debugging, p.GUI can be used.
        self.physics_client_id = p.connect(p.DIRECT) 
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.physics_client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client_id) # Standard Earth gravity.
        p.setTimeStep(1.0/240.0, physicsClientId=self.physics_client_id) # Set simulation timestep.
        
        self.setup_environment() # Load ground plane and visual markers.
        self.drone_id = self.load_drone_model() # Load the drone's URDF model.
        
        # Initial drone state variables.
        self.x, self.y, self.z = 0.0, 0.0, 1.0  # Current position (metres).
        self.yaw = 0.0  # Current yaw orientation (radians).
        # Set the drone's initial position and orientation in the simulation.
        p.resetBasePositionAndOrientation(self.drone_id, [self.x, self.y, self.z],
                                          p.getQuaternionFromEuler([0, 0, self.yaw]),
                                          physicsClientId=self.physics_client_id)

        # Target state variables for smooth movement.
        self.target_x, self.target_y, self.target_z = self.x, self.y, self.z
        self.target_yaw = self.yaw
        
        # Drone movement parameters.
        self.yaw_rate_dps = 60.0  # Yaw rate in degrees per second.
        self.forward_speed_mps = 1.5  # Forward speed in metres per second.
        self.altitude_speed_mps = 1.0  # Altitude change speed in metres per second.
        # Smoothing factors control how quickly the drone reaches its target state.
        self.position_smooth_factor = 4.0 # Higher values mean faster response.
        self.yaw_smooth_factor = 4.0

        # Store camera view dimensions.
        self.main_view_width, self.main_view_height = main_view_width, main_view_height
        self.onboard_view_width, self.onboard_view_height = onboard_view_width, onboard_view_height
        logger.info("BulletDroneSim initialised.")

    def setup_environment(self):
        """Sets up the simulation environment (ground plane and grid lines)."""
        p.loadURDF("plane.urdf", physicsClientId=self.physics_client_id) # Load a simple ground plane.
        # Add debug lines to create a visual grid on the ground plane.
        grid_colour = [0.2, 0.2, 0.2] # Dark grey grid lines.
        for i in range(-ENVIRONMENT_BOUND, ENVIRONMENT_BOUND + 1):
            p.addUserDebugLine([i, -ENVIRONMENT_BOUND, 0.01], [i, ENVIRONMENT_BOUND, 0.01], grid_colour, 1, physicsClientId=self.physics_client_id)
            p.addUserDebugLine([-ENVIRONMENT_BOUND, i, 0.01], [ENVIRONMENT_BOUND, i, 0.01], grid_colour, 1, physicsClientId=self.physics_client_id)

    def load_drone_model(self):
        """Loads the drone model from a URDF file, with a fallback to a simple box shape."""
        try:
            # Attempt to load the standard quadrotor URDF model.
            drone_urdf_path = str(Path(pybullet_data.getDataPath()) / "quadrotor.urdf")
            drone = p.loadURDF(drone_urdf_path, [0, 0, 1], physicsClientId=self.physics_client_id)
            logger.info(f"Loaded drone model from: {drone_urdf_path}")
        except Exception as e:
            # If URDF loading fails, create a simple box as a fallback drone model.
            logger.warning(f"Failed to load quadrotor.urdf: {e}. Using a fallback box shape for the drone.")
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], physicsClientId=self.physics_client_id)
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.05], rgbaColor=[0.7, 0.7, 0.1, 1], physicsClientId=self.physics_client_id) 
            drone = p.createMultiBody(baseMass=0.5, baseCollisionShapeIndex=collision_shape,
                                      baseVisualShapeIndex=visual_shape, basePosition=[0, 0, 1],
                                      physicsClientId=self.physics_client_id)
        return drone

    def reset_simulation(self):
        """Resets the drone to its initial position and state."""
        self.x, self.y, self.z = 0.0, 0.0, 1.0
        self.yaw = 0.0
        self.target_x, self.target_y, self.target_z = self.x, self.y, self.z
        self.target_yaw = self.yaw
        p.resetBasePositionAndOrientation(self.drone_id, [self.x, self.y, self.z],
                                          p.getQuaternionFromEuler([0, 0, self.yaw]),
                                          physicsClientId=self.physics_client_id)
        logger.info("Drone simulation reset to initial state.")

    def apply_command(self, command, command_delta_time):
        """
        Applies a given high-level command (e.g., 'FORWARD', 'TURN_LEFT') to update
        the drone's target state (position and yaw). The actual movement towards this
        target state occurs in the `step` method.

        Args:
            command (str): The drone command string.
            command_delta_time (float): The duration for which this command influences
                                        the target state change, typically the EEG epoch duration.
        """
        current_target_x, current_target_y, current_target_z = self.target_x, self.target_y, self.target_z
        current_target_yaw = self.target_yaw

        # Update target state based on the command.
        if command == 'TURN_LEFT': current_target_yaw += radians(self.yaw_rate_dps) * command_delta_time
        elif command == 'TURN_RIGHT': current_target_yaw -= radians(self.yaw_rate_dps) * command_delta_time
        elif command == 'FORWARD':
            # Move forward along the current target yaw direction.
            current_target_x += self.forward_speed_mps * command_delta_time * cos(self.target_yaw)
            current_target_y += self.forward_speed_mps * command_delta_time * sin(self.target_yaw)
        elif command == 'UP': current_target_z += self.altitude_speed_mps * command_delta_time
        elif command == 'DOWN': current_target_z -= self.altitude_speed_mps * command_delta_time
        # 'HOVER' command implies no change to the target state.
        
        # Clip target positions and altitude to remain within defined simulation bounds.
        self.target_x = np.clip(current_target_x, -ENVIRONMENT_BOUND, ENVIRONMENT_BOUND)
        self.target_y = np.clip(current_target_y, -ENVIRONMENT_BOUND, ENVIRONMENT_BOUND)
        self.target_z = np.clip(current_target_z, TARGET_ALTITUDE_MIN, TARGET_ALTITUDE_MAX)
        self.target_yaw = current_target_yaw % (2 * np.pi) # Normalise yaw to [0, 2*pi).

    def step(self, frame_delta_time):
        """
        Advances the simulation by one step. It smoothly interpolates the drone's
        current physical state (position, yaw) towards its target state and updates
        the PyBullet simulation.

        Args:
            frame_delta_time (float): The time elapsed since the last visual frame (Pygame frame time).
                                     This is used for smooth interpolation.
        """
        # Interpolate current position towards target position.
        self.x += (self.target_x - self.x) * self.position_smooth_factor * frame_delta_time
        self.y += (self.target_y - self.y) * self.position_smooth_factor * frame_delta_time
        self.z += (self.target_z - self.z) * self.position_smooth_factor * frame_delta_time

        # Interpolate current yaw towards target yaw, handling wrap-around correctly.
        yaw_diff = (self.target_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi # Shortest angle.
        self.yaw += yaw_diff * self.yaw_smooth_factor * frame_delta_time
        self.yaw = self.yaw % (2 * np.pi) # Normalise yaw.

        # Update the drone's physical state in the PyBullet simulation.
        new_orientation_quat = p.getQuaternionFromEuler([0, 0, self.yaw])
        p.resetBasePositionAndOrientation(self.drone_id, [self.x, self.y, self.z], new_orientation_quat,
                                          physicsClientId=self.physics_client_id)
        p.stepSimulation(physicsClientId=self.physics_client_id) # Advance PyBullet's internal simulation time.

    def get_third_person_view_image(self):
        """Renders and returns the third-person camera view of the drone."""
        # Camera positioning relative to the drone for a chase-cam effect.
        camera_distance, camera_height_offset = 5, 3
        eye_x = self.x - camera_distance * cos(self.yaw) # Camera position behind the drone.
        eye_y = self.y - camera_distance * sin(self.yaw)
        eye_z = self.z + camera_height_offset
        target_position = [self.x, self.y, self.z] # Camera looks at the drone's centre.
        
        view_matrix = p.computeViewMatrix([eye_x, eye_y, eye_z], target_position, [0,0,1], physicsClientId=self.physics_client_id)
        projection_matrix = p.computeProjectionMatrixFOV(60, self.main_view_width/self.main_view_height, 0.1, 100, physicsClientId=self.physics_client_id)
        # Retrieve the camera image from PyBullet.
        _, _, rgba, _, _ = p.getCameraImage(self.main_view_width, self.main_view_height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physics_client_id)
        # Reshape and return RGB data (discarding alpha channel).
        return np.reshape(rgba, (self.main_view_height, self.main_view_width, 4))[:, :, :3]

    def get_onboard_view_image(self):
        """Renders and returns the drone's onboard (first-person) camera view."""
        eye_position = [self.x, self.y, self.z + 0.1] # Camera slightly above drone's centre.
        look_distance = 1.0 # How far the camera looks ahead.
        # Target position in front of the drone, along its current yaw.
        target_position = [self.x + look_distance * cos(self.yaw), 
                           self.y + look_distance * sin(self.yaw), 
                           self.z + 0.1]
        view_matrix = p.computeViewMatrix(eye_position, target_position, [0,0,1], physicsClientId=self.physics_client_id)
        projection_matrix = p.computeProjectionMatrixFOV(75, self.onboard_view_width/self.onboard_view_height, 0.05, 50, physicsClientId=self.physics_client_id)
        _, _, rgba, _, _ = p.getCameraImage(self.onboard_view_width, self.onboard_view_height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physics_client_id)
        return np.reshape(rgba, (self.onboard_view_height, self.onboard_view_width, 4))[:, :, :3]

    def close(self):
        """Disconnects from the PyBullet simulation."""
        p.disconnect(physicsClientId=self.physics_client_id)
        logger.info("Disconnected from PyBullet simulation.")

# --- Pygame UI (Inspired by Google Dark Mode Theme) ---
# Colour palette definition for the UI elements.
COLOR_BACKGROUND = (32, 33, 36)       # Dark grey, almost black.
COLOR_PANEL = (48, 49, 52)          # Slightly lighter grey for panels.
COLOR_TEXT = (228, 229, 232)        # Light grey/off-white for primary text.
COLOR_TEXT_SECONDARY = (154, 160, 166) # Medium grey for secondary text.
COLOR_TEXT_HIGHLIGHT = (138, 180, 248) # Google's signature blue for highlights.
COLOR_BUTTON_NORMAL = (60, 64, 67)    # Dark grey for buttons.
COLOR_BUTTON_HOVER = (74, 78, 82)     # Lighter grey for button hover state.
COLOR_BUTTON_TEXT = (228, 229, 232)   # Light text on buttons.
COLOR_ACCENT_GREEN = (95, 180, 100)   # Green for positive feedback/status.
COLOR_ACCENT_RED = (217, 79, 72)      # Red for warnings/negative feedback.
COLOR_ACCENT_YELLOW = (253, 213, 87)  # Yellow for neutral/cautionary feedback.
COLOR_BORDER = (94, 98, 102)          # Grey for borders.
COLOR_ONBOARD_BORDER = COLOR_TEXT_HIGHLIGHT # Blue border for the onboard camera view.

def draw_button(surface, rect, text, font, base_colour, hover_colour, text_colour, corner_radius=5, border_thickness=1, border_colour=COLOR_BORDER, enabled=True):
    """
    Draws a UI button with text, hover effects, and an enabled/disabled state.

    Args:
        surface (pygame.Surface): The Pygame surface to draw on.
        rect (pygame.Rect): The rectangular area for the button.
        text (str): The text to display on the button.
        font (pygame.font.Font): The font used for the button text.
        base_colour (tuple): The button's normal background colour.
        hover_colour (tuple): The button's background colour when hovered.
        text_colour (tuple): The colour of the button's text.
        corner_radius (int): Radius for rounded corners.
        border_thickness (int): Thickness of the button border.
        border_colour (tuple): Colour of the button border.
        enabled (bool): If False, the button is drawn in a dimmed state and is non-interactive.

    Returns:
        bool: True if the mouse is hovering over the button and it is enabled, False otherwise.
    """
    mouse_pos = pygame.mouse.get_pos()
    is_hovered = rect.collidepoint(mouse_pos) and enabled
    
    # Adjust colours if the button is disabled.
    current_base_colour = base_colour if enabled else pygame.Color(base_colour).lerp(COLOR_PANEL, 0.5)
    current_text_colour = text_colour if enabled else pygame.Color(text_colour).lerp(COLOR_PANEL, 0.5)
    
    final_colour = hover_colour if is_hovered else current_base_colour

    if rect.width > 0 and rect.height > 0: # Ensure valid dimensions before drawing.
        pygame.draw.rect(surface, final_colour, rect, border_radius=corner_radius)
        if border_thickness > 0 and border_colour:
            current_border_colour = border_colour if enabled else pygame.Color(border_colour).lerp(COLOR_PANEL, 0.5)
            pygame.draw.rect(surface, current_border_colour, rect, border_radius=corner_radius, width=border_thickness)
            
        text_surf = font.render(text, True, current_text_colour)
        text_rect = text_surf.get_rect(center=rect.center)
        surface.blit(text_surf, text_rect)
    return is_hovered # Interaction state depends on being enabled.

def draw_text(surface, text, position, font, colour=COLOR_TEXT, center_aligned=False, background_colour=None):
    """
    Renders and draws text on a surface.

    Args:
        surface (pygame.Surface): The Pygame surface to draw on.
        text (str): The text string to render.
        position (tuple): (x, y) coordinates for the text.
        font (pygame.font.Font): The font to use.
        colour (tuple, optional): Text colour. Defaults to COLOR_TEXT.
        center_aligned (bool, optional): If True, position is the center of the text. 
                                         Otherwise, it's the top-left. Defaults to False.
        background_colour (tuple, optional): Background colour for the text. Defaults to None (transparent).

    Returns:
        int: The y-coordinate of the bottom of the rendered text, useful for layout.
    """
    text_surf = font.render(text, True, colour, background_colour)
    text_rect = text_surf.get_rect()
    if center_aligned: text_rect.center = position
    else: text_rect.topleft = position
    surface.blit(text_surf, text_rect)
    return text_rect.bottom # Return bottom y-coordinate for sequential UI element placement.

# --- Main Application Logic ---
def main(args):
    """
    Main function to run the EEG Drone Simulator application.
    Initialises Pygame, PyBullet, loads data, and runs the main simulation loop.

    Args:
        args (argparse.Namespace): Command-line arguments (subject_id, session_type).
    """
    pygame.init()       # Initialise all Pygame modules.
    pygame.font.init()  # Initialise the font module.

    # Define window and view dimensions.
    WIN_W, WIN_H = 1600, 900  # Main window width and height.
    # Main simulation view takes up 2/3 of the window width.
    MAIN_VIEW_W, MAIN_VIEW_H = int(WIN_W * 2 / 3), WIN_H 
    ONBOARD_VIEW_W, ONBOARD_VIEW_H = 240, 180 # Dimensions for the smaller onboard camera view.
    UI_PANEL_W = WIN_W - MAIN_VIEW_W # Width of the UI panel on the right.

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"EEG Drone Sim - S{args.subject_id}{args.session_type.upper()} - Control: {'Predictor' if predictor_available else 'GroundTruth'}")
    clock = pygame.time.Clock() # Pygame clock for controlling frame rate.
    
    # Attempt to load preferred 'Roboto' font, with fallback to system default.
    try:
        font_ui_normal = pygame.font.SysFont("Roboto", 17) 
        font_ui_large = pygame.font.SysFont("Roboto", 20) 
        font_ui_small = pygame.font.SysFont("Roboto", 14)
        font_ui_title = pygame.font.SysFont("Roboto", 24, bold=True)
    except pygame.error:
        logger.warning("Roboto font not found. Using system default font, UI appearance may differ.")
        font_ui_normal = pygame.font.SysFont(None, 22) # None uses Pygame's default font.
        font_ui_large = pygame.font.SysFont(None, 26)
        font_ui_small = pygame.font.SysFont(None, 20)
        font_ui_title = pygame.font.SysFont(None, 30, bold=True)

    # Load EEG data and the corresponding sequence of commands (flight plan).
    eeg_data_X, flight_plan_commands, eeg_epochs = load_eeg_data_and_commands(args.subject_id, args.session_type)
    # Duration of each command, derived from the EEG epoch length.
    command_interval_duration = eeg_epochs.tmax - eeg_epochs.tmin 
    
    # Initialise the PyBullet drone simulation.
    drone_sim = BulletDroneSim(MAIN_VIEW_W, MAIN_VIEW_H, ONBOARD_VIEW_W, ONBOARD_VIEW_H)

    # Simulation state variables.
    current_command_idx = 0         # Index of the current command in the flight plan.
    time_accumulator = 0.0          # Accumulates time to trigger next command.
    simulation_running = False      # Flag to control pause/resume of the simulation.
    executed_commands_history = []  # Stores recently executed commands for display.
    last_prediction_confidence = None # Stores confidence of the last prediction.
    
    # Scoring state variables (for when predictor is active).
    correct_predictions = 0
    total_predictions_made = 0

    # Control mode state (True for predictor, False for ground truth).
    global predictor_available # Use the global flag determined during initialisation.
    user_wants_predictor_mode = predictor_available # Default to predictor if available.

    ui_panel_rect = pygame.Rect(MAIN_VIEW_W, 0, UI_PANEL_W, WIN_H) # Rectangle for the UI panel.
    
    # Define button rects. These are updated each frame in the drawing section
    # before being used by the event handler. This is a common Pygame pattern.
    button_width, button_height = 120, 32
    run_pause_button_rect = pygame.Rect(0,0,0,0) # Placeholder, defined in draw loop.
    reset_button_rect = pygame.Rect(0,0,0,0)     # Placeholder, defined in draw loop.
    predictor_toggle_button_rect = pygame.Rect(0,0,0,0) # Placeholder, defined in draw loop.

    is_active = True # Main loop control flag.
    while is_active:
        frame_dt = clock.tick(60) / 1000.0 # Delta time in seconds for the last frame, capped at 60 FPS.

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT: is_active = False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left mouse click.
                # Button collision checks use rects defined in the *previous* frame's drawing phase.
                # This is generally acceptable if button positions are relatively stable.
                if run_pause_button_rect.collidepoint(event.pos):
                    simulation_running = not simulation_running
                    logger.info(f"Simulation {'Resumed' if simulation_running else 'Paused'}")
                elif reset_button_rect.collidepoint(event.pos):
                    logger.info("Reset button clicked. Resetting simulation and flight plan.")
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
                    # Reset score when switching modes to avoid confusion.
                    correct_predictions = 0
                    total_predictions_made = 0
                    last_prediction_confidence = None


        # --- Simulation Logic ---
        if simulation_running:
            time_accumulator += frame_dt
            # Check if it's time to issue the next command from the flight plan.
            if time_accumulator >= command_interval_duration:
                time_accumulator -= command_interval_duration # Reset accumulator for next interval.
                
                # Fetch current EEG epoch and the corresponding ground-truth command.
                current_eeg_epoch_data = eeg_data_X[current_command_idx]
                actual_command = flight_plan_commands[current_command_idx]
                
                final_command_to_drone = actual_command 
                is_prediction_attempted = False
                current_confidence = None

                global predictor # Access the globally initialised predictor.
                # If predictor is available and user has selected predictor mode:
                if predictor_available and user_wants_predictor_mode and predictor:
                    predicted_cmd_from_model, confidence_from_model = predictor.predict_command(current_eeg_epoch_data)
                    is_prediction_attempted = True
                    total_predictions_made += 1
                    current_confidence = confidence_from_model

                    if predicted_cmd_from_model is None: 
                        # Predictor failed or returned None; fall back to ground truth for this step.
                        final_command_to_drone = actual_command 
                        logger.warning(f"Predictor returned None for epoch {current_command_idx}. Using ground truth: '{actual_command}'")
                    else:
                        final_command_to_drone = predicted_cmd_from_model
                        if final_command_to_drone == actual_command:
                            correct_predictions += 1
                        logger.info(f"Epoch {current_command_idx}: Pred: '{final_command_to_drone}' (Conf: {confidence_from_model:.2f}), Actual: '{actual_command}' -> {'CORRECT' if final_command_to_drone == actual_command else 'INCORRECT'}")
                else: 
                    # Using ground truth (either predictor not available, or user selected ground truth mode).
                    final_command_to_drone = actual_command
                    current_confidence = None # No prediction confidence in ground truth mode.
                    log_mode_reason = "GroundTruthMode" if not user_wants_predictor_mode else "PredictorNotAvailable"
                    logger.info(f"Epoch {current_command_idx}: Using ground truth: '{final_command_to_drone}' (Mode: {log_mode_reason})")
                
                last_prediction_confidence = current_confidence # Store for UI display.

                # Update command history for UI display.
                executed_commands_history.append({
                    "cmd": final_command_to_drone, 
                    "source": "Predictor" if is_prediction_attempted and user_wants_predictor_mode else "GroundTruth",
                    "correct": (final_command_to_drone == actual_command) if is_prediction_attempted and user_wants_predictor_mode else None # None if not a prediction.
                })
                if len(executed_commands_history) > 15: executed_commands_history.pop(0) # Keep history to a manageable size.
                
                # Apply the determined command to the drone simulation.
                drone_sim.apply_command(final_command_to_drone, command_interval_duration) 
                
                # Advance to the next command in the flight plan, looping if necessary.
                current_command_idx = (current_command_idx + 1) % len(flight_plan_commands)
            
            # Step the PyBullet simulation physics and drone movement.
            drone_sim.step(frame_dt)

        # --- Rendering ---
        screen.fill(COLOR_BACKGROUND) # Clear screen with background colour.
        
        # Render third-person view from PyBullet.
        tp_image_data = drone_sim.get_third_person_view_image()
        # PyBullet returns images that may need axes swapped for Pygame's surface format.
        tp_surface = pygame.surfarray.make_surface(tp_image_data.swapaxes(0, 1))
        screen.blit(tp_surface, (0, 0))

        # Render onboard camera view.
        onboard_image_data = drone_sim.get_onboard_view_image()
        onboard_surface = pygame.surfarray.make_surface(onboard_image_data.swapaxes(0, 1))
        # Position onboard view in the top-right corner of the main view area.
        onboard_rect = pygame.Rect(MAIN_VIEW_W - ONBOARD_VIEW_W - 10, 10, ONBOARD_VIEW_W, ONBOARD_VIEW_H)
        screen.blit(onboard_surface, onboard_rect.topleft)
        pygame.draw.rect(screen, COLOR_ONBOARD_BORDER, onboard_rect, 2, border_radius=3) # Add a border.

        # Draw the UI panel on the right.
        pygame.draw.rect(screen, COLOR_PANEL, ui_panel_rect)
        ui_x_start = MAIN_VIEW_W + 20 # Starting x-coordinate for UI elements, with padding.
        current_y = 20 # Vertical cursor for laying out UI elements sequentially.

        # UI Panel: Title
        current_y = draw_text(screen, "EEG Drone Simulator", (ui_x_start, current_y), font_ui_title, COLOR_TEXT_HIGHLIGHT) + 5
        current_y = draw_text(screen, f"Subject {args.subject_id} | Session {args.session_type.upper()}", (ui_x_start, current_y), font_ui_small, COLOR_TEXT_SECONDARY) + 15
        
        # UI Panel: Control Buttons
        # Button rects are redefined here for the current frame's layout.
        # This ensures accurate collision detection in the next frame's event loop.
        run_pause_button_text = "PAUSE" if simulation_running else "RUN"
        run_pause_button_colour = COLOR_ACCENT_RED if simulation_running else COLOR_ACCENT_GREEN
        run_pause_button_rect = pygame.Rect(ui_x_start, current_y, button_width, button_height)
        draw_button(screen, run_pause_button_rect, run_pause_button_text, font_ui_normal, run_pause_button_colour, 
                    pygame.Color(run_pause_button_colour).lerp(COLOR_TEXT, 0.2), COLOR_BUTTON_TEXT, 
                    border_colour=pygame.Color(run_pause_button_colour).lerp(COLOR_BACKGROUND, 0.3))

        reset_button_rect = pygame.Rect(ui_x_start + button_width + 10, current_y, button_width, button_height)
        draw_button(screen, reset_button_rect, "RESET", font_ui_normal, COLOR_BUTTON_NORMAL, COLOR_BUTTON_HOVER, COLOR_BUTTON_TEXT)
        
        # Predictor toggle button: dynamically calculates width to fill remaining space.
        x_pos_predictor_toggle = ui_x_start + (button_width + 10) * 2
        # Width is from its start to the right edge of the panel, minus right padding.
        width_predictor_toggle = (ui_panel_rect.right - 20) - x_pos_predictor_toggle 
        if width_predictor_toggle < 50: width_predictor_toggle = 50 # Ensure a minimum sensible width.
        
        predictor_toggle_button_rect = pygame.Rect(x_pos_predictor_toggle, current_y, width_predictor_toggle, button_height)
        toggle_text = "Use GroundTruth" if user_wants_predictor_mode else "Use Predictor"
        draw_button(screen, predictor_toggle_button_rect, toggle_text, font_ui_normal, COLOR_BUTTON_NORMAL, COLOR_BUTTON_HOVER, COLOR_BUTTON_TEXT, enabled=predictor_available)
        current_y += button_height + 15 # Advance y-cursor below buttons.

        # UI Panel: Active Control Mode Display
        active_mode_display_text = "Predictor" if user_wants_predictor_mode and predictor_available else "Ground Truth"
        if not predictor_available: active_mode_display_text += " (N/A)" # Indicate if predictor is not usable.
        active_mode_colour = COLOR_TEXT_HIGHLIGHT if user_wants_predictor_mode and predictor_available else COLOR_ACCENT_YELLOW
        current_y = draw_text(screen, f"Control Mode: {active_mode_display_text}", (ui_x_start, current_y), font_ui_normal, active_mode_colour) + 10

        # UI Panel: Drone Status
        current_y = draw_text(screen, "Drone Status", (ui_x_start, current_y), font_ui_large, COLOR_TEXT) + 5
        draw_text(screen, f"Altitude: {drone_sim.z:.2f} m", (ui_x_start + 10, current_y), font_ui_normal, COLOR_TEXT_SECONDARY)
        current_y += 20 # Spacing for next line.
        current_cmd_display_ui = executed_commands_history[-1]["cmd"] if executed_commands_history else "N/A"
        draw_text(screen, f"Last Command: {current_cmd_display_ui}", (ui_x_start + 10, current_y), font_ui_normal, COLOR_TEXT_SECONDARY)
        current_y += 20
        if last_prediction_confidence is not None and user_wants_predictor_mode and predictor_available:
            # Colour-code confidence display.
            conf_colour = COLOR_ACCENT_GREEN if last_prediction_confidence > 0.7 else (COLOR_ACCENT_YELLOW if last_prediction_confidence > 0.4 else COLOR_ACCENT_RED)
            draw_text(screen, f"Confidence: {last_prediction_confidence:.2f}", (ui_x_start + 10, current_y), font_ui_normal, conf_colour)
            current_y += 20
        current_y += 5 # Extra spacing.

        # UI Panel: Performance Score (only relevant if predictor is used)
        current_y = draw_text(screen, "Performance Score", (ui_x_start, current_y), font_ui_large, COLOR_TEXT) + 5
        draw_text(screen, f"Correct Predictions: {correct_predictions}", (ui_x_start + 10, current_y), font_ui_normal, COLOR_TEXT_SECONDARY)
        current_y += 20
        draw_text(screen, f"Total Predictions: {total_predictions_made}", (ui_x_start + 10, current_y), font_ui_normal, COLOR_TEXT_SECONDARY)
        current_y += 20
        accuracy = (correct_predictions / total_predictions_made * 100) if total_predictions_made > 0 else 0.0
        # Colour-code accuracy display.
        acc_colour = COLOR_ACCENT_GREEN if accuracy >= 70 else (COLOR_ACCENT_YELLOW if accuracy >= 40 else COLOR_ACCENT_RED)
        draw_text(screen, f"Accuracy: {accuracy:.1f}%", (ui_x_start + 10, current_y), font_ui_normal, acc_colour)
        current_y += 25
        
        # UI Panel: Upcoming Commands (Flight Plan Preview)
        current_y = draw_text(screen, "Upcoming Plan", (ui_x_start, current_y), font_ui_large, COLOR_TEXT) + 5
        max_plan_display = 5 # Number of upcoming commands to show.
        for i in range(max_plan_display):
            idx_in_plan = (current_command_idx + i) % len(flight_plan_commands)
            cmd_text_display = f"{idx_in_plan + 1}. {flight_plan_commands[idx_in_plan]}"
            text_colour_plan = COLOR_TEXT_HIGHLIGHT if i == 0 else COLOR_TEXT_SECONDARY # Highlight the very next command.
            draw_text(screen, cmd_text_display, (ui_x_start + 10, current_y + i * 18), font_ui_small, text_colour_plan)
        current_y += max_plan_display * 18 + 10 # Advance y-cursor below this section.

        # UI Panel: Executed Command History
        current_y = draw_text(screen, "Executed History", (ui_x_start, current_y), font_ui_large, COLOR_TEXT) + 5
        max_history_display = 5 # Number of past commands to show.
        for i, item in enumerate(reversed(executed_commands_history[-max_history_display:])): # Show most recent first.
            cmd_text_hist = f"{len(executed_commands_history) - i}. {item['cmd']} ({item['source']})"
            item_colour_hist = COLOR_TEXT_SECONDARY
            if item["source"] == "Predictor": # Colour-code based on prediction correctness.
                if item["correct"] is True: item_colour_hist = COLOR_ACCENT_GREEN
                elif item["correct"] is False: item_colour_hist = COLOR_ACCENT_RED
            draw_text(screen, cmd_text_hist, (ui_x_start + 10, current_y + i * 18), font_ui_small, item_colour_hist)
        
        pygame.display.flip() # Update the full display Surface to the screen.

    # --- Cleanup ---
    drone_sim.close() # Disconnect from PyBullet.
    pygame.quit()     # Uninitialise Pygame modules.
    logger.info("EEG Drone Simulator exited gracefully.")

if __name__ == '__main__':
    # --- Command-Line Argument Parsing ---
    # Sets up an argument parser to accept subject ID and session type from the command line.
    parser = argparse.ArgumentParser(description="EEG Drone 3D Simulator with Scoring and MindReader Integration")
    parser.add_argument('--subject_id', type=int, default=1, 
                        help='Subject ID for EEG data loading (default: 1). Integer from 1 to 9.')
    parser.add_argument('--session_type', type=str, default='T', choices=['T', 'E', 't', 'e'],
                        help="Session type for EEG data: 'T' for training/calibration, 'E' for evaluation/test (default: 'T'). Case-insensitive.")
    
    cli_args = parser.parse_args()
    cli_args.session_type = cli_args.session_type.upper() # Standardise session type to uppercase.
    
    # Validate subject_id (assuming BCICIV 2a dataset with 9 subjects).
    if not 1 <= cli_args.subject_id <= 9:
        logger.error(f"Invalid subject_id: {cli_args.subject_id}. Must be an integer between 1 and 9.")
        sys.exit(1)
        
    main(cli_args) # Run the main application logic.
