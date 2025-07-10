import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import random
import numpy as np

class TrackGenerator:
    def __init__(self, seed, step_length=10, num_points=200):
        self.seed = seed
        self.random = random.Random(seed)
        self.step_length = step_length
        self.num_points = num_points

        # Track state
        self.current_position = (0, 0)
        self.track_points = [self.current_position]

        # Angles in degrees
        self.current_direction_angle = 0  # start pointing right (0 deg)
        self.target_direction_angle = 0  # for smooth turns
        self.previous_angles = [0] * 5  # Keep history for smoothing
        
        # Track the general forward direction to prevent backtracking
        self.initial_direction = 0  # Starting direction
        self.cumulative_turn = 0  # Track total turning to prevent loops

        self.steps_until_next_turn = self.random.randint(20, 50)
        self.turn_indices = []

        self.turning = False
        self.turn_steps_remaining = 0
        self.turn_total_steps = 0
        self.turn_angle_increment = 0
        
        # Limits to prevent sharp turns and backtracking
        self.max_turn_angle = 60  # Reduced maximum turn angle
        self.max_cumulative_turn = 120  # Maximum total turn from initial direction
        self.backtrack_threshold = 120  # Angle threshold to consider backtracking

    def normalize_angle(self, angle):
        """Normalize angle between -180 and 180 degrees."""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def get_smoothed_angle(self):
        """Return a smoothed angle based on recent history."""
        return np.mean(self.previous_angles[-3:])  # Average last 3 angles

    def would_cause_backtrack(self, proposed_angle):
        """Check if a proposed angle would cause backtracking."""
        # Calculate angle difference from initial direction
        angle_from_initial = abs(self.normalize_angle(proposed_angle - self.initial_direction))
        
        # If we're more than 120 degrees from initial direction, it's backtracking
        if angle_from_initial > self.backtrack_threshold:
            return True
            
        # Also check if cumulative turn is too much
        total_turn_from_initial = abs(self.normalize_angle(proposed_angle - self.initial_direction))
        if total_turn_from_initial > self.max_cumulative_turn:
            return True
            
        return False

    def get_safe_turn_angle(self, desired_angle):
        """Return a safe turn angle that won't cause backtracking."""
        proposed_angle = self.normalize_angle(self.current_direction_angle + desired_angle)
        
        # If proposed angle would cause backtracking, reduce it
        if self.would_cause_backtrack(proposed_angle):
            # Try reducing the angle
            safe_angle = desired_angle * 0.3  # Reduce to 30% of original
            proposed_angle = self.normalize_angle(self.current_direction_angle + safe_angle)
            
            # If still problematic, try opposite direction with small angle
            if self.would_cause_backtrack(proposed_angle):
                safe_angle = -desired_angle * 0.2  # Small turn in opposite direction
                proposed_angle = self.normalize_angle(self.current_direction_angle + safe_angle)
                
                # If still problematic, just go straight
                if self.would_cause_backtrack(proposed_angle):
                    safe_angle = 0
                    
            return safe_angle
        
        return desired_angle

    def start_big_turn(self):
        """Initialize a big smooth turn that won't cause backtracking."""
        # Choose turn angle: between 20 and 50 degrees, with random sign
        big_turn_angle = self.random.uniform(20, 50) * (1 if self.random.random() < 0.5 else -1)
        
        # Make sure this turn won't cause backtracking
        big_turn_angle = self.get_safe_turn_angle(big_turn_angle)
        
        # Set target angle relative to current angle
        self.target_direction_angle = self.normalize_angle(self.current_direction_angle + big_turn_angle)

        # Smooth turn will take 15 to 30 steps to complete
        self.turn_total_steps = self.random.randint(15, 30)
        self.turn_steps_remaining = self.turn_total_steps

        # Calculate angle increment per step for smooth transition
        angle_diff = self.normalize_angle(self.target_direction_angle - self.current_direction_angle)
        self.turn_angle_increment = angle_diff / self.turn_total_steps

        self.turning = True
        # Register turn start index for highlighting
        self.turn_indices.append(len(self.track_points))

    def step_turn(self):
        """Perform one step of a smooth big turn."""
        if self.turn_steps_remaining > 0:
            new_angle = self.current_direction_angle + self.turn_angle_increment
            
            # Double-check we're not backtracking during the turn
            if not self.would_cause_backtrack(new_angle):
                self.current_direction_angle = self.normalize_angle(new_angle)
                self.previous_angles.append(self.current_direction_angle)
                self.turn_steps_remaining -= 1
            else:
                # If we would backtrack, end the turn early
                self.turning = False
                self.turn_steps_remaining = 0
        else:
            self.turning = False
            # After turn ends, reset target to current angle
            self.target_direction_angle = self.current_direction_angle

    def generate_next_point(self):
        if self.turning:
            # Continue smooth big turn
            self.step_turn()
        else:
            # Not currently turning
            if self.steps_until_next_turn > 0:
                self.steps_until_next_turn -= 1
                # Very small random "noise" to keep it natural but mostly forward
                noise = self.random.uniform(-1, 1)  # Reduced noise
                smoothed_angle = self.get_smoothed_angle()
                proposed_angle = smoothed_angle + noise
                
                # Check if this small adjustment would cause backtracking
                if not self.would_cause_backtrack(proposed_angle):
                    self.current_direction_angle = self.normalize_angle(proposed_angle)
                    self.previous_angles.append(self.current_direction_angle)
                # If it would backtrack, just keep the current angle
                
            else:
                # Time for next turn
                self.steps_until_next_turn = self.random.randint(30, 70)

                # Decide turn type with bias to mostly small turns (85% small turns)
                turn_type_roll = self.random.random()
                if turn_type_roll < 0.85:
                    # Small soft turn (2 to 15 degrees)
                    small_turn_angle = self.random.uniform(2, 15) * (1 if self.random.random() < 0.5 else -1)
                    
                    # Get safe turn angle to prevent backtracking
                    safe_turn_angle = self.get_safe_turn_angle(small_turn_angle)
                    
                    self.current_direction_angle += safe_turn_angle
                    self.current_direction_angle = self.normalize_angle(self.current_direction_angle)
                    self.previous_angles.append(self.current_direction_angle)

                    # Mark this as a turn point only if we actually turned
                    if abs(safe_turn_angle) > 0.5:  # Only mark significant turns
                        self.turn_indices.append(len(self.track_points))
                else:
                    # Start a smooth big turn (circle-like)
                    self.start_big_turn()

        # Move forward by step length
        rad = math.radians(self.current_direction_angle)
        x, y = self.current_position
        dx = math.cos(rad) * self.step_length
        dy = math.sin(rad) * self.step_length
        new_pos = (x + dx, y + dy)
        self.current_position = new_pos
        self.track_points.append(new_pos)

    def generate_full_track(self):
        for _ in range(self.num_points):
            self.generate_next_point()
        return self.track_points, self.turn_indices


class TrackApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Non-Backtracking Track Generator with Dual View")

        # UI Controls
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Label(controls_frame, text="Seed:").pack(side=tk.LEFT, padx=5)
        self.seed_var = tk.StringVar(value="12345")
        self.seed_entry = ttk.Entry(controls_frame, textvariable=self.seed_var, width=15)
        self.seed_entry.pack(side=tk.LEFT)

        # Random seed button
        ttk.Button(controls_frame, text="Random Seed", command=self.generate_random_seed).pack(side=tk.LEFT, padx=5)

        ttk.Label(controls_frame, text="Track Length:").pack(side=tk.LEFT, padx=5)
        self.length_var = tk.IntVar(value=1000)
        self.length_entry = ttk.Entry(controls_frame, textvariable=self.length_var, width=7)
        self.length_entry.pack(side=tk.LEFT)

        self.generate_btn = ttk.Button(controls_frame, text="Generate Full Track", command=self.generate_full_track)
        self.generate_btn.pack(side=tk.LEFT, padx=5)

        self.live_btn = ttk.Button(controls_frame, text="Start Live Generation", command=self.start_live_generation)
        self.live_btn.pack(side=tk.LEFT, padx=5)

        self.stop_live = False

        # Create two Matplotlib figures side by side
        fig_frame = ttk.Frame(self)
        fig_frame.pack(fill=tk.BOTH, expand=True)

        # Zoomed out full track
        self.fig_full, self.ax_full = plt.subplots(figsize=(5, 5))
        self.ax_full.set_title("Zoomed-Out Full Track")
        self.ax_full.set_aspect('equal')
        self.ax_full.grid(True)
        self.canvas_full = FigureCanvasTkAgg(self.fig_full, master=fig_frame)
        self.canvas_full.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Zoomed in live track
        self.fig_live, self.ax_live = plt.subplots(figsize=(5, 5))
        self.ax_live.set_title("Zoomed-In Live Generation")
        self.ax_live.set_aspect('equal')
        self.ax_live.grid(True)
        self.canvas_live = FigureCanvasTkAgg(self.fig_live, master=fig_frame)
        self.canvas_live.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Initialize variables for live generation
        self.track_gen = None
        self.live_points_x = []
        self.live_points_y = []
        self.live_turns_x = []
        self.live_turns_y = []

    def generate_random_seed(self):
        """Generate a random seed and update the seed entry."""
        random_seed = random.randint(0, 999999999)
        self.seed_var.set(str(random_seed))

    def generate_full_track(self):
        try:
            seed = self.seed_var.get()
            # Try to convert to int if it's a numeric string
            try:
                seed = int(seed)
            except ValueError:
                pass  # Keep as string if not numeric
            
            length = self.length_var.get()
            self.track_gen = TrackGenerator(seed, step_length=10, num_points=length)
            points, turn_indices = self.track_gen.generate_full_track()

            xs, ys = zip(*points)
            self.ax_full.clear()

            # Plot track
            self.ax_full.plot(xs, ys, marker='o', linestyle='-', color='blue', markersize=3, label='Track Path')

            # Highlight turns
            turn_x = [points[i][0] for i in turn_indices]
            turn_y = [points[i][1] for i in turn_indices]
            self.ax_full.scatter(turn_x, turn_y, color='red', s=40, label='Turns')

            # Set zoomed out limits with margin
            margin = 100
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            self.ax_full.set_xlim(min_x - margin, max_x + margin)
            self.ax_full.set_ylim(min_y - margin, max_y + margin)

            self.ax_full.set_title(f"Non-Backtracking Track (Seed={seed}, Length={length})")
            self.ax_full.set_aspect('equal')
            self.ax_full.grid(True)
            self.ax_full.legend()
            self.canvas_full.draw()
        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def start_live_generation(self):
        # Reset live view
        self.stop_live = False
        try:
            seed = self.seed_var.get()
            # Try to convert to int if it's a numeric string
            try:
                seed = int(seed)
            except ValueError:
                pass  # Keep as string if not numeric
                
            length = self.length_var.get()
            self.track_gen = TrackGenerator(seed, step_length=10, num_points=length)

            self.live_points_x = [0]
            self.live_points_y = [0]
            self.live_turns_x = []
            self.live_turns_y = []

            self.ax_live.clear()
            self.ax_live.set_title("Zoomed-In Live Generation")
            self.ax_live.set_aspect('equal')
            self.ax_live.grid(True)

            self.live_step(0)
        except Exception as e:
            tk.messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def live_step(self, step_count):
        if self.stop_live or step_count >= self.track_gen.num_points:
            return

        # Generate next point
        self.track_gen.generate_next_point()
        x, y = self.track_gen.current_position
        self.live_points_x.append(x)
        self.live_points_y.append(y)

        # Check if last point was a turn (using stored indices)
        if len(self.track_gen.turn_indices) > 0 and self.track_gen.turn_indices[-1] == len(self.live_points_x) - 1:
            self.live_turns_x.append(x)
            self.live_turns_y.append(y)

        self.ax_live.clear()

        # Plot path so far
        self.ax_live.plot(self.live_points_x, self.live_points_y, marker='o', linestyle='-', color='blue', markersize=5, label='Track Path')

        # Plot turns in red
        if self.live_turns_x:
            self.ax_live.scatter(self.live_turns_x, self.live_turns_y, color='red', s=60, label='Turns')

        # Zoom tightly around current position (with margin)
        margin = 70
        cx, cy = x, y
        self.ax_live.set_xlim(cx - margin, cx + margin)
        self.ax_live.set_ylim(cy - margin, cy + margin)

        self.ax_live.set_title(f"Zoomed-In Live Generation (Step {step_count + 1})")
        self.ax_live.set_aspect('equal')
        self.ax_live.grid(True)
        self.ax_live.legend()

        self.canvas_live.draw()

        # Schedule next step
        self.after(40, lambda: self.live_step(step_count + 1))


if __name__ == "__main__":
    app = TrackApp()
    app.mainloop()