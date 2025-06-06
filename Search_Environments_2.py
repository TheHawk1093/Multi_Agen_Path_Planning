import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# === Parameters ===
MAP_SIZE = 30
NUM_OBSTACLES = 20
NUM_ROBOTS = 5
OBSTACLE_LENGTH_MIN = 6
OBSTACLE_LENGTH_MAX = 13

def generate_maze(obstacle_type: int) -> np.ndarray:
    """
    Generates a maze according to the given obstacle_type.
    Option 6: Creates a narrow passage with up to 3-step backtracking alcoves for NUM_ROBOTS robots.
    """
    maze = np.ones((MAP_SIZE, MAP_SIZE), dtype=int)  # Initialize with walls
    maze[1:-1, 1:-1] = 0  # Create open area inside

    if obstacle_type == 1:
        for i in range(NUM_OBSTACLES):
            x = 1 if i % 2 == 0 else MAP_SIZE - OBSTACLE_LENGTH_MAX - 1
            y = (i + 1) * (MAP_SIZE // (NUM_OBSTACLES + 1))
            maze[y:y+1, x:x+OBSTACLE_LENGTH_MAX] = 1

    elif obstacle_type == 2:
        for i in range(NUM_OBSTACLES):
            x = (i + 1) * (MAP_SIZE // (NUM_OBSTACLES + 1))
            y = 1 if i % 2 == 0 else MAP_SIZE - OBSTACLE_LENGTH_MAX - 1
            maze[y:y+OBSTACLE_LENGTH_MAX, x:x+1] = 1

    elif obstacle_type == 3:
        for i in range(NUM_OBSTACLES):
            if i % 2 == 0:  # Horizontal obstacles
                width = random.randint(OBSTACLE_LENGTH_MIN, OBSTACLE_LENGTH_MAX)
                x = random.randint(1, MAP_SIZE - width - 1)
                y = (i + 1) * (MAP_SIZE // (NUM_OBSTACLES + 1))
                maze[y:y+1, x:x+width] = 1
            else:  # Vertical obstacles
                height = random.randint(OBSTACLE_LENGTH_MIN, OBSTACLE_LENGTH_MAX)
                x = (i + 1) * (MAP_SIZE // (NUM_OBSTACLES + 1))
                y = random.randint(1, MAP_SIZE - height - 1)
                maze[y:y+height, x:x+1] = 1

    elif obstacle_type in [4, 5]:
        for _ in range(NUM_OBSTACLES):
            while True:
                width = 1 if obstacle_type == 5 else random.randint(OBSTACLE_LENGTH_MIN, OBSTACLE_LENGTH_MAX)
                height = 1 if obstacle_type == 4 else random.randint(OBSTACLE_LENGTH_MIN, OBSTACLE_LENGTH_MAX)
                x = random.randint(1, MAP_SIZE - width - 1)
                y = random.randint(1, MAP_SIZE - height - 1)
                if np.all(maze[y:y+height, x:x+width] == 0):  # Ensure no overlap
                    maze[y:y+height, x:x+width] = 1
                    break

    elif obstacle_type == 6:
        # --- Narrow passage parameters ---
        passage_row = MAP_SIZE // 2
        passage_start_col = MAP_SIZE // 6
        passage_end_col = MAP_SIZE - MAP_SIZE // 6 - 1
        passage_width = passage_end_col - passage_start_col + 1  # actual corridor length

        # Clear the narrow passage
        maze[passage_row, passage_start_col:passage_end_col+1] = 0

        # Add walls above and below to make it truly one-cell wide
        maze[passage_row-1, passage_start_col:passage_end_col+1] = 1
        maze[passage_row+1, passage_start_col:passage_end_col+1] = 1

        # Make entry and exit on sides
        maze[passage_row, passage_start_col-1] = 0   # left entrance
        maze[passage_row, passage_end_col+1] = 0     # right exit

        # Add up to 3 "alcoves" (side branches) for backtracking/waiting
        # Alcoves alternate above and below passage
        alcove_cols = np.linspace(passage_start_col + 2, passage_end_col - 2, 3, dtype=int)
        for idx, col in enumerate(alcove_cols):
            # Above
            if 1 <= passage_row-2 < MAP_SIZE-1:
                maze[passage_row-2, col] = 0
                maze[passage_row-1, col] = 0
            # Below
            if 1 <= passage_row+2 < MAP_SIZE-1:
                maze[passage_row+2, col] = 0
                maze[passage_row+1, col] = 0

    return maze

def generate_start_goal(maze: np.ndarray, obstacle_type: int = 1) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Generates start and goal positions for robots, depending on obstacle_type.
    Option 6: Places NUM_ROBOTS starts at one end of the passage, goals at the other end.
    """
    starts, goals = [], []
    if obstacle_type != 6:
        while len(starts) < NUM_ROBOTS:
            start = (random.randint(1, MAP_SIZE//3), random.randint(1, MAP_SIZE-2))
            goal = (random.randint(2*MAP_SIZE//3, MAP_SIZE-2), random.randint(1, MAP_SIZE-2))
            if (maze[start] == 0 and maze[goal] == 0 and
                start != goal and start not in starts and goal not in goals):
                starts.append(start)
                goals.append(goal)
        return starts, goals

    # --- Special start/goal for narrow passage (type 6) ---
    passage_row = MAP_SIZE // 2
    passage_start_col = MAP_SIZE // 6
    passage_end_col = MAP_SIZE - MAP_SIZE // 6 - 1

    # Place robots at/near left entrance (staggered in vertical, so they don't overlap)
    # And goals at the right exit (staggered)
    for i in range(NUM_ROBOTS):
        # Spread vertically
        delta = i - NUM_ROBOTS // 2
        start_row = min(max(passage_row + delta, 1), MAP_SIZE-2)
        goal_row = min(max(passage_row - delta, 1), MAP_SIZE-2)
        starts.append((start_row, passage_start_col-1))
        goals.append((goal_row, passage_end_col+1))

    return starts, goals

def plot_maze(maze: np.ndarray, starts: list[tuple[int, int]], goals: list[tuple[int, int]], 
               paths=None, title="Maze"):
    """
    Grid-based maze visualization with precise cell representation
    """
    plt.figure(figsize=(12, 12), dpi=150)
    
    # Distinct colors for robots (loop if more robots than colors)
    distinct_colors = [
        '#1f77b4',  # Muted blue
        '#ff7f0e',  # Bright orange
        '#2ca02c',  # Vivid green
        '#d62728',  # Brick red
        '#9467bd',  # Muted purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
    ]
    
    plt.xlim(-0.5, maze.shape[1]-0.5)
    plt.ylim(maze.shape[0]-0.5, -0.5)
    
    # Draw grid
    for x in range(maze.shape[1]):
        plt.axvline(x-0.5, color='lightgray', linewidth=0.5)
    for y in range(maze.shape[0]):
        plt.axhline(y-0.5, color='lightgray', linewidth=0.5)
    
    # Visualize maze (obstacles)
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == 1:
                plt.fill([x-0.5, x+0.5, x+0.5, x-0.5], 
                         [y-0.5, y-0.5, y+0.5, y+0.5], 
                         color='black')
    
    # Plot paths and markers for each robot
    for i, (start, goal) in enumerate(zip(starts, goals)):
        color = distinct_colors[i % len(distinct_colors)]
        plt.plot(start[1], start[0], marker='^', 
                 markersize=7, color=color, 
                 markerfacecolor=color,
                 markeredgecolor='black', markeredgewidth=0.7, label=f"Start {i+1}" if i==0 else "")
        plt.plot(goal[1], goal[0], marker='s', 
                 markersize=7, color=color, 
                 markerfacecolor=color,
                 markeredgecolor='black', markeredgewidth=0.7, label=f"Goal {i+1}" if i==0 else "")
        # Plot paths if provided
        if paths and i < len(paths) and paths[i] and len(paths[i]) > 2:
            path_x = [p[1] for p in paths[i]]
            path_y = [p[0] for p in paths[i]]
            plt.plot(path_x, path_y, color=color, 
                     linewidth=1, linestyle='--', alpha=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(False)
    plt.tight_layout()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    print("Choose obstacle type:")
    print("1: Horizontal bars")
    print("2: Vertical bars")
    print("3: Mixed bars")
    print("4: Random rectangles")
    print("5: Random singles")
    print("6: Narrow multi-robot passage (max 3-step backtrack)")
    obstacle_type = int(input("Enter obstacle type (1-6): "))
    maze = generate_maze(obstacle_type)
    starts, goals = generate_start_goal(maze, obstacle_type)
    print("\nRobot positions (0-based coordinates):")
    for i, (s, g) in enumerate(zip(starts, goals)):
        print(f"Robot {i+1}: Start at ({s[0]},{s[1]}), Goal at ({g[0]},{g[1]})")
    plot_maze(maze, starts, goals, title=f"Env type {obstacle_type}")
