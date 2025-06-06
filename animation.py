import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.patches import Rectangle, Circle

# Import your custom modules
from Search_Environments_2 import generate_maze, generate_start_goal, MAP_SIZE, NUM_ROBOTS
from Multi_Robot_Search_Algorithms import a_star, d_star, m_star

def set_backend_for_display():
    """Set an interactive backend for display purposes"""
    try:
        matplotlib.use('TkAgg')  # Try TkAgg first
    except:
        try:
            matplotlib.use('Qt5Agg')  # Fallback to Qt5
        except:
            matplotlib.use('Agg')  # Final fallback

def set_backend_for_saving():
    """Set the Agg backend for saving files"""
    matplotlib.use('Agg')

def plot_path_results(maze, starts, goals, paths, algorithm_name="A*"):
    """
    Display static plot of the paths found and handle saving options.
    Returns the figure object for potential reuse.
    """
    set_backend_for_display()
    plt.switch_backend('TkAgg') if 'TkAgg' in matplotlib.get_backend() else plt.switch_backend('Qt5Agg')
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_xlim(-0.5, maze.shape[1] - 0.5)
    ax.set_ylim(maze.shape[0] - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
    ax.grid(which="minor", color='gray', linestyle='-', linewidth=0.5)

    # Draw obstacles
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == 1:
                ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, color='black'))

    # Color palette for robots
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (start, goal, path) in enumerate(zip(starts, goals, paths)):
        color = distinct_colors[i % len(distinct_colors)]
        # Start marker
        ax.add_patch(Circle((start[1], start[0]), 0.4, color=color, ec='black'))
        # Goal marker
        ax.add_patch(Rectangle((goal[1]-0.4, goal[0]-0.4), 0.8, 0.8, color=color, ec='black'))
        # Path lines
        if path:
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, color=color, linewidth=2, linestyle='-', alpha=0.5)
            # Final position marker
            ax.add_patch(Circle((path[-1][1], path[-1][0]), 0.3, color=color, ec='black', zorder=10))

    ax.set_title(f"Path Results - {algorithm_name}", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
    # Add statistics to the plot
    path_lengths = [len(path)-1 if path else 0 for path in paths]
    valid_paths = sum(1 for length in path_lengths if length > 0)
    stats_text = (f"Success Rate: {valid_paths}/{len(paths)}\n"
                 f"Path Lengths: {', '.join(map(str, path_lengths))}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.show(block=False)  # Non-blocking show
    
    return fig

def animate_robot_movement(maze, starts, goals, paths, algorithm_name="A*"):
    """Create animation with collision detection"""
    set_backend_for_saving()
    plt.switch_backend('Agg')
    
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_xlim(-0.5, maze.shape[1] - 0.5)
    ax.set_ylim(maze.shape[0] - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
    ax.grid(which="minor", color='gray', linestyle='-', linewidth=0.5)

    # Draw obstacles
    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y, x] == 1:
                ax.add_patch(Rectangle((x-0.5, y-0.5), 1, 1, color='black'))

    # Color palette
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    robot_markers = []
    robot_colors = []
    for i, (start, goal) in enumerate(zip(starts, goals)):
        color = distinct_colors[i % len(distinct_colors)]
        robot_colors.append(color)
        # Start marker
        ax.add_patch(Circle((start[1], start[0]), 0.4, color=color, ec='black'))
        # Goal marker
        ax.add_patch(Rectangle((goal[1]-0.4, goal[0]-0.4), 0.8, 0.8, color=color, ec='black'))
        # Moving robot marker
        robot_marker = Circle((start[1], start[0]), 0.3, color=color, ec='black', zorder=10)
        ax.add_patch(robot_marker)
        robot_markers.append(robot_marker)

    # Path lines
    for i, path in enumerate(paths):
        if path:
            color = robot_colors[i]
            path_x = [p[1] for p in path]
            path_y = [p[0] for p in path]
            ax.plot(path_x, path_y, color=color, linewidth=2, linestyle='-', alpha=0.5)

    ax.set_title(f"Robot Movement - {algorithm_name}", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    max_path_length = max(len(path) for path in paths) if paths else 0

    def update(frame):
        position_to_indices = {}
        for i, path in enumerate(paths):
            if frame < len(path):
                pos = (path[frame][1], path[frame][0])
                robot_markers[i].center = pos
                position_to_indices.setdefault(pos, []).append(i)

        # Handle collisions
        for pos, indices in position_to_indices.items():
            if len(indices) > 1:  # Collision
                for idx in indices:
                    robot_markers[idx].set_color('red')
            else:
                idx = indices[0]
                robot_markers[idx].set_color(robot_colors[idx])

        return robot_markers

    anim = FuncAnimation(
        fig, update, frames=max_path_length,
        interval=200, blit=True, repeat=False
    )
    return anim

def handle_saving(fig, animation, algorithm_name):
    """Handle all saving options for both static and animated results"""
    while True:
        print("\nSave options:")
        print("1. Save static image (PNG)")
        print("2. Save animation (GIF/MP4)")
        print("3. Save both")
        print("4. Don't save anything")
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '4':
            break
            
        if choice in ('1', '3'):
            # Save static image
            filename = input("Enter filename for PNG (without extension) or press Enter for default: ").strip()
            if not filename:
                filename = f"{algorithm_name.lower().replace('*', 'star')}_static"
            try:
                fig.savefig(f"{filename}.png", dpi=150, bbox_inches='tight')
                print(f"Static image saved as {filename}.png")
            except Exception as e:
                print(f"Error saving static image: {e}")
        
        if choice in ('2', '3'):
            # Save animation
            save_animation(animation, algorithm_name)
        
        if choice in ('1', '2', '3'):
            break

def save_animation(animation, algorithm_name):
    """Save animation with format options"""
    filename_base = input(f"Enter base filename for animation (without extension) or press Enter for default: ").strip()
    if not filename_base:
        filename_base = f"{algorithm_name.lower().replace('*', 'star')}_animation"
    
    print("\nSelect animation format:")
    print("1. GIF")
    print("2. MP4")
    print("3. Both")
    format_choice = input("Enter your choice (1-3): ").strip()
    
    if format_choice in ('1', '3'):
        try:
            animation.save(f"{filename_base}.gif", writer=PillowWriter(fps=5))
            print(f"Animation saved as {filename_base}.gif")
        except Exception as e:
            print(f"Failed to save GIF: {e}")
    
    if format_choice in ('2', '3'):
        try:
            writer = FFMpegWriter(fps=5, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
            animation.save(f"{filename_base}.mp4", writer=writer)
            print(f"Animation saved as {filename_base}.mp4")
        except Exception as e:
            print(f"Failed to save MP4: {e}")

def test_algorithm(algorithm, algorithm_name):
    """Test an algorithm with visualization and saving options"""
    obstacle_type = int(input("Enter obstacle type (1-5): "))
    maze = generate_maze(obstacle_type)
    starts, goals = generate_start_goal(maze)
    
    print(f"\nRunning {algorithm_name}...")
    start_time = time.time()
    paths = algorithm(np.copy(maze), starts, goals)
    comp_time = time.time() - start_time
    
    # Calculate metrics
    path_lengths = [len(path)-1 if path else 0 for path in paths]
    valid_paths = sum(1 for length in path_lengths if length > 0)
    
    print(f"\n{algorithm_name} Results:")
    print(f"Computation Time: {comp_time:.2f}s")
    print(f"Success Rate: {valid_paths}/{len(paths)}")
    for i, length in enumerate(path_lengths):
        print(f"Robot {i+1} Path Length: {length} steps")
    
    # Show static results
    fig = plot_path_results(maze, starts, goals, paths, algorithm_name)
    
    # Prepare animation (but don't create until needed)
    animation = None
    if input("\nWould you like to generate an animation? (y/n): ").lower() == 'y':
        animation = animate_robot_movement(maze, starts, goals, paths, algorithm_name)
    
    # Handle saving options
    if fig and (animation or input("Would you like to save the static image? (y/n): ").lower() == 'y'):
        if animation is None and input("Animation not created. Create it now? (y/n): ").lower() == 'y':
            animation = animate_robot_movement(maze, starts, goals, paths, algorithm_name)
        handle_saving(fig, animation, algorithm_name)
    
    plt.close('all')

def run_comparison():
    """Compare all algorithms with the same environment"""
    obstacle_type = int(input("Enter obstacle type (1-5): "))
    maze = generate_maze(obstacle_type)
    starts, goals = generate_start_goal(maze)
    
    algorithms = [
        (a_star, "A*"),
        (d_star, "D*"),
        (m_star, "M*")
    ]
    
    results = []
    for algo, name in algorithms:
        print(f"\n{'='*40}")
        print(f"Running {name}")
        print(f"{'='*40}")
        
        start_time = time.time()
        paths = algo(np.copy(maze), starts, goals)
        comp_time = time.time() - start_time
        
        path_lengths = [len(path)-1 if path else 0 for path in paths]
        valid_paths = sum(1 for length in path_lengths if length > 0)
        
        results.append({
            "name": name,
            "comp_time": comp_time,
            "path_lengths": path_lengths,
            "valid_paths": valid_paths,
            "total_robots": len(paths)
        })
        
        print(f"\n{name} Results:")
        print(f"Computation Time: {comp_time:.2f}s")
        print(f"Success Rate: {valid_paths}/{len(paths)}")
        for i, length in enumerate(path_lengths):
            print(f"Robot {i+1} Path Length: {length} steps")
        
        # Show and optionally save results
        fig = plot_path_results(maze, starts, goals, paths, name)
        if input(f"\nWould you like to save {name} results? (y/n): ").lower() == 'y':
            animation = animate_robot_movement(maze, starts, goals, paths, name)
            handle_saving(fig, animation, name)
        
        plt.close('all')
    
    # Print summary table
    print("\nSummary of Results:")
    header = f"{'Algorithm':<10} {'Time (s)':<10} {'Success':<10} " + " ".join([f"R{i+1} Len" for i in range(len(starts))])
    print(header)
    print("-" * (30 + 8 * len(starts)))
    for res in results:
        path_str = " ".join([f"{l:<8}" for l in res["path_lengths"]])
        print(f"{res['name']:<10} {res['comp_time']:<10.2f} {res['valid_paths']}/{res['total_robots']:<10} {path_str}")

def main_menu():
    """Display main menu and handle user choice"""
    print("\nMulti-Robot Path Visualizer")
    print("1. Test single algorithm")
    print("2. Compare all algorithms")
    print("3. Exit")
    
    while True:
        choice = input("Select option (1-3): ").strip()
        
        if choice == '1':
            print("\nAvailable algorithms:")
            print("1. A*")
            print("2. D*")
            print("3. M*")
            algo_choice = input("Select algorithm (1-3): ").strip()
            
            algorithms = {
                '1': (a_star, "A*"),
                '2': (d_star, "D*"),
                '3': (m_star, "M*")
            }
            
            if algo_choice in algorithms:
                test_algorithm(*algorithms[algo_choice])
            else:
                print("Invalid choice")
                
        elif choice == '2':
            run_comparison()
            
        elif choice == '3':
            print("Exiting program.")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    set_backend_for_display()
    main_menu()