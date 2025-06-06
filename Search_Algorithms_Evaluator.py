import time
import numpy as np
import matplotlib.pyplot as plt
from Search_Environments_2 import generate_maze, generate_start_goal, MAP_SIZE, NUM_ROBOTS
from Multi_Robot_Search_Algorithms import a_star, d_star, m_star

def evaluate_algorithm(algorithm, maze, starts, goals):
    start_time = time.time()
    paths = algorithm(np.copy(maze), starts, goals)
    execution_time = time.time() - start_time
    return execution_time, paths

def plot_results(maze, starts, goals, astar_paths, dstar_paths, mstar_paths):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    titles = ["A* Search", "D* Search", "M* Search"]
    path_sets = [astar_paths, dstar_paths, mstar_paths]
    distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for ax, title, path_set in zip(axes, titles, path_sets):
        ax.set_xlim(-0.5, maze.shape[1] - 0.5)
        ax.set_ylim(maze.shape[0] - 0.5, -0.5)
        
        # Draw gridlines
        ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
        ax.grid(which="minor", color='gray', linestyle='-', linewidth=0.5)
        
        # Draw obstacles as continuous blocks
        for y in range(maze.shape[0]):
            for x in range(maze.shape[1]):
                if maze[y, x] == 1:
                    ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black'))
        
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot paths
        for i, (start, goal, path) in enumerate(zip(starts, goals, path_set)):
            color = distinct_colors[i % len(distinct_colors)]
            ax.plot(start[1], start[0], marker='o', markersize=8, color=color, markeredgecolor='black', label=f'Robot {i+1}')
            ax.plot(goal[1], goal[0], marker='s', markersize=8, color=color, markeredgecolor='black')
            
            if path:
                path_x = [p[1] for p in path]
                path_y = [p[0] for p in path]
                ax.plot(path_x, path_y, color=color, linewidth=2, linestyle='-')  # Solid line
        
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.legend()
    plt.show()

def evaluate_algorithms(obstacle_type):
    maze = generate_maze(obstacle_type)
    starts, goals = generate_start_goal(maze)
    
    print("Evaluating A* Algorithm...")
    astar_time, astar_paths = evaluate_algorithm(a_star, maze, starts, goals)
    print(f"A* Execution Time: {astar_time:.4f} seconds")
    
    print("Evaluating D* Algorithm...")
    dstar_time, dstar_paths = evaluate_algorithm(d_star, maze, starts, goals)
    print(f"D* Execution Time: {dstar_time:.4f} seconds")
    
    print("Evaluating M* Algorithm...")
    mstar_time, mstar_paths = evaluate_algorithm(m_star, maze, starts, goals)
    print(f"M* Execution Time: {mstar_time:.4f} seconds")
    
    plot_results(maze, starts, goals, astar_paths, dstar_paths, mstar_paths)

if __name__ == "__main__":
    obstacle_type = int(input("Enter obstacle type (1-5): "))
    evaluate_algorithms(obstacle_type)
