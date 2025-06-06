import numpy as np
import math
from queue import PriorityQueue, Queue
from Search_Environments_2 import generate_maze, generate_start_goal, plot_maze, MAP_SIZE, NUM_ROBOTS

# 4-directional neighbors (Up, Down, Left, Right)
def get_neighbors_4d(maze: np.ndarray, node: tuple) -> list:
    """
    Get valid neighboring positions in 4 directions.
    
    Args:
    - maze (np.ndarray): The maze representation (2D grid).
    - node (tuple): (x, y) position of the current node.
    
    Returns:
    - list of tuples: Valid neighbor positions.
    """
    x, y = node
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
            neighbors.append((nx, ny))
    
    return neighbors

# 8-directional neighbors (includes diagonal moves)
def get_neighbors_8d(maze: np.ndarray, node: tuple) -> list:
    """
    Get valid neighboring positions in 8 directions.
    
    Args:
    - maze (np.ndarray): The maze representation (2D grid).
    - node (tuple): (x, y) position of the current node.
    
    Returns:
    - list of tuples: Valid neighbor positions.
    """
    x, y = node
    neighbors = []
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),  
        (-1, -1), (-1, 1), (1, -1), (1, 1)
    ]
    
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
            neighbors.append((nx, ny))
    
    return neighbors

# Function to compute movement cost
def get_cost(node: tuple, neighbor: tuple) -> float:
    """
    Compute movement cost between two nodes.
    
    Args:
    - node (tuple): Current position (x, y).
    - neighbor (tuple): Neighbor position (x, y).
    
    Returns:
    - float: Cost of moving from node to neighbor.
    """
    x1, y1 = node
    x2, y2 = neighbor
    
    if abs(x1 - x2) + abs(y1 - y2) == 1:
        return 1  # Straight move
    elif abs(x1 - x2) == 1 and abs(y1 - y2) == 1:
        return math.sqrt(2)  # Diagonal move
    else:
        return float('inf')  # Invalid move

# Heuristic function for A* (Euclidean distance)
def heuristic(node: tuple, neighbor: tuple) -> float:
    """
    Calculate Euclidean distance heuristic between two points.
    
    Args:
    - node (tuple): First point coordinates (x1, y1)
    - neighbor (tuple): Second point coordinates (x2, y2)
    
    Returns:
    - float: Euclidean distance between points
    """
    return (abs(node[0] - neighbor[0])+ abs(node[1] - neighbor[1]))

def a_star(maze: np.ndarray, starts: list, goals: list):
    """
    Final Multi-robot A* with:
    - Proper goal cell reservation
    - Complete collision avoidance
    - Extended backtracking
    - Start point included in path
    - Empty path returned if no solution found
    """
    num_robots = len(starts)
    paths = [[] for _ in range(num_robots)]
    reservations = {}  # (x, y, t): robot_id
    occupied_goals = set()  # Track permanently occupied goals

    # For each robot, record when it reaches goal
    robot_goal_time = [None] * num_robots

    # Initialize robot data
    robot_data = []
    for i in range(num_robots):
        start = starts[i]
        goal = goals[i]
        if start == goal:
            paths[i] = [start]  # Start is included
            robot_goal_time[i] = 0
            occupied_goals.add(goal)
            continue
            
        fringe = PriorityQueue()
        h_start = heuristic(start, goal)
        fringe.put((h_start, 0, start, 0))  # (f, g, node, t)
        
        robot_data.append({
            'fringe': fringe,
            'prev_node': {},
            'cum_costs': {(start, 0): 0},
            'total_costs': {(start, 0): h_start},
            'goal': goal,
            'path': [start],  # Initialize with start position
            'active': True,
            'current_pos': start,
            'current_time': 0,
            'history': [start]  # Track full movement history
        })

    global_time = 0
    active_robots = len([d for d in robot_data if d.get('active', False)]) + \
                   sum(1 for i in range(num_robots) if starts[i] == goals[i])
    max_time = maze.size * 3  # Reasonable finite time limit

    while active_robots > 0 and global_time < max_time:
        global_time += 1
        planned_moves = {}

        # Phase 1: Plan moves and expand A* for each robot
        for i in range(len(robot_data)):
            if not robot_data[i]['active']:
                continue
                
            while not robot_data[i]['fringe'].empty():
                f, g, current, t = robot_data[i]['fringe'].queue[0]
                if t > global_time:
                    break
                robot_data[i]['fringe'].get()

                # --- Goal check ---
                if current == robot_data[i]['goal']:
                    # Reconstruct path (including start)
                    path = []
                    node = (current, t)
                    while node in robot_data[i]['prev_node']:
                        path.append((node[0][0], node[0][1], node[1]))
                        node = robot_data[i]['prev_node'][node]
                    path.reverse()
                    robot_data[i]['path'] = [starts[i]] + [(x, y) for x, y, t in path]  # Ensure start is included
                    robot_data[i]['active'] = False
                    active_robots -= 1
                    robot_goal_time[i] = t
                    occupied_goals.add(current)
                    # Reserve goal for all future times
                    for future_t in range(t, max_time):
                        reservations[(current, future_t)] = i
                    break

                # Generate moves/neighbors with collision checking
                for neighbor in get_neighbors_4d(maze, current):
                    new_time = t + 1
                    
                    # Collision detection
                    collision = False
                    if neighbor in occupied_goals:
                        collision = True
                    if (neighbor, new_time) in reservations:
                        collision = True
                    if (neighbor, t) in reservations:
                        other_robot = reservations[(neighbor, t)]
                        if (current, new_time) in reservations:
                            if reservations[(current, new_time)] == other_robot:
                                collision = True
                    if len(robot_data[i]['history']) >= 2:
                        prev_pos = robot_data[i]['history'][-2]
                        if neighbor == prev_pos and (prev_pos, new_time) in reservations:
                            collision = True
                            
                    if collision:
                        continue
                        
                    # Calculate costs
                    edge_cost = 1
                    tentative_g = g + edge_cost
                    new_f = tentative_g + heuristic(neighbor, robot_data[i]['goal'])
                    
                    # Update if better path found
                    if (neighbor, new_time) not in robot_data[i]['cum_costs'] or \
                            tentative_g < robot_data[i]['cum_costs'].get((neighbor, new_time), float('inf')):
                        robot_data[i]['prev_node'][(neighbor, new_time)] = (current, t)
                        robot_data[i]['cum_costs'][(neighbor, new_time)] = tentative_g
                        robot_data[i]['total_costs'][(neighbor, new_time)] = new_f
                        robot_data[i]['fringe'].put((new_f, tentative_g, neighbor, new_time))
                        robot_data[i]['history'] = robot_data[i]['history'][-10:] + [neighbor]

                # Wait action or extended backtracking
                if (current, t + 1) not in reservations:
                    robot_data[i]['fringe'].put((f + 1, g + 1, current, t + 1))
                else:
                    # Extended backtracking
                    for prev_pos in reversed(robot_data[i]['history'][:-1]):
                        if (prev_pos, t + 1) not in reservations:
                            robot_data[i]['fringe'].put((f + 2, g + 1, prev_pos, t + 1))
                            break

        # Phase 2: Reserve planned positions
        for i in range(len(robot_data)):
            if not robot_data[i]['active']:
                continue
                
            if not robot_data[i]['fringe'].empty():
                f, g, pos, t = robot_data[i]['fringe'].queue[0]
                if t == global_time + 1:
                    if (pos, t) not in planned_moves:
                        planned_moves[(pos, t)] = []
                    planned_moves[(pos, t)].append(i)

        # Phase 3: Conflict resolution
        for (pos, t), robots in planned_moves.items():
            if len(robots) > 1:
                # Sort by priority
                robots.sort(key=lambda x: (
                    len(robot_data[x]['path']) if robot_data[x]['path'] else float('inf'),
                    heuristic(robot_data[x]['current_pos'], robot_data[x]['goal'])
                ))
                
                winner = robots[0]
                for robot in robots[1:]:
                    # Replan for losing robots
                    new_queue = []
                    while not robot_data[robot]['fringe'].empty():
                        item = robot_data[robot]['fringe'].get()
                        new_queue.append(item)
                        
                    # Try backtracking
                    backtracked = False
                    for prev_pos in reversed(robot_data[robot]['history'][:-1]):
                        new_time = global_time + 1
                        if (prev_pos, new_time) not in reservations:
                            g = robot_data[robot]['cum_costs'].get((robot_data[robot]['current_pos'], global_time), 0) + 1
                            f = g + heuristic(prev_pos, robot_data[robot]['goal'])
                            new_queue.append((f, g, prev_pos, new_time))
                            backtracked = True
                            break
                            
                    if not backtracked:
                        # If can't backtrack, wait
                        current_pos = robot_data[robot]['current_pos']
                        g = robot_data[robot]['cum_costs'].get((current_pos, global_time), 0) + 1
                        f = g + heuristic(current_pos, robot_data[robot]['goal'])
                        new_queue.append((f, g, current_pos, global_time + 1))
                        
                    # Reinsert items
                    for item in sorted(new_queue, key=lambda x: x[0]):
                        robot_data[robot]['fringe'].put(item)
                        
                reservations[(pos, t)] = winner
            else:
                reservations[(pos, t)] = robots[0]

    # Final path processing
    for i in range(num_robots):
        if starts[i] == goals[i]:
            paths[i] = [starts[i]]  # Start = goal case
            continue
            
        if i >= len(robot_data) or not robot_data[i]['path']:
            paths[i] = []  # No path found
            continue
            
        goal = goals[i]
        path = robot_data[i]['path']
        
        # Ensure path includes start and goes to goal
        if goal in path:
            idx = path.index(goal)
            paths[i] = path[:idx+1]
        else:
            paths[i] = []  # No valid path to goal

    return paths

'''
def a_star(maze: np.ndarray, starts: list, goals: list):
    """
    Multi-robot A* with collision avoidance, backtracking, and
    reservation of a robot's goal cell for all future time steps after arrival.
    Returns only the path up to (and including) the first arrival at the goal for each robot.
    """
    num_robots = len(starts)
    paths = [[] for _ in range(num_robots)]
    reservations = {}  # (x, y, t): robot_id

    # For each robot, record when it reaches goal
    robot_goal_time = [None] * num_robots

    # Initialize robot data
    robot_data = []
    for i in range(num_robots):
        start = starts[i]
        goal = goals[i]
        if start == goal:
            paths[i] = [start]
            robot_goal_time[i] = 0
            continue
        fringe = PriorityQueue()
        h_start = heuristic(start, goal)
        fringe.put((h_start, 0, start, 0))
        robot_data.append({
            'fringe': fringe,
            'prev_node': {},
            'cum_costs': {(start, 0): 0},
            'total_costs': {(start, 0): h_start},
            'goal': goal,
            'path': [],
            'active': True,
            'current_pos': start,
            'current_time': 0,
            'last_positions': [start]
        })
    global_time = 0
    active_robots = len([d for d in robot_data if d.get('active', False)]) + sum(1 for i in range(num_robots) if starts[i] == goals[i])
    max_time = maze.size * 2

    while active_robots > 0 and global_time < max_time:
        global_time += 1
        planned_moves = {}

        # Phase 1: Plan moves and expand A* for each robot
        for i in range(len(robot_data)):
            if not robot_data[i]['active']:
                continue
            while not robot_data[i]['fringe'].empty():
                f, g, current, t = robot_data[i]['fringe'].queue[0]
                if t > global_time:
                    break
                robot_data[i]['fringe'].get()

                # --- Goal check ---
                if current == robot_data[i]['goal']:
                    path = []
                    node = (current, t)
                    while node in robot_data[i]['prev_node']:
                        path.append((node[0][0], node[0][1], node[1]))
                        node = robot_data[i]['prev_node'][node]
                    path.reverse()
                    robot_data[i]['path'] = [(x, y) for x, y, t in path]
                    robot_data[i]['active'] = False
                    active_robots -= 1
                    robot_goal_time[i] = t
                    # Reserve this goal cell for all future times (fix collision bug)
                    for future_t in range(t, max_time):
                        reservations[(current, future_t)] = i
                    break

                # Generate moves/neighbors
                for neighbor in get_neighbors_4d(maze, current):
                    new_time = t + 1
                    collision = False
                    if (neighbor, new_time) in reservations:
                        collision = True
                    if (neighbor, t) in reservations:
                        swapping_robot = reservations[(neighbor, t)]
                        if (current, new_time) in reservations:
                            if reservations[(current, new_time)] == swapping_robot:
                                collision = True
                    if len(robot_data[i]['last_positions']) >= 2:
                        prev_pos = robot_data[i]['last_positions'][-2]
                        if neighbor == prev_pos and (prev_pos, new_time) in reservations:
                            collision = True
                    if collision:
                        continue
                    edge_cost = 1
                    tentative_g = g + edge_cost
                    new_f = tentative_g + heuristic(neighbor, robot_data[i]['goal'])
                    if (neighbor, new_time) not in robot_data[i]['cum_costs'] or \
                            tentative_g < robot_data[i]['cum_costs'].get((neighbor, new_time), float('inf')):
                        robot_data[i]['prev_node'][(neighbor, new_time)] = (current, t)
                        robot_data[i]['cum_costs'][(neighbor, new_time)] = tentative_g
                        robot_data[i]['total_costs'][(neighbor, new_time)] = new_f
                        robot_data[i]['fringe'].put((new_f, tentative_g, neighbor, new_time))
                        robot_data[i]['last_positions'] = robot_data[i]['last_positions'][-1:] + [neighbor]

                # Wait action or forced backtrack
                if (current, t + 1) not in reservations:
                    robot_data[i]['fringe'].put((f + 1, g + 1, current, t + 1))
                else:
                    if len(robot_data[i]['last_positions']) > 1:
                        prev_pos = robot_data[i]['last_positions'][-2]
                        robot_data[i]['fringe'].put((f + 2, g + 1, prev_pos, t + 1))

        # Phase 2: Reserve planned positions for timestep t+1
        for i in range(len(robot_data)):
            if not robot_data[i]['active']:
                continue
            if not robot_data[i]['fringe'].empty():
                f, g, pos, t = robot_data[i]['fringe'].queue[0]
                if t == global_time + 1:
                    if (pos, t) not in planned_moves:
                        planned_moves[(pos, t)] = []
                    planned_moves[(pos, t)].append(i)

        # Phase 3: Conflict resolution
        for (pos, t), robots in planned_moves.items():
            if len(robots) > 1:
                robots.sort(key=lambda x: heuristic(robot_data[x]['current_pos'], robot_data[x]['goal']))
                winner = robots[0]
                for robot in robots[1:]:
                    new_queue = []
                    while not robot_data[robot]['fringe'].empty():
                        item = robot_data[robot]['fringe'].get()
                        new_queue.append(item)
                    current_pos = robot_data[robot]['current_pos']
                    prev_positions = robot_data[robot]['last_positions'][-3:]
                    for p in reversed(prev_positions):
                        new_time = global_time + 1
                        if (p, new_time) not in reservations:
                            g = robot_data[robot]['cum_costs'].get((current_pos, global_time), 0) + 1
                            f = g + heuristic(p, robot_data[robot]['goal'])
                            new_queue.append((f, g, p, new_time))
                    for item in sorted(new_queue, key=lambda x: x[0]):
                        robot_data[robot]['fringe'].put(item)
                reservations[(pos, t)] = winner
            else:
                reservations[(pos, t)] = robots[0]

    # --- Now, for each robot, only return path up to (including) first goal arrival ---
    for i in range(num_robots):
        # If precomputed path exists
        path = robot_data[i]['path'] if i < len(robot_data) and robot_data[i]['path'] else paths[i]
        if not path or starts[i] == goals[i]:
            paths[i] = [starts[i]]
            continue
        goal = goals[i]
        try:
            idx = path.index(goal)
            paths[i] = path[:idx+1]
        except ValueError:
            paths[i] = path
    return paths
'''


# D* Algorithm for multiple robots - SPACE FOR YOUR IMPLEMENTATION
def d_star(maze: np.ndarray, starts: list, goals: list):
    # Tag constants representing node states
    NEW, OPEN, CLOSED = 0, 1, 2

    # Manhattan heuristic (for extensibility)
    def manhattan(node1, node2):
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])

    # Get 4-connected neighbors (up, down, left, right) not in the blocked set
    def get_neighbors_4d(node, blocked_set):
        x, y = node
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if (0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1]
                and maze[nx, ny] == 0 and (nx, ny) not in blocked_set):
                neighbors.append((nx, ny))
        return neighbors

    def initialize_robot(goal, blocked_set):
        tags, h, k, backpointers, open_list = {}, {}, {}, {}, {}

        def get_tag(node): return tags.get(node, NEW)
        def get_h(node): return h.get(node, float('inf'))
        def get_k(node): return k.get(node, float('inf'))

        if goal not in blocked_set:
            tags[goal] = OPEN
            h[goal] = 0
            k[goal] = 0
            open_list[0] = [goal]

        return tags, h, k, backpointers, open_list, get_tag, get_h, get_k

    def process_state(tags, h, k, backpointers, open_list, blocked_set, get_tag, get_h, get_k):
        if not open_list: return -1

        min_k = min(open_list.keys())
        x = open_list[min_k].pop()
        if not open_list[min_k]: del open_list[min_k]

        k_old = min_k
        tags[x] = CLOSED
        neighbors = get_neighbors_4d(x, blocked_set)

        if k_old < get_h(x):
            for y in neighbors:
                if get_tag(y) != NEW and get_h(y) <= k_old and get_h(x) > get_h(y) + 1:
                    backpointers[x] = y
                    h[x] = get_h(y) + 1

        elif k_old == get_h(x):
            for y in neighbors:
                if (get_tag(y) == NEW or 
                    (backpointers.get(y) == x and get_h(y) != get_h(x) + 1) or 
                    (backpointers.get(y) != x and get_h(y) > get_h(x) + 1)):
                    backpointers[y] = x
                    h[y] = get_h(x) + 1
                    insert(y, h[y], tags, h, k, open_list)

        else:
            for y in neighbors:
                if get_tag(y) == NEW or (backpointers.get(y) == x and get_h(y) != get_h(x) + 1):
                    backpointers[y] = x
                    h[y] = get_h(x) + 1
                    insert(y, h[y], tags, h, k, open_list)
                elif backpointers.get(y) != x and get_h(y) > get_h(x) + 1:
                    insert(x, h[x], tags, h, k, open_list)
                elif (backpointers.get(y) != x and get_h(x) > get_h(y) + 1 and
                      get_tag(y) == CLOSED and get_h(y) > k_old):
                    insert(y, h[y], tags, h, k, open_list)

        return min(open_list.keys()) if open_list else -1

    def insert(node, new_h, tags, h, k, open_list):
        node_tag = tags.get(node, NEW)
        if node_tag == NEW:
            h[node] = new_h
            k[node] = new_h
            tags[node] = OPEN
        elif node_tag == OPEN:
            k[node] = min(k.get(node, float('inf')), new_h)
        elif node_tag == CLOSED:
            h[node] = new_h
            k[node] = new_h
            tags[node] = OPEN
        if k[node] not in open_list:
            open_list[k[node]] = []
        if node not in open_list[k[node]]:
            open_list[k[node]].append(node)

    def get_path(start, goal, backpointers):
        if start == goal: return [start]
        path = [start]
        current = start
        while current != goal:
            if current not in backpointers:
                return []
            current = backpointers[current]
            path.append(current)
        return path

    def compute_full_path(current_pos, goal, blocked_set):
        tags, h, k, backpointers, open_list, get_tag, get_h, get_k = initialize_robot(goal, blocked_set)
        max_iterations = maze.shape[0] * maze.shape[1] * 4
        iterations = 0
        while iterations < max_iterations:
            iterations += 1
            k_min = process_state(tags, h, k, backpointers, open_list, blocked_set, get_tag, get_h, get_k)
            if k_min == -1 or get_tag(current_pos) == CLOSED:
                break
        return get_path(current_pos, goal, backpointers)

    permanent_blocked = set(tuple(p) for p in np.argwhere(maze == 1))
    num_robots = len(starts)
    if num_robots == 0:
        return []
    if len(goals) != num_robots:
        raise ValueError("Number of starts and goals must match")

    valid_robots = []
    for i in range(num_robots):
        if (0 <= starts[i][0] < maze.shape[0] and 0 <= starts[i][1] < maze.shape[1] and
            0 <= goals[i][0] < maze.shape[0] and 0 <= goals[i][1] < maze.shape[1] and
            maze[starts[i][0], starts[i][1]] == 0 and maze[goals[i][0], goals[i][1]] == 0):
            valid_robots.append(i)

    planning_paths = [[] for _ in range(num_robots)]
    final_paths = [[] for _ in range(num_robots)]
    current_positions = starts.copy()
    reached_goals = [False] * num_robots
    unable_to_reach = [False] * num_robots

    for i in range(num_robots):
        final_paths[i].append(starts[i])
        if starts[i] == goals[i]:
            reached_goals[i] = True
            permanent_blocked.add(goals[i])

    initial_path_lengths = [float('inf')] * num_robots
    for i in valid_robots:
        if reached_goals[i]:
            planning_paths[i] = [starts[i]]
            initial_path_lengths[i] = 1
            continue
        path = compute_full_path(starts[i], goals[i], permanent_blocked)
        if not path:
            unable_to_reach[i] = True
            continue
        planning_paths[i] = path
        initial_path_lengths[i] = len(path)

    sum_path_lengths = sum(len(p) for p in planning_paths if p)
    max_time_steps = max(
        sum_path_lengths * 2,
        5 * max((len(p) for p in planning_paths if p), default=100) + num_robots * 20,
        100
    )

    time_step = 0
    while (not all(reached_goals) and
           not all(reached_goals[i] or unable_to_reach[i] for i in range(num_robots)) and
           time_step < max_time_steps):

        for i in range(num_robots):
            if not reached_goals[i] and not unable_to_reach[i] and current_positions[i] == goals[i]:
                reached_goals[i] = True
                permanent_blocked.add(goals[i])

        next_positions = current_positions.copy()
        original_next_positions = current_positions.copy()
        for i in range(num_robots):
            if reached_goals[i] or unable_to_reach[i]:
                continue
            if not planning_paths[i] or len(planning_paths[i]) <= 1:
                path = compute_full_path(current_positions[i], goals[i], permanent_blocked)
                if not path:
                    unable_to_reach[i] = True
                    continue
                planning_paths[i] = path
            next_positions[i] = planning_paths[i][1] if len(planning_paths[i]) > 1 else current_positions[i]
            original_next_positions[i] = next_positions[i]

        vertex_conflicts = {}
        for i in range(num_robots):
            if not reached_goals[i] and not unable_to_reach[i]:
                pos = next_positions[i]
                vertex_conflicts.setdefault(pos, []).append(i)

        edge_conflicts = set()
        for i in range(num_robots):
            for j in range(i+1, num_robots):
                if (not reached_goals[i] and not reached_goals[j] and
                    not unable_to_reach[i] and not unable_to_reach[j] and
                    next_positions[i] == current_positions[j] and
                    next_positions[j] == current_positions[i]):
                    edge_conflicts.add((i, j))

        robots_to_replan = set()
        for pos, robots in vertex_conflicts.items():
            if len(robots) > 1:
                def priority_key(robot_idx):
                    path = planning_paths[robot_idx]
                    if not path:
                        return (float('inf'), initial_path_lengths[robot_idx], robot_idx)
                    return (len(path), initial_path_lengths[robot_idx], robot_idx)
                robots.sort(key=priority_key)
                for i in robots[1:]:
                    robots_to_replan.add(i)
                    next_positions[i] = current_positions[i]

        for i, j in edge_conflicts:
            path_i = planning_paths[i] if planning_paths[i] else []
            path_j = planning_paths[j] if planning_paths[j] else []
            remaining_i = len(path_i)
            remaining_j = len(path_j)
            if (remaining_i < remaining_j or
                (remaining_i == remaining_j and (
                    initial_path_lengths[i] < initial_path_lengths[j] or
                    (initial_path_lengths[i] == initial_path_lengths[j] and i < j)))):
                robots_to_replan.add(j)
                next_positions[j] = current_positions[j]
            else:
                robots_to_replan.add(i)
                next_positions[i] = current_positions[i]

        for i in robots_to_replan:
            if reached_goals[i] or unable_to_reach[i]:
                continue
            temp_blocked = permanent_blocked.copy()
            for k in range(num_robots):
                if k != i and not reached_goals[k] and not unable_to_reach[k]:
                    temp_blocked.add(next_positions[k])
            new_path = compute_full_path(current_positions[i], goals[i], temp_blocked)
            if new_path:
                planning_paths[i] = new_path
                if len(new_path) > 1 and new_path[1] not in temp_blocked:
                    next_positions[i] = new_path[1]
            else:
                new_path = compute_full_path(current_positions[i], goals[i], permanent_blocked)
                if not new_path:
                    unable_to_reach[i] = True
                else:
                    planning_paths[i] = new_path

        for i in range(num_robots):
            if reached_goals[i]:
                final_paths[i].append(goals[i])
            elif unable_to_reach[i]:
                pass
            else:
                current_positions[i] = next_positions[i]
                final_paths[i].append(current_positions[i])
                if next_positions[i] != original_next_positions[i] or next_positions[i] == current_positions[i]:
                    planning_paths[i] = compute_full_path(current_positions[i], goals[i], permanent_blocked)
                    if not planning_paths[i]:
                        unable_to_reach[i] = True
                elif planning_paths[i] and len(planning_paths[i]) > 1:
                    planning_paths[i] = planning_paths[i][1:]

        time_step += 1

    # Only return the path *upto* (and including) first goal reach -- no padding!
    for i in range(num_robots):
        if not reached_goals[i]:
            final_paths[i] = []
        else:
            goal = goals[i]
            path = final_paths[i]
            try:
                idx = path.index(goal)
                final_paths[i] = path[:idx+1]
            except ValueError:
                final_paths[i] = path

    return final_paths



# M* Algorithm for multiple robots - SPACE FOR YOUR IMPLEMENTATION
'''
def m_star(maze: np.ndarray, starts: list, goals: list):
    """
    Implement M* Algorithm for multi-robot path planning.
    
    Args:
    - maze (np.ndarray): The maze grid.
    - starts (list of tuples): List of start positions.
    - goals (list of tuples): List of goal positions.
    
    Returns:
    - list of lists of tuples: Paths for each robot.
    """
    # Define M* problem class to interact with the algorithm
    class MStarProblem:
        def __init__(self, maze, starts, goals):
            self.maze = maze
            self.starts = starts
            self.goals = goals
            self.num_robots = len(starts)
            # Precompute optimal policies for each robot
            self.optimal_policies = self.compute_all_policies()
        
        def compute_all_policies(self):
            """Compute optimal policy for each robot using A*"""
            policies = {}
            
            for robot_idx in range(self.num_robots):
                # Compute optimal policy for each robot
                policy = self.compute_policy(robot_idx)
                policies[robot_idx] = policy
                
            return policies
        
        def compute_policy(self, robot_idx):
            """Compute optimal policy for a single robot using A*"""
            start = self.starts[robot_idx]
            goal = self.goals[robot_idx]
            
            # A* search for individual robot
            open_list = []
            closed_set = set()
            g_scores = {start: 0}
            came_from = {}
            f_scores = {start: heuristic(start, goal)}
            
            # Priority queue (manually implemented for tuple support)
            import heapq
            heapq.heappush(open_list, (f_scores[start], start))
            
            while open_list:
                _, current = heapq.heappop(open_list)
                
                if current == goal:
                    # Reconstruct policy
                    policy = {}
                    node = current
                    while node in came_from:
                        prev_node = came_from[node]
                        policy[prev_node] = node
                        node = prev_node
                    return policy
                
                closed_set.add(current)
                
                for neighbor in get_neighbors_4d(self.maze, current):
                    if neighbor in closed_set:
                        continue
                    
                    tentative_g = g_scores[current] + get_cost(current, neighbor)
                    
                    if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                        came_from[neighbor] = current
                        g_scores[neighbor] = tentative_g
                        f_scores[neighbor] = tentative_g + heuristic(neighbor, goal)
                        if neighbor not in [n for _, n in open_list]:
                            heapq.heappush(open_list, (f_scores[neighbor], neighbor))
            
            return {}  # No path found
        
        @property
        def initial_state(self):
            return tuple(self.starts)
        
        def is_goal(self, state):
            """Check if current joint state is the goal state"""
            return state == tuple(self.goals)
        
        def get_individual_actions(self, state, robot_idx):
            """Get valid actions for a single robot"""
            pos = state[robot_idx]
            return get_neighbors_4d(self.maze, pos)
        
        def apply_individual_action(self, state, robot_idx, action):
            """Apply action for a single robot"""
            new_state = list(state)
            new_state[robot_idx] = action
            return tuple(new_state)
        
        def g_cost(self, state, actions):
            """Calculate cost of an action (simplified as 1 for all moves)"""
            return 1
        
        def h_cost(self, state):
            """Heuristic estimate to goal (sum of individual heuristics)"""
            total_h = 0
            for robot_idx in range(self.num_robots):
                robot_pos = state[robot_idx]
                robot_goal = self.goals[robot_idx]
                total_h += heuristic(robot_pos, robot_goal)
            return total_h
        
        def collision_check(self, state):
            """Check for collisions in the joint state"""
            collision_set = set()
            
            # Check for position conflicts
            positions = {}
            for i, pos in enumerate(state):
                if pos in positions:
                    collision_set.add(i)
                    collision_set.add(positions[pos])
                else:
                    positions[pos] = i
            
            return collision_set
        
        def get_optimal_policy(self, robot_idx, state):
            """Return optimal action for robot from precomputed policy"""
            pos = state[robot_idx]
            policy = self.optimal_policies[robot_idx]
            
            if pos in policy:
                return policy[pos]
            
            # If no policy exists, use get_neighbors to find a valid move
            neighbors = get_neighbors_4d(self.maze, pos)
            if neighbors:
                # Choose neighbor closest to goal
                goal = self.goals[robot_idx]
                min_dist = float('inf')
                best_neighbor = None
                
                for neighbor in neighbors:
                    dist = heuristic(neighbor, goal)
                    if dist < min_dist:
                        min_dist = dist
                        best_neighbor = neighbor
                
                return best_neighbor
            
            # If no valid move, stay in place
            return pos
    
    # Create M* problem instance
    problem = MStarProblem(maze, starts, goals)
    
    # Helper functions for M* algorithm
def m_star(maze: np.ndarray, starts: list, goals: list):
    """
    Implement M* Algorithm for multi-robot path planning.
    
    Args:
    - maze (np.ndarray): The maze grid.
    - starts (list of tuples): List of start positions.
    - goals (list of tuples): List of goal positions.
    
    Returns:
    - list of lists of tuples: Paths for each robot.
    """
    # Define M* problem class to interact with the algorithm
    class MStarProblem:
        def __init__(self, maze, starts, goals):
            self.maze = maze
            self.starts = starts
            self.goals = goals
            self.num_robots = len(starts)
            # Precompute optimal policies for each robot
            self.optimal_policies = self.compute_all_policies()
        
        def compute_all_policies(self):
            """Compute optimal policy for each robot using A*"""
            policies = {}
            
            for robot_idx in range(self.num_robots):
                # Compute optimal policy for each robot
                policy = self.compute_policy(robot_idx)
                policies[robot_idx] = policy
                
            return policies
        
        def compute_policy(self, robot_idx):
            """Compute optimal policy for a single robot using A*"""
            start = self.starts[robot_idx]
            goal = self.goals[robot_idx]
            
            # A* search for individual robot
            open_list = []
            closed_set = set()
            g_scores = {start: 0}
            came_from = {}
            f_scores = {start: heuristic(start, goal)}
            
            # Priority queue (manually implemented for tuple support)
            import heapq
            heapq.heappush(open_list, (f_scores[start], 0, start))  # f, g, node
            
            while open_list:
                _, g, current = heapq.heappop(open_list)
                
                if current == goal:
                    # Reconstruct policy
                    policy = {}
                    node = current
                    while node in came_from:
                        prev_node = came_from[node]
                        policy[prev_node] = node
                        node = prev_node
                    return policy
                
                closed_set.add(current)
                
                for neighbor in get_neighbors_4d(self.maze, current):
                    if neighbor in closed_set:
                        continue
                    
                    tentative_g = g + get_cost(current, neighbor)
                    
                    if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                        came_from[neighbor] = current
                        g_scores[neighbor] = tentative_g
                        f_scores[neighbor] = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_list, (f_scores[neighbor], tentative_g, neighbor))
            
            return {}  # No path found
        
        @property
        def initial_state(self):
            return tuple(self.starts)
        
        def is_goal(self, state):
            """Check if current joint state is the goal state"""
            return state == tuple(self.goals)
        
        def get_individual_actions(self, state, robot_idx):
            """Get valid actions for a single robot"""
            pos = state[robot_idx]
            return get_neighbors_4d(self.maze, pos) + [pos]  # Include staying in place
        
        def apply_individual_action(self, state, robot_idx, action):
            """Apply action for a single robot"""
            new_state = list(state)
            new_state[robot_idx] = action
            return tuple(new_state)
        
        def g_cost(self, state, actions):
            """Calculate cost of an action (simplified as 1 for all moves)"""
            return 1
        
        def h_cost(self, state):
            """Heuristic estimate to goal (sum of individual heuristics)"""
            total_h = 0
            for robot_idx in range(self.num_robots):
                robot_pos = state[robot_idx]
                robot_goal = self.goals[robot_idx]
                total_h += heuristic(robot_pos, robot_goal)
            return total_h
        
        def collision_check(self, state):
            """Check for collisions in the joint state"""
            collision_set = set()
            
            # Check for position conflicts
            positions = {}
            for i, pos in enumerate(state):
                if pos in positions:
                    collision_set.add(i)
                    collision_set.add(positions[pos])
                else:
                    positions[pos] = i
                    
            return collision_set if collision_set else set()
        
        def get_optimal_policy(self, robot_idx, state):
            """Return optimal action for robot from precomputed policy"""
            pos = state[robot_idx]
            policy = self.optimal_policies[robot_idx]
            
            if pos in policy:
                return policy[pos]
            elif pos == self.goals[robot_idx]:
                return pos  # Stay at goal
            else:
                # If no policy exists, use get_neighbors to find a valid move
                neighbors = get_neighbors_4d(self.maze, pos)
                if neighbors:
                    # Choose neighbor closest to goal
                    goal = self.goals[robot_idx]
                    min_dist = float('inf')
                    best_neighbor = None
                    
                    for neighbor in neighbors:
                        dist = heuristic(neighbor, goal)
                        if dist < min_dist:
                            min_dist = dist
                            best_neighbor = neighbor
                    
                    return best_neighbor
            
            # If no valid move, stay in place
            return pos
    
    # Run M* search
'''

def m_star(maze, starts, goals):
    """
    Implement M* Algorithm for multi-robot path planning.
    
    Args:
    - maze (np.ndarray): The maze grid.
    - starts (list of tuples): List of start positions.
    - goals (list of tuples): List of goal positions.
    
    Returns:
    - list of lists of tuples: Paths for each robot.
    """
    num_robots = len(starts)
    
    # Helper functions
    def heuristic(pos, goal):
        """Manhattan distance heuristic"""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def get_cost(current, neighbor):
        """Cost between adjacent cells"""
        return 1
    
    def get_neighbors_4d(maze, pos):
        """Get valid 4-directional neighbors"""
        x, y = pos
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            # Check if within bounds and not a wall
            if (0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and 
                maze[nx, ny] == 0):  # Assuming 0 is free space
                neighbors.append((nx, ny))
                
        return neighbors
    
    # Compute optimal policies for each robot
    def compute_policy(maze, start, goal):
        """Compute optimal policy for a single robot using A*"""
        open_list = [(heuristic(start, goal), 0, start)]  # f, g, node
        closed_set = set()
        g_scores = {start: 0}
        came_from = {}
        f_scores = {start: heuristic(start, goal)}
        
        while open_list:
            # Manual priority queue - find min f-score
            min_idx = 0
            for i in range(1, len(open_list)):
                if open_list[i][0] < open_list[min_idx][0]:
                    min_idx = i
            
            # Pop the node with lowest f-score
            f, g, current = open_list.pop(min_idx)
            
            if current == goal:
                # Reconstruct policy
                policy = {}
                node = current
                while node in came_from:
                    prev_node = came_from[node]
                    policy[prev_node] = node
                    node = prev_node
                return policy
            
            closed_set.add(current)
            
            for neighbor in get_neighbors_4d(maze, current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = g + get_cost(current, neighbor)
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_scores[neighbor] = tentative_g + heuristic(neighbor, goal)
                    open_list.append((f_scores[neighbor], tentative_g, neighbor))
        
        return {}  # No path found
    
    # Compute optimal policies for all robots
    optimal_policies = {}
    for robot_idx in range(num_robots):
        optimal_policies[robot_idx] = compute_policy(maze, starts[robot_idx], goals[robot_idx])
    
    # Get optimal action for a robot from precomputed policy
    def get_optimal_policy(robot_idx, state):
        pos = state[robot_idx]
        policy = optimal_policies[robot_idx]
        
        if pos in policy:
            return policy[pos]
        elif pos == goals[robot_idx]:
            return pos  # Stay at goal
        else:
            # If no policy exists, find a valid move
            neighbors = get_neighbors_4d(maze, pos)
            if neighbors:
                # Choose neighbor closest to goal
                goal = goals[robot_idx]
                min_dist = float('inf')
                best_neighbor = None
                
                for neighbor in neighbors:
                    dist = heuristic(neighbor, goal)
                    if dist < min_dist:
                        min_dist = dist
                        best_neighbor = neighbor
                
                return best_neighbor
        
        # If no valid move, stay in place
        return pos
    
    # Check for collisions in the joint state
    def collision_check(state):
        collision_set = set()
        
        # Check for position conflicts
        positions = {}
        for i, pos in enumerate(state):
            if pos in positions:
                collision_set.add(i)
                collision_set.add(positions[pos])
            else:
                positions[pos] = i
                
        return collision_set
    
    # Calculate heuristic cost for joint state
    def h_cost(state):
        total_h = 0
        for robot_idx in range(num_robots):
            robot_pos = state[robot_idx]
            robot_goal = goals[robot_idx]
            total_h += heuristic(robot_pos, robot_goal)
        return total_h
    
    # Get valid actions for a single robot
    def get_individual_actions(state, robot_idx):
        pos = state[robot_idx]
        return get_neighbors_4d(maze, pos) + [pos]  # Include staying in place
    
    # Check if current joint state is the goal state
    def is_goal(state):
        return state == tuple(goals)
    
    # M* Search Algorithm
    def m_star_search():
        # Initialize open list and closed set
        open_list = [(h_cost(tuple(starts)), 0, tuple(starts))]  # f, time added, state
        closed_set = set()
        
        # Initialize collision sets and backpropagation sets
        collision_sets = {tuple(starts): set()}
        backprop_sets = {tuple(starts): set()}
        
        # Dictionary to store node information
        node_info = {
            tuple(starts): {
                'g': 0,
                'h': h_cost(tuple(starts)),
                'f': h_cost(tuple(starts)),
                'parent': None,
                'collision_set': set()
            }
        }
        
        counter = 1  # Counter for tie-breaking
        
        while open_list:
            # Get node with lowest f-value
            min_idx = 0
            for i in range(1, len(open_list)):
                if open_list[i][0] < open_list[min_idx][0]:
                    min_idx = i
            
            # Pop the node with lowest f-score
            _, _, current_state = open_list.pop(min_idx)
            
            # Skip if already processed with better path
            if current_state in closed_set:
                continue
                
            # Get current node information
            current_node = node_info[current_state]
            
            # Check if goal reached
            if is_goal(current_state):
                # Reconstruct paths
                final_paths = [[] for _ in range(num_robots)]
                
                # Start with goal state
                state = current_state
                while state is not None:
                    # Add each robot's position to its path
                    for i in range(num_robots):
                        final_paths[i].append(state[i])
                    
                    # Move to parent state
                    if node_info[state]['parent'] is not None:
                        state = node_info[state]['parent']
                    else:
                        break
                
                # Reverse paths and return
                for i in range(num_robots):
                    final_paths[i].reverse()
                
                return final_paths
            
            # Add to closed set
            closed_set.add(current_state)
            
            # Check for collisions
            collisions = collision_check(current_state)
            current_collision_set = collision_sets.get(current_state, set())
            
            # Update collision set if needed
            if collisions:
                new_collision_set = current_collision_set.union(collisions)
                if new_collision_set != current_collision_set:
                    collision_sets[current_state] = new_collision_set
                    
                    # Backpropagate collision information
                    parent_state = current_node['parent']
                    if parent_state is not None:
                        if parent_state not in backprop_sets:
                            backprop_sets[parent_state] = set()
                        backprop_sets[parent_state].add(current_state)
                        
                        if parent_state in collision_sets:
                            parent_collision_set = collision_sets[parent_state]
                            updated_parent_set = parent_collision_set.union(new_collision_set)
                            if updated_parent_set != parent_collision_set:
                                collision_sets[parent_state] = updated_parent_set
                                
                                # Reopen parent if needed
                                if parent_state in closed_set:
                                    closed_set.remove(parent_state)
                                    parent_f = node_info[parent_state]['f']
                                    open_list.append((parent_f, counter, parent_state))
                                    counter += 1
                
                # Use updated collision set
                current_collision_set = new_collision_set
            
            # Generate successors based on collision set
            if not current_collision_set:
                # No collisions, use optimal policy for all robots
                new_state = []
                for robot_idx in range(num_robots):
                    optimal_action = get_optimal_policy(robot_idx, current_state)
                    new_state.append(optimal_action)
                new_state = tuple(new_state)
                
                # Calculate cost
                g = current_node['g'] + 1  # Simplified cost
                h = h_cost(new_state)
                f = g + h
                
                # Check if better path found
                if new_state not in node_info or g < node_info[new_state]['g']:
                    node_info[new_state] = {
                        'g': g,
                        'h': h,
                        'f': f,
                        'parent': current_state,
                        'collision_set': set()
                    }
                    
                    if new_state not in closed_set:
                        open_list.append((f, counter, new_state))
                        counter += 1
            else:
                # Generate all combinations for robots in collision set
                # For robots not in collision set, use optimal policy
                
                # Create a list of possible actions for each robot
                robot_actions = []
                for robot_idx in range(num_robots):
                    if robot_idx in current_collision_set:
                        actions = get_individual_actions(current_state, robot_idx)
                        robot_actions.append(actions)
                    else:
                        optimal_action = get_optimal_policy(robot_idx, current_state)
                        robot_actions.append([optimal_action])
                
                # Generate all combinations using recursive function
                def generate_combinations(current_combo, robot_idx):
                    if robot_idx == num_robots:
                        # Calculate cost
                        new_state = tuple(current_combo)
                        g = current_node['g'] + 1  # Simplified cost
                        h = h_cost(new_state)
                        f = g + h
                        
                        # Check if better path found
                        if new_state not in node_info or g < node_info[new_state]['g']:
                            node_info[new_state] = {
                                'g': g,
                                'h': h,
                                'f': f,
                                'parent': current_state,
                                'collision_set': set()
                            }
                            
                            if new_state not in closed_set:
                                nonlocal counter
                                open_list.append((f, counter, new_state))
                                counter += 1
                        return
                    
                    # Try each action for the current robot
                    for action in robot_actions[robot_idx]:
                        current_combo.append(action)
                        generate_combinations(current_combo, robot_idx + 1)
                        current_combo.pop()  # Backtrack
                
                # Start combination generation
                generate_combinations([], 0)
        
        # No solution found
        return None
    
    # Run search
    paths = m_star_search()
    
    # If no solution found, return default path
    if not paths:
        return [[(start[0], start[1]), (goal[0], goal[1])] for start, goal in zip(starts, goals)]
    
    return paths


if __name__ == "__main__":
    obstacle_type = int(input("Enter obstacle type (1-5): "))
    maze = generate_maze(obstacle_type)
    starts, goals = generate_start_goal(maze)
    
    # Evaluate the algorithms
    from Search_Algorithms_Evaluator import evaluate_algorithms
    evaluate_algorithms(obstacle_type)