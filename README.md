# 🧩 Maze Generator and Solver

A Python program that builds a maze and then solves the path between 2 nodes, all visualised in Pygame. The current version supports 4 generation and solving algorithms. Mainly built to be modular to be implemented into other programs.

## 🛠️ Technologies
```
Python
```
```
Pygame
```

## ⚙️ Features
Here's what you can do with this Maze Generator and Solver:

+ **Animated Generation of Mazes**: Choose between 4 generation algorithms and watch the visuals of the generation algorithm:
  - Recursive Backtracker: Explores forward until stuck, then backtracks to set each cell exactly once
  - Prim's Algorithm: Randomly expanding from an initial cell, adding one new cell at a time through selected walls
  - Kruskal's Algorithm: Randomly removing walls only when they connect two separate cell sets, preventing loops
  - Binary Tree: Opens one of two possible directions for each cell, producing a simple but directionally biased maze


+ **Animated Solving of Mazes**: Choose between 4 solving algorithms and watch the visuals of the solving algorithm:
  - Breadth First Search: Explores the maze level by level in a queue to find the shortest path
  - Depth First Search: Explores the maze by going as deep as possible along one path before backtracking
  - Djstrika's Shortest Path Algorithm: Moves to the closest unvisited point to find the shortest path in a weighted maze
  - A* Algorithm: Combines the distance traveled and a smart guess to the goal, prioritizing paths that seem fastest overall

+ **Modular Class System**: Implement each algorithm into any program you wish with the easy-to-understand class system.

+ **Test Function**: Example of how to use is provided at the bottom of the code.

## ⚓ Requirements

If you are on Python version < 3.12:
```
pip install pygame
```
Else (Python version => 3.12):
```
pip install pygame-ce
```

> [!WARNING]
> Generation animation freezes when user click out of pygame window, generation will continue however. (fix coming in next update)
