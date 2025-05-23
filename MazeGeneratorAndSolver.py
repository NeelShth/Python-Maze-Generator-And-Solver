#run "pip3 install requirements.txt" in cmd before running this
import pygame
import random
import heapq
from collections import deque
import time

#--------------------Config--------------------
CONTAINER_WIDTH = 800
CONTAINER_HEIGHT = 600

#Maze Size

#TODO - Fix Animation Of Maze Sizes > 14
COLS = 14
ROWS = 14

#Scaled Node Size
Node_SIZE = min(CONTAINER_WIDTH // COLS, CONTAINER_HEIGHT // ROWS)

#Final Dimensions Based On Maze Size
WIDTH = COLS * Node_SIZE
HEIGHT = ROWS * Node_SIZE

#Colours
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
PURPLE = (160, 32, 240)

#Pygame Setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Generator & Solver")
clock = pygame.time.Clock()

#Directions
DIRS = {'N': (0, -1), 'S': (0, 1), 'E': (1, 0), 'W': (-1, 0)}
#----------------------------------------------

#---------------------Node---------------------
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
        self.visited = False
        self.parent = None
        self.g = float('inf')
        self.h = 0

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

    def draw(self, surface, colour=None):
        x, y = self.x * Node_SIZE, self.y * Node_SIZE
        if colour:
            pygame.draw.rect(surface, colour, (x + 1, y + 1, Node_SIZE - 2, Node_SIZE - 2))
        if self.walls['N']:
            pygame.draw.line(surface, BLACK, (x, y), (x + Node_SIZE, y), 2)
        if self.walls['S']:
            pygame.draw.line(surface, BLACK, (x, y + Node_SIZE), (x + Node_SIZE, y + Node_SIZE), 2)
        if self.walls['E']:
            pygame.draw.line(surface, BLACK, (x + Node_SIZE, y), (x + Node_SIZE, y + Node_SIZE), 2)
        if self.walls['W']:
            pygame.draw.line(surface, BLACK, (x, y), (x, y + Node_SIZE), 2)

    def remove_wall(self, neighbour):
        dx = self.x - neighbour.x
        dy = self.y - neighbour.y
        #Removes Adjacent Walls Between Nodes
        if dx == 1:
            self.walls['W'] = False
            neighbour.walls['E'] = False
        elif dx == -1:
            self.walls['E'] = False
            neighbour.walls['W'] = False
        elif dy == 1:
            self.walls['N'] = False
            neighbour.walls['S'] = False
        elif dy == -1:
            self.walls['S'] = False
            neighbour.walls['N'] = False
#----------------------------------------------

#---------------------Maze---------------------
class Maze:
    def __init__(self, start_coords=None, end_coords=None):
        self.grid = [[Node(x, y) for y in range(ROWS)] for x in range(COLS)]
        self.start = None
        self.end = None
        if start_coords or end_coords:
            self.set_start_end(start_coords, end_coords)

    def set_start_end(self, start_coords=None, end_coords=None):
        if start_coords:
            sx, sy = start_coords
            self.start = self.get_Node(sx, sy)
        else:
            self.start = random.choice(random.choice(self.grid))

        if end_coords:
            ex, ey = end_coords
            self.end = self.get_Node(ex, ey)
        else:
            self.end = random.choice(random.choice(self.grid))
            while self.end == self.start:
                self.end = random.choice(random.choice(self.grid))

    def get_Node(self, x, y):
        if 0 <= x < COLS and 0 <= y < ROWS:
            return self.grid[x][y]
        return None

    def neighbours(self, Node, only_unvisited=False):
        results = []
        for dir, (dx, dy) in DIRS.items():
            neighbour = self.get_Node(Node.x + dx, Node.y + dy)
            if neighbour:
                if only_unvisited and neighbour.visited:
                    continue
                results.append((dir, neighbour))
        return results
    
    def randomize_start(self):
        self.start = random.choice(random.choice(self.grid))

    def randomize_end(self):
        self.end = random.choice(random.choice(self.grid))
        while self.end == self.start:
            self.end = random.choice(random.choice(self.grid))

    def randomize_start_end(self):
        self.start = random.choice(random.choice(self.grid))
        self.end = random.choice(random.choice(self.grid))
        while self.start == self.end:
            self.end = random.choice(random.choice(self.grid))
    
    def generate_maze(self, method="backtracker"):
        for row in self.grid:
            for Node in row:
                Node.visited = False
                
        if method == "backtracker":
            self.backtracker()
        elif method == "prim":
            self.prim()
        elif method == "kruskal":
            self.kruskal()
        elif method == "binary":
            self.binary_tree()

    def backtracker(self):
        stack = [self.start]
        self.start.visited = True

        while stack:
            current = stack[-1]
            neighbours = [n for _, n in self.neighbours(current, only_unvisited=True)]
            if neighbours:
                next_Node = random.choice(neighbours)
                current.remove_wall(next_Node)
                next_Node.visited = True
                stack.append(next_Node)
            else:
                stack.pop()

            self.draw()
            pygame.display.update()
            pygame.time.delay(10)

    def prim(self):
        frontier = [self.start]
        self.start.visited = True
        edges = []

        for _, neighbour in self.neighbours(self.start, only_unvisited=True):
            edges.append((self.start, neighbour))

        while edges:
            Node, neighbour = random.choice(edges)
            edges.remove((Node, neighbour))
            if not neighbour.visited:
                Node.remove_wall(neighbour)
                neighbour.visited = True
                for _, n in self.neighbours(neighbour, only_unvisited=True):
                    edges.append((neighbour, n))

            self.draw()
            pygame.display.update()
            pygame.time.delay(10)

    def kruskal(self):
        parent = {}
        def find(Node):
            while parent[Node] != Node:
                parent[Node] = parent[parent[Node]]
                Node = parent[Node]
            return Node

        def union(a, b):
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a
                return True
            return False

        for row in self.grid:
            for Node in row:
                parent[Node] = Node

        edges = []
        for x in range(COLS):
            for y in range(ROWS):
                Node = self.grid[x][y]
                for dx, dy in [(1, 0), (0, 1)]:
                    neighbour = self.get_Node(x + dx, y + dy)
                    if neighbour:
                        edges.append((Node, neighbour))

        random.shuffle(edges)

        for a, b in edges:
            if union(a, b):
                a.remove_wall(b)

            self.draw()
            pygame.display.update()
            pygame.time.delay(10)

    def binary_tree(self):
        for x in range(COLS):
            for y in range(ROWS):
                Node = self.grid[x][y]
                neighbours = []
                if y > 0:
                    neighbours.append(self.grid[x][y - 1])
                if x < COLS - 1:
                    neighbours.append(self.grid[x + 1][y])
                if neighbours:
                    neighbour = random.choice(neighbours)
                    Node.remove_wall(neighbour)

            self.draw()
            pygame.display.update()
            pygame.time.delay(10)

    def draw(self, path=None, open_set=None, closed_set=None):
        screen.fill(WHITE)
        for row in self.grid:
            for Node in row:
                colour = WHITE
                if self.start and Node == self.start:
                    colour = GREEN
                elif self.end and Node == self.end:
                    colour = RED
                elif path and Node in path:
                    colour = PURPLE
                elif open_set and Node in open_set:
                    colour = YELLOW
                elif closed_set and Node in closed_set:
                    colour = BLUE
                Node.draw(screen, colour=colour)
        pygame.display.flip()
#----------------------------------------------

#------------------Pathfinder------------------
class Pathfinder:
    def __init__(self, maze):
        self.maze = maze

    def reconstruct_path(self, end):
        path = []
        current = end
        while current.parent:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def bfs(self):
        start, end = self.maze.start, self.maze.end
        queue = deque([start])
        visited = set([start])

        while queue:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); exit()

            current = queue.popleft()
            if current == end:
                return self.reconstruct_path(end)

            for dir, neighbour in self.maze.neighbours(current):
                if not current.walls[dir] and neighbour not in visited:
                    neighbour.parent = current
                    visited.add(neighbour)
                    queue.append(neighbour)

            self.maze.draw(open_set=queue, closed_set=visited)

    def dfs(self):
        start, end = self.maze.start, self.maze.end
        stack = [start]
        visited = set([start])

        while stack:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); exit()

            current = stack.pop()
            if current == end:
                return self.reconstruct_path(end)

            for dir, neighbour in self.maze.neighbours(current):
                if not current.walls[dir] and neighbour not in visited:
                    neighbour.parent = current
                    visited.add(neighbour)
                    stack.append(neighbour)

            self.maze.draw(open_set=stack, closed_set=visited)

    def djstrika(self):
        start, end = self.maze.start, self.maze.end
        start.g = 0
        heap = [(0, start)]
        visited = set()

        while heap:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); exit()

            _, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current == end:
                return self.reconstruct_path(end)

            for dir, neighbour in self.maze.neighbours(current):
                if not current.walls[dir]:
                    temp_g = current.g + 1
                    if temp_g < neighbour.g:
                        neighbour.g = temp_g
                        neighbour.parent = current
                        heapq.heappush(heap, (neighbour.g, neighbour))

            self.maze.draw(open_set=[n for _, n in heap], closed_set=visited)

    def astar(self):
        start, end = self.maze.start, self.maze.end
        start.g = 0
        start.h = abs(end.x - start.x) + abs(end.y - start.y)
        heap = [(start.g + start.h, start)]
        visited = set()

        while heap:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); exit()

            _, current = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)

            if current == end:
                return self.reconstruct_path(end)

            for dir, neighbour in self.maze.neighbours(current):
                if not current.walls[dir]:
                    g = current.g + 1
                    h = abs(end.x - neighbour.x) + abs(end.y - neighbour.y)
                    if g < neighbour.g:
                        neighbour.g = g
                        neighbour.h = h
                        neighbour.parent = current
                        heapq.heappush(heap, (g + h, neighbour))

            self.maze.draw(open_set=[n for _, n in heap], closed_set=visited)

    def solve(self, method="astar"):
        if method == "astar":
            path = self.astar()
        elif method == "djstrika":
            path = self.djstrika()
        elif method == "bfs":
            path = self.bfs()
        elif method == "dfs":
            path = self.dfs()
        else:
            raise ValueError(f"Unknown method: {method}")
        if path:
            self.maze.draw(path=path)
#----------------------------------------------


#---------------------test---------------------
def test():
    print()
    start_time = time.time()
    
    #Initialise Maze
    maze = Maze()

    #Random Start & Random End
    #maze.randomize_start_end()
    
    #Custom Start & Random End
    #maze.set_start_end(start_coords=(0, 0))
    #maze.randomize_end()

    
    #Random Start & Custom End
    maze.randomize_start()
    maze.set_start_end(end_coords=(COLS-1, ROWS-1))

    #Custom Start & Custom End
    #maze.set_start_end(start_coords=(0, 0), end_coords=(COLS - 1, ROWS - 1))
    
    maze.generate_maze(method="prim") #Change to "backtracker", "prim", "binary", "kruskal"
    maze.draw()
    pygame.display.update()
    
    end_time = time.time() - start_time
    print(f"Generation Time: {end_time} seconds")
    start_time = time.time()
    
    solver = Pathfinder(maze)
    solver.solve(method="djstrika") #Change to "djstrika", "bfs", "dfs","astar"
    
    end_time = time.time() - start_time
    print(f"Solving Time: {end_time} seconds")
    
if __name__ == "__main__":
    test()