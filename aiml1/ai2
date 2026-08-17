from queue import PriorityQueue


def best_first_search(graph, start, target, heuristics):
    visited = set()
    pq = PriorityQueue()

    # Add starting node
    pq.put((heuristics[start], start, [start]))
    visited.add(start)

    while not pq.empty():
        h, node, path = pq.get()

        # Check if target is reached
        if node == target:
            return path, h

        # Explore neighboring nodes
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                pq.put(
                    (heuristics[neighbor],
                     neighbor,
                     path + [neighbor])
                )

    return None, None


# Example Usage
if __name__ == "__main__":

    graph = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F'],
        'D': [],
        'E': [],
        'F': []
    }

    # Heuristic values
    heuristics = {
        'A': 10,
        'B': 8,
        'C': 5,
        'D': 6,
        'E': 4,
        'F': 0
    }

    path, cost = best_first_search(
        graph,
        'A',
        'F',
        heuristics
    )

    print(
        f"Path found by Best-First Search: "
        f"{path} with Target Heuristic: {cost}"
    )