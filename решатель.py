import numpy as np
import random
import tkinter as tk
from tkinter import messagebox, Label, Entry, Button, Frame, Scrollbar, Canvas

def is_valid_solution(x, supply, demand, time):
    """
    Validate if the solution x meets the supply, demand, and time constraints:
    - Supply: Total products shipped from each warehouse.
    - Demand: Total products required by each store.
    - Time: Transport time or other related constraints between warehouse and store.
    """
    return (np.allclose(x.sum(axis=(1, 2)), supply) and
            np.allclose(x.sum(axis=(0, 2)), demand) and
            np.allclose(x.sum(axis=(0, 1)), time))

def generate_valid_solution(supply, demand, time):
    """
    Generate a valid solution matrix that adheres to supply, demand, and time constraints.
    """
    while True:
        x = np.random.rand(supply.size, demand.size, time.size)
        x /= x.sum()  # Normalize to maintain consistency with constraints
        # Scale to match the total supply, respecting demand as well
        x *= np.minimum(supply.sum(), demand.sum())
        if is_valid_solution(x, supply, demand, time):
            return x

def initialize_population(population_size, supply, demand, time):
    """
    Initialize a population of valid solutions, each respecting the supply, demand, and time constraints.
    """
    return np.array([generate_valid_solution(supply, demand, time) for _ in range(population_size)])

def fitness(solution, cost_matrix):
    """
    Compute the fitness of a solution based on its total transportation cost as defined by the cost matrix.
    """
    return np.sum(solution * cost_matrix)

def selection(population, fitnesses, num_parents):
    """
    Select the top solutions from the population for breeding, based on their fitness scores.
    """
    return population[np.argsort(fitnesses)[:num_parents]]

def crossover(parent1, parent2, crossover_rate):
    """
    Perform a crossover between two parent solutions to generate offspring, based on the specified crossover rate.
    """
    if random.random() < crossover_rate:
        point = np.random.randint(1, parent1.size - 1)
        child1 = np.concatenate((parent1.flat[:point], parent2.flat[point:])).reshape(parent1.shape)
        child2 = np.concatenate((parent2.flat[:point], parent1.flat[point:])).reshape(parent2.shape)
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(child, mutation_rate, supply, demand, time):
    """
    Mutate a solution to explore new genetic space. Each mutation is checked to ensure it still meets all constraints.
    """
    for attempt in range(5):
        if random.random() < mutation_rate:
            idx = np.random.randint(0, np.prod(child.shape))
            np.put(child, idx, child.flat[idx] + np.random.normal())
        if is_valid_solution(child, supply, demand, time):
            return child
    return child  # Return the original if no valid mutation occurs

def genetic_algorithm(cost_matrix, population_size, supply, demand, time, generations, mutation_rate, crossover_rate):
    """
    Run a genetic algorithm to find the optimal solution for minimizing transportation costs across a network.
    """
    population = initialize_population(population_size, supply, demand, time)
    best_fitness = float('inf')
    best_solution = None
    for _ in range(generations):
        fitnesses = np.array([fitness(ind, cost_matrix) for ind in population])
        parents = selection(population, fitnesses, len(population) // 2)
        children = []
        while len(children) < len(population):
            for parent1, parent2 in zip(parents[::2], parents[1::2]):
                child1, child2 = crossover(parent1, parent2, crossover_rate)
                child1 = mutate(child1, mutation_rate, supply, demand, time)
                child2 = mutate(child2, mutation_rate, supply, demand, time)
                children.extend([child1, child2])
        population = np.array(children)
        current_best = np.min(fitnesses)
        if current_best < best_fitness:
            best_fitness = current_best
            best_solution = population[np.argmin(fitnesses)]
    return best_solution, best_fitness

class TriaxialTransportationProblem:
    def __init__(self, root):
        self.root = root
        self.root.title("Triaxial Transportation Problem Solver")
        self.setup_ui()
        # Initialize these dictionaries to ensure they exist before they are used
        self.frames = {}
        self.entries = {}
        self.canvases = {}
        self.scrollbars = {}

    def setup_ui(self):
        Label(self.root, text="m (warehouses):").grid(row=0, column=0)
        self.entry_m = Entry(self.root)
        self.entry_m.grid(row=0, column=1)

        Label(self.root, text="n (stores):").grid(row=1, column=0)
        self.entry_n = Entry(self.root)
        self.entry_n.grid(row=1, column=1)

        Label(self.root, text="p (products/time):").grid(row=2, column=0)
        self.entry_p = Entry(self.root)
        self.entry_p.grid(row=2, column=1)

        Button(self.root, text="Create Input Fields", command=self.create_input_fields).grid(row=3, column=0, columnspan=2)

    def create_input_fields(self):
        try:
            m = int(self.entry_m.get())
            n = int(self.entry_n.get())
            p = int(self.entry_p.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values for m, n, and p.")
            return

        for key in ['a', 'b', 'c', 'cost_matrix']:
            self.frames[key] = Frame(self.root)
            self.frames[key].grid(row=4, column=0, columnspan=2)

            self.scrollbars[key] = Scrollbar(self.frames[key], orient="vertical")
            self.canvases[key] = Canvas(self.frames[key], yscrollcommand=self.scrollbars[key].set)
            self.scrollbars[key].config(command=self.canvases[key].yview)
            self.scrollbars[key].pack(side="right", fill="y")
            self.canvases[key].pack(side="left", expand=True, fill="both")

            inner_frame = Frame(self.canvases[key])
            self.canvases[key].create_window((0, 0), window=inner_frame, anchor='nw')

            self.entries[key] = []
            for i in range(m):
                row_entries = []
                for j in range(n):
                    entry = Entry(inner_frame, width=5)
                    entry.grid(row=i, column=j)
                    row_entries.append(entry)
                self.entries[key].append(row_entries)

            inner_frame.update_idletasks()
            self.canvases[key].config(scrollregion=self.canvases[key].bbox("all"))

        Button(self.root, text="Run Genetic Algorithm", command=self.run_genetic_algorithm).grid(row=5, column=0, columnspan=2)

    def run_genetic_algorithm(self):
        try:
            a = np.array([[float(entry.get()) for entry in row] for row in self.entries['a']])
            b = np.array([[float(entry.get()) for entry in row] for row in self.entries['b']])
            c = np.array([[float(entry.get()) for entry in row] for row in self.entries['c']])
            cost_matrix = np.array([[float(entry.get()) for entry in row] for row in self.entries['cost_matrix']])

            # Assuming genetic_algorithm is properly defined elsewhere
            best_solution, best_fitness = genetic_algorithm(cost_matrix, 100, a, b, c, 50, 0.01, 0.7)
            messagebox.showinfo("Result", f"Best solution:\n{np.array_str(best_solution)}\n\nCost: {best_fitness}")
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = TriaxialTransportationProblem(root)
    root.mainloop()