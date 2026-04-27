import numpy as np

# FUNCIONES OBJETIVO

def sphere(x):
    return np.sum(x**2)

def schwefel(x):
    V = 4189.829101
    # el optimo de Schwefel da 0 cuando x ~ 420.9687
    return 10 * V + np.sum(-x * np.sin(np.sqrt(np.abs(x))))

def griewank(x):
    i = np.arange(1, len(x) + 1)
    sum_term = np.sum(x**2 / 4000.0)
    prod_term = np.prod(np.cos(x / np.sqrt(i)))
    return 1 + sum_term - prod_term

# AGS

def ags(fitness_func, bounds, dim, pop_size=100, generations=500, mutation_rate=0.1):
    min_val, max_val = bounds
    
    # poblacion aleatoria
    population = np.random.uniform(min_val, max_val, (pop_size, dim))
    best_solution = None
    best_fitness = float('inf')

    for gen in range(generations):
        # evaluar puntaje (buscamos el mínimo)
        fitness = np.array([fitness_func(ind) for ind in population])
        
        # guardar el mejor de la generación
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < best_fitness:
            best_fitness = fitness[min_idx]
            best_solution = population[min_idx]

        # seleccion
        new_population = []
        for _ in range(pop_size):
            # elegimos 3 individuos al azar y nos quedamos con el mejor
            tournament = np.random.choice(pop_size, 3)
            winner_idx = tournament[np.argmin(fitness[tournament])]
            parent1 = population[winner_idx]
            
            tournament = np.random.choice(pop_size, 3)
            winner_idx = tournament[np.argmin(fitness[tournament])]
            parent2 = population[winner_idx]

            # cruce
            mask = np.random.rand(dim) < 0.5
            child = np.where(mask, parent1, parent2)

            # mutacion (gaussiana con limite de espacio)
            if np.random.rand() < mutation_rate:
                mutation_step = np.random.normal(0, (max_val - min_val) * 0.1, dim)
                child += mutation_step
                # asegurar que el hijo no se salga de los limites
                child = np.clip(child, min_val, max_val)

            new_population.append(child)

        population = np.array(new_population)

    return best_solution, best_fitness

if __name__ == "__main__":
    print("Ejecutando AGS...\n")

    # sphere: 2 dimensiones, limites [-5, 5]
    best_x, best_f = ags(sphere, bounds=(-5, 5), dim=2)
    print(f"Sphere -> Mejor Puntaje: {best_f:.6f} | Coordenadas: {np.round(best_x, 4)}")

    # schwefel: 10 dimensiones, limites [-500, 500]
    best_x, best_f = ags(schwefel, bounds=(-500, 500), dim=10)
    print(f"Schwefel -> Mejor Puntaje: {best_f:.6f}")

    # griewank: 10 dimensiones, limites [-600, 600]
    best_x, best_f = ags(griewank, bounds=(-600, 600), dim=10)
    print(f"Griewank -> Mejor Puntaje: {best_f:.6f}")