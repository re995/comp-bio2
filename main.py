# Computational Biology Exercise 2
# Genetic Algorithms - Futoshiki
# Yair Yariv Yardeni - 315009969
# Ron Even           - 313260317

# Dynamically download required modules
import subprocess
import sys
import pkg_resources

required = {"numpy", "matplotlib", "attrs"}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed
if missing:
    print("Installing missing libraries... Please wait")
    python_path = sys.executable
    subprocess.check_call([python_path, "-m", "pip", "install", *missing])

import itertools
import math
import pprint
import random
import sys
from abc import abstractmethod
from copy import deepcopy
from enum import Enum
from functools import lru_cache
from threading import Thread
from typing import List, Tuple

import attr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import AutoLocator


@attr.s
class Position:
    """
    Helper class for representing a row/column position
    """
    row = attr.ib(type=int)
    column = attr.ib(type=int)

    def __mul__(self, multiplier):
        return Position(self.row.__mul__(multiplier), self.column.__mul__(multiplier))

    def reverse(self):
        return Position(self.column, self.row)


@attr.s
class FilledCell:
    """
    Helper class for representing a pre-defined cell value and position
    """
    value = attr.ib(type=int)
    position = attr.ib(type=Position)


class SolverType(Enum):
    """
    Enum for representing type of generic algorithm generation selection
    """
    REGULAR = 1
    DARWIN = 2
    LAMARK = 3


class FutoshikiSolverBase:
    def __init__(self, puzzle_name, matrix_size, predefined_digits: List[FilledCell],
                 greater_than_signs: List[Tuple[Position, Position]],
                 initial_solution_count, solver_type: SolverType, max_generations_count):
        self._puzzle_name = puzzle_name
        self._matrix_size = matrix_size
        self._predefined_digits = predefined_digits
        self._greater_than_signs = greater_than_signs  # (a, b): a > b

        self._solutions = [self._generate_random_solution() for _ in range(initial_solution_count)]
        self._generation_count = 0

        self._solver_type = solver_type

        self._plot_best_scores = []
        self._plot_avg_scores = []
        self._plot_mutation_chances = []
        self._max_generations_count = max_generations_count

    def _generate_random_solution(self) -> List[List[int]]:
        """
        Generates a single random solution
        The generated solution isn't necessarily valid, but will make sure every row is valid permutation
        """
        solution = []
        for i in range(self._matrix_size):
            permutation = list(range(1, self._matrix_size + 1))
            random.shuffle(permutation)
            solution.append(permutation)

        return solution

    @abstractmethod
    @lru_cache
    def _calc_fitness(self, solution: Tuple[Tuple[int]]) -> float:
        """
        Abstract fitness calculation function. The more the result is high, the better the solution is
        """
        pass

    def calc_fitness(self, solution: List[List[int]]) -> float:
        """
        Fitness calculation function wrapper, to allow LRU cache for fitness calculation (for better performance)
        """
        solution_tuple = tuple(map(tuple, solution))
        return self._calc_fitness(solution_tuple)

    @property
    @abstractmethod
    def max_fitness(self):
        """
        Abstract max fitness property. This determines the highest fitness possible (of a valid solution)
        :return:
        """
        pass

    def is_valid_solution(self, solution: List[List[int]]):
        """
        Determines whether the given solution is valid
        """
        return self.calc_fitness(solution) == self.max_fitness

    @abstractmethod
    def _step_generation(self, replication_ratio, mutation_chance, elitism_ratio) -> List[List[List[int]]]:
        """
        Abstract function to implement a single generation step.
        The implementation should return a list of the solutions for the next generation, produced by replication,crossover,mutation etc.
        :param replication_ratio: Ratio for how many solutions will be replicated as is. The complementary ratio will be used for crossover
        :param mutation_chance: Chance for applying a mutation on solutions
        :param elitism_ratio: Ratio for how many times the best solution will be replicated from the previous to next generation
        """
        pass

    def print_solution(self, solution: List[List[int]]):
        """
        Helper function to print a given solution with its bigger-than signs
        """
        result_matrix = [[" " for _ in range(self._matrix_size * 2 - 1)] for _ in range(self._matrix_size * 2 - 1)]

        for i in range(0, self._matrix_size * 2, 2):
            for j in range(0, self._matrix_size * 2, 2):
                result_matrix[i][j] = str(solution[i // 2][j // 2])

        for greater_than_sign in self._greater_than_signs:
            side_greater, side_less = greater_than_sign
            side_greater *= 2
            side_less *= 2
            middle = Position((side_greater.row + side_less.row) // 2, (side_greater.column + side_less.column) // 2)

            if side_greater.row > side_less.row:
                arrow = '^'
            elif side_greater.row < side_less.row:
                arrow = 'V'
            elif side_greater.column > side_less.column:
                arrow = '<'
            elif side_greater.column < side_less.column:
                arrow = '>'
            else:
                raise

            result_matrix[middle.row][middle.column] = arrow

        pprint.pprint(result_matrix)

    @abstractmethod
    def should_stop(self):
        """
        Abstract function to determine whether the genetic algorithm can stop
        """
        pass

    def _run_algorithm(self, steps_count, replication_ratio, mutation_chance, elitism_ratio):
        """
        Runs the whole genetic algorithm with the given parameters.
        This runs _step_generation function repeatedly until should_stop returns True.
        If the best solution fitness value didn't change for 200 generations, it forces mutation on every solution for the next generation
        If the best solution fitness value didn't change for 400 generations after that, it drops the current solutions and starts over
        These conditions handle the case of early convergence, when an improvement in solutions isn't found
        """
        last_best_score_change = 0
        orig_mutation_chance = mutation_chance
        overall_best_solution = []
        overall_best_score = -1

        solved = False

        for _ in range(steps_count):
            if self.should_stop():
                solved = True
                break

            self._solutions = self._step_generation(replication_ratio, mutation_chance,
                                                    elitism_ratio)
            self._generation_count += 1

            scores = [self.calc_fitness(s) for s in self._solutions]
            best_score_index = np.argmax(scores)
            best_score = scores[best_score_index]
            avg_score = np.average(scores)
            best_solution = self._solutions[best_score_index]
            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_solution = best_solution

            if self._generation_count % 200 == 0:
                print("Currently best solution:")
                self.print_solution(overall_best_solution)
                print("============================================================")

            self._plot_best_scores.append(best_score)
            self._plot_avg_scores.append(avg_score)
            self._plot_mutation_chances.append(mutation_chance)

            best_score_groups = itertools.groupby(self._plot_best_scores)
            best_score_groups = [(label, sum(1 for _ in group)) for label, group in best_score_groups]

            current_best_score_generations = best_score_groups[-1][1]

            if current_best_score_generations == 200:
                mutation_chance = 1
            elif current_best_score_generations > 600 and self._generation_count - last_best_score_change > 600:
                self._solutions = [self._generate_random_solution() for _ in
                                     range(len(self._solutions))]
            else:
                mutation_chance = orig_mutation_chance

        if solved:
            print(f"Done! Best solution was:")
            self.print_solution(max(self._solutions, key=lambda solution: self.calc_fitness(solution)))

        return solved
        #plt.savefig(f"{self._puzzle_name}_{self._solver_type.name}_{self._generation_count}gens.jpg")

    def run_simulation(self, algo_iterations_per_render, replication_ratio, mutation_chance, elitism_ratio):
        """
        Runs the whole simulation, including live plotting.
        The genetic algorithm runs asynchronously so it won't be slowed down by plotting
        """

        # Init chart view
        fig, ax_plot = plt.subplots()
        ax_plot.set_xticks([])
        ax_plot.set_yticks([])

        ax_plot.set_xlabel("Generations")
        ax_plot.set_ylabel("Fitness score")

        best_line, = ax_plot.plot([], [], label="Best score")
        avg_line, = ax_plot.plot([], [], label="Average score")
        mutation_line, = ax_plot.plot([], [], label="Mutation chance")

        plt.suptitle(f"{self._puzzle_name} ({self._solver_type.name})")

        plt.legend(loc="lower right")

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        status_text = ax_plot.text(0.62, 0.5, "", transform=ax_plot.transAxes, fontsize=10,
                                       verticalalignment='top', bbox=props)

        def update(frame):
            generations_count = min(len(self._plot_best_scores), len(self._plot_avg_scores), len(self._plot_mutation_chances))
            plt.title(f"Generation {self._generation_count}")

            done = self._run_algorithm(algo_iterations_per_render, replication_ratio, mutation_chance, elitism_ratio)
            if done:
                ani.pause()

            best_line.set_data(np.arange(generations_count), np.array(self._plot_best_scores[:generations_count]))
            avg_line.set_data(np.arange(generations_count), np.array(self._plot_avg_scores[:generations_count]))
            mutation_line.set_data(np.arange(generations_count), np.array(self._plot_mutation_chances[:generations_count]))

            status_text.set_text(f"Best Score: {self._plot_best_scores[-1]:.2f}\n"
                                 f"Best Score (Overall): {max(self._plot_best_scores):.2f}\n"
                                 f"Average Score: {self._plot_avg_scores[-1]:.2f}\n"
                                 f"Mutation Chance: {self._plot_mutation_chances[-1]*100}%")

            ax_plot.set_xlim(0, generations_count)
            ax_plot.set_ylim(0, self.max_fitness + 0.1)
            ax_plot.xaxis.set_major_locator(AutoLocator())
            ax_plot.yaxis.set_major_locator(AutoLocator())
            ax_plot.axhline(y=self.max_fitness, color='r', linestyle='-', linewidth=0.5)

            return ax_plot, best_line, avg_line

        ani = animation.FuncAnimation(fig, update, interval=1)
        plt.show()


class FutoshikiSolverImpl(FutoshikiSolverBase):

    @lru_cache(maxsize=256*1024)
    def count_digits_occurrences(self, solution: Tuple[Tuple[int]]):
        """
        Counts digits occurrences in the given solution rows/columns
        A perfect permutation board (where both all rows and all columns are permutations) will result in all-1s matrices
        """
        row_counts_mat = []
        col_counts_mat = []

        for i in range(self._matrix_size):
            row_counts = [0] * self._matrix_size
            col_counts = [0] * self._matrix_size
            for j in range(self._matrix_size):
                row_counts[solution[i][j] - 1] += 1
                col_counts[solution[j][i] - 1] += 1

            row_counts_mat.append(row_counts)
            col_counts_mat.append(col_counts)

        return row_counts_mat, col_counts_mat

    def calc_permutation_score(self, solution: Tuple[Tuple[int]]):
        """
        Calculates the permutations score.
        For every row/column, if a digit is present only once, the score is raised 1
        Therefore, for correct board (permutation-wise), the maximum score is row_count * col_count * 2
        Eventually we normalize the score, so it'll have the same weight as other score factors.
        """
        success = 0
        row_counts_mat, col_counts_mat = self.count_digits_occurrences(solution)
        for i in range(self._matrix_size):
            for index in range(self._matrix_size):
                if row_counts_mat[i][index] == 1:
                    success += 1
                if col_counts_mat[i][index] == 1:
                    success += 1
        return success / (self._matrix_size ** 2 * 2)

    def calc_greater_than_score(self, solution: Tuple[Tuple[int]]):
        """
        Calculates the greater than signs score.
        For every greater than sign, if the digits don't contradict it, the score is raised by 1
        Therefore, for correct board (greater-than-wise), the maximum score is len(greater_than_signs)
        Eventually we normalize the score, so it'll have the same weight as other score factors.
        """
        success = 0
        for greater_than_sign in self._greater_than_signs:
            greater, less = greater_than_sign
            if solution[greater.row][greater.column] > solution[less.row][less.column]:
                success += 1

        return success / len(self._greater_than_signs)

    @lru_cache(maxsize=2048)
    def _calc_fitness(self, solution) -> float:
        """
        Calculates the fitness, summing permutation score and greater-than signs score
        Note: Pre-defined digits are always correct, because we handle it in _step_generation
        :param solution:
        :return:
        """
        fitness = 0
        fitness += self.calc_permutation_score(solution)
        fitness += self.calc_greater_than_score(solution)

        return fitness

    @property
    def max_fitness(self):
        """
        The maximum score a solution can get (meaning it is the correct one)
        This is increased by 1 for each score factor, so it's total of 2
        """
        return 2

    def _get_crossovers(self, solutions, probabilities, count):
        """
        Returns a solution population by crossovers of given solutions, using biased selection
        Crossovers are made row-wise, by randomizing a row index, i, and constructing the result by taking the first i rows from the first solution,
        and concatenating them to the last _matrix_size - i rows in the second solution.
        This way row permutations are maintained, and there's a chance column permutations can be improved
        """
        ret = []
        for i in range(count):
            sol1_index = np.random.choice(range(len(solutions)), p=probabilities)
            sol2_index = np.random.choice(range(len(solutions)), p=probabilities)

            sol1 = solutions[sol1_index]
            sol2 = solutions[sol2_index]

            index = random.randint(0, self._matrix_size - 1)
            crossover = sol1[:index] + sol2[index:]
            ret.append(crossover)

        return ret

    def _get_replications(self, solutions, probabilities, count):
        """
        Returns a solution population by replications of given solutions, using biased selection
        """
        ret = []
        for i in range(count):
            sol_index = np.random.choice(range(len(solutions)), p=probabilities)
            ret.append(solutions[sol_index])

        return ret

    def _mutate_solutions(self, solutions, mutation_chance):
        """
        Mutates the given solutions (in-place)
        Every solution will be mutated with probability of mutation_chance.
        Solution mutation is made selecting 1 to 5 rows (randomly),
        for each row, exchanging 2 values in it.
        This way, row permutations are maintained, while giving a chance for columns permutations to be fixed.
        However, greater-than signs and predefined digits can be harmed. Solution with harmed signs will naturally get lower score,
        and predefined digits will be fixed anyway in the end of _step_generation
        """
        for i in range(len(solutions)):
            if random.random() > mutation_chance:
                continue

            for _ in range(random.randint(1, 5)):
                options = list(range(self._matrix_size))
                index1 = random.choice(options)
                options.remove(index1)
                index2 = random.choice(options)

                orientation_index = random.choice(list(range(self._matrix_size)))

                mutated1 = Position(orientation_index, index1)
                mutated2 = Position(orientation_index, index2)
                solution = deepcopy(solutions[i])

                solution[mutated1.row][mutated1.column], solution[mutated2.row][mutated2.column] = \
                solution[mutated2.row][mutated2.column], solution[mutated1.row][mutated1.column]
                solutions[i] = solution

    def _fix_value_in_row_permutation(self, row, value, dst_index):
        """
        Helper function for _optimize_solutions.
        Makes one fix step of a column permutation, by exchanging two values in the needed row.
        This makes sure row permutation is maintained, while making a fix step for column permutation
        """
        src_index = -1
        for index, current_value in enumerate(row):
            if current_value == value:
                src_index = index
                break
        row[src_index], row[dst_index] = row[dst_index], row[src_index]

    def _optimize_solutions(self, solutions):
        """
        Makes one optimization step of every one of the given solution. Used in Darwin/Lamark mode.
        The optimization looks for an invalid column permutation (if there is one).
        Then, it counts the occurrences of digits in the solution rows/columns.
        When a column is not a permutation, it must mean there is at least one digit with 0 occurrences, and at least one digit with at least 2 occurences.
        Therefore, it replaces one of the digits with (at least) 2 occurrences with the digit with 0 occurrences.
        To not harm row permutations, it uses _fix_value_in_row_permutation helper function in order to do that.
        """
        solutions = deepcopy(solutions)
        for solution in solutions:
            solution_tuple = tuple(map(tuple, solution))
            row_counts_mat, col_counts_mat = self.count_digits_occurrences(solution_tuple)
            for column_index, column in enumerate(col_counts_mat):
                missing_value = -1
                overflow_value = -1
                for index, occurrence in enumerate(column):
                    if occurrence >= 2:
                        overflow_value = index + 1
                        continue
                    if occurrence == 0:
                        missing_value = index + 1
                        continue
                    if missing_value != -1 and overflow_value != -1:
                        break
                if missing_value == -1 and overflow_value == -1:
                    continue

                for row in range(self._matrix_size):
                    if solution[row][column_index] == overflow_value:
                        self._fix_value_in_row_permutation(solution[row], missing_value, column_index)
                        break

        return solutions

    def _step_generation(self, replication_ratio, mutation_chance, elitism_ratio):
        """
        Makes a single generation step, calling replication, crossover, mutation, etc. functions
        After creating the population for the new generation, it fixes all predefined digits violations.
        It does that to decrease the solution search space, and doesn't harm row permutations.
        This is because it exchanges digits in the same row for the fix, maintaining row permutations
        """
        # Don't optimize the best solution, so elitism will be maintained
        scores = [self.calc_fitness(s) for s in self._solutions]
        best_score_index = np.argmax(scores)
        best_solution = self._solutions[best_score_index]

        if self._solver_type in [SolverType.DARWIN, SolverType.LAMARK]:
            for i in range(self._matrix_size):
                optimized_solutions = self._optimize_solutions(self._solutions)

            scores = [self.calc_fitness(s) for s in optimized_solutions]

            if self._solver_type == SolverType.LAMARK:
                self._solutions = optimized_solutions

        total_score = sum(scores)

        probabilities = [s/total_score for s in scores]

        new_solutions = []
        # Replication
        replication_count = math.ceil(replication_ratio * len(self._solutions))
        elitism_count = math.ceil(len(self._solutions) * elitism_ratio)
        crossover_count = len(self._solutions) - replication_count - elitism_count

        new_solutions += self._get_replications(self._solutions, probabilities, replication_count)

        # Crossover
        new_solutions += self._get_crossovers(self._solutions, probabilities, crossover_count)

        # Mutation
        self._mutate_solutions(new_solutions, mutation_chance)

        # Add the best solution from the generation start
        new_solutions += [deepcopy(best_solution)] * elitism_count

        # Force predefined digits on the solutions
        for new_solution in new_solutions:
            for predefined in self._predefined_digits:
                value_at_predefined_location = new_solution[predefined.position.row][predefined.position.column]
                if value_at_predefined_location != predefined.value:
                    other_index = new_solution[predefined.position.row].index(predefined.value)
                    new_solution[predefined.position.row][predefined.position.column], new_solution[predefined.position.row][other_index] = new_solution[predefined.position.row][other_index], new_solution[predefined.position.row][predefined.position.column]

        return new_solutions

    def should_stop(self):
        """
        Determines whether the genetic algorithm should stop.
        It stops if a valid solution is found, or the generation count threshold was reached
        """
        return any(filter(self.is_valid_solution, self._solutions)) or self._generation_count >= self._max_generations_count


def main():
    """
    Input is: <PATH_TO_BOARD_FILE> <SOLVER_TYPE> <MAX_GENERATIONS_COUNT>
    """
    input_board_txt = sys.argv[1] if len(sys.argv) >= 2 else input("Please enter input board path: ")
    with open(input_board_txt) as f:
        lines = f.readlines()

    solver_type = sys.argv[2] if len(sys.argv) >= 3 else input("Please enter solver type id:\n1: Regular\n2: Darwin\n3: Lamark\n")
    solver_type = SolverType(int(solver_type))

    max_generations_count = sys.argv[3] if len(sys.argv) >= 4 else input("Please enter max generations count: ")
    max_generations_count = int(max_generations_count)

    i = 0
    matrix_size = int(lines[i])
    i += 1

    predefined_digits_count = int(lines[i])
    i += 1
    predefined_digits_descs = lines[i: i + predefined_digits_count]
    i += predefined_digits_count

    greater_than_count = int(lines[i])
    i += 1
    greater_than_descs = lines[i: i + greater_than_count]

    predefined_digits = []
    for predefined_digit_desc in predefined_digits_descs:
        i, j, value = map(int, predefined_digit_desc.split())
        predefined_digits.append(FilledCell(value, Position(i -1, j - 1)))

    greater_than_signs = []
    for greater_than_desc in greater_than_descs:
        i1, j1, i2, j2 = map(int, greater_than_desc.split())
        greater_than_signs.append((Position(i1 - 1, j1 - 1), Position(i2 - 1, j2 - 1)))

    solver = FutoshikiSolverImpl(input_board_txt.split('.')[0], matrix_size, predefined_digits, greater_than_signs, 100, solver_type, max_generations_count)
    algo_iterations_per_render = 10 if solver_type == SolverType.REGULAR else 5
    solver.run_simulation(algo_iterations_per_render, replication_ratio=0.1, mutation_chance=0.05, elitism_ratio=0.05)


if __name__ == '__main__':
    main()
