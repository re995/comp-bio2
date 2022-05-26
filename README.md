## Computational Biology Exercise 2

### For quick running, you can use the following template:

```ex2.exe BOARD.txt 1 30000``` (for regular mode)

```ex2.exe BOARD.txt 2 30000``` (for Darwin mode)

```ex2.exe BOARD.txt 3 30000``` (for Lamark mode)

Or you can run ```ex2.exe``` without arguments, to be asked for parameters interactively


### In order to run the project, you can run ex2.exe with the following arguments:
* Path to board text file (with the given format)
* Genetic algorithm type:
   * 1 for Regular
   * 2 for Darwin (Make local optimizations to solutions and use optimization results for biased selection, but take original solutions for next generation. Equivalent to a person working out in the gym, won't pass their acquired qualities to the next generation)
   * 3 for Lamark (Make local optimizations to solutions and use optimization result for both biased selection, and for the next generation)
* How many generations to run the simulation for (if solution is found, it is printed immediately and the simulation stops).

The code prints the currently best board from time to time, and eventually (when a valid solution is found or generation count exceeds) it prints the best solution


