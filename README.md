# rl-psr
Code for "Reconciling Rewards with Predictive State Representations", A. Baisero and C. Amato, IJCAI 2021.

## Instructions

### Required Packages

The following packages are required, and can be found on github:
* `rl-parsers <https://github.com/abaisero/rl-parsers>`_
* `gym-pomdps <https://github.com/abaisero/gym-pomdps>`_

Preferably, you should be creating an environment specifically to install the
required packages first, then this one, and then run the experiments.  For each
of the above, in that same order:

1. move to the package directory, and install using

   ```{python}
   python -m pip install .
   ```

2. run the tests to make sure installation was correct

   ```{python}
   python -m unittest discover
   ```

### Experiment Code

The code to replicate the experiments from the IJCAI 2021 paper is contained in
the experiments/ folder.  Move there, then run the following commands.  NOTE:
in these scripts, `bsr` stands for belief-state representation (i.e. POMDP),
`psr` stands for predictive-state representation, and `rpsr` stands for
reward-predictvie state representation.

File `pomdps.all.txt` contains all the POMDPs tested to verify the significance
of the accuracy problem of PSRs; while `pomdps.ijcai21.txt` contains the 6
domains whose PSRs are non-accurate, for which we could run value iteration,
and which form the basis for the more thorough evaluation.

1. Search for core sets of PSRs and R-PSRs:

   ```{python}
   <pomdps.ijcai21.txt ./search.local
   ```

   This will compute core sets, print their ranks, and store them in `cores/`.

2. Compute reward errors of PSRs and R-PSRs w.r.t. the POMDP rewards:

   ```{python}
   <pomdps.ijcai21.txt ./info.local
   ```

   This will compute error measures, print them to standard output, and store
   them in `infos/`.

3. Run POMDP-VI, PSR-VI and R-PSR-VI:

   ```{python}
   <pomdps.ijcai21.txt ./vi.local
   ```

   This will run the value iteration algorithms for 150 iterations, and store
   the resulting value functions in `vfs/`.  This is the slowest step;  it will
   take many hours if a single machine is used.

4. Plot a quasi-Bellman-error measure to check for convergence of the value
   functions:

   ```{python}
   <pomdps.ijcai21.txt ./plot.local
   ```

   This will plot the convergence properties of the value functions, and store
   them in `plots/`.

5. Evaluate the value functions' respective policies:

   ```{python}
   <pomdps.ijcai21.txt ./eval.local
   ```

   This will run the Random, POMDP-VI, PSR-VI and R-PSR-VI policies for 100
   episodes of 1000 steps each, calculate the true and estimated returns, and
   store them in `evals/`.

6. Compile the evaluation results into tables:

   ```{python}
   <pomdps.ijcai21.txt ./tables | tee tables.tex
   ```

   This will aggregate the results obtained by the evaluation step, print the
   results in a tex/table format, and save the results in `tables.tex`.
