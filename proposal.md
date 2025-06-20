# Neuromancer: Some ideas for future development

## New features

1. Helpers for convenience:
   - A means of splitting (specified sizes or fractions) a dataset between `train`, `dev`, and `test` within a single container (perhaps combining three instances of `DataSource` from [./datasources.py](./datasources.py))
   - Checks for data `shape` and `dtype` consistency in `nm.trainer.Trainer` <!-- I'm getting a "RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float" 12 stack frames deep -->
2. Modeling support
   - Explore support for symbolic variables with structured types (e.g., complex numbers, quaternions)
   - More ergonomic expressions on array-valued variables: e.g., `x[:, [0]]**2 + x[:, [1]]**2` as `nm.sum(x**2)`
   - Time derivatives as first-class modeling objects
   - Automatic reformulation of high-order derivatives with auxiliary variables
   - Explore options for structural analysis of differential-algebraic systems
3. Additional diagnostics from `Trainer.train`, e.g.
   - Diagnostic extraction via a custom callback
   - A default extraction utility to plot losses while training
   - Integration with a progress bar like [`tqdm`](https://tqdm.github.io/)
4. Performance/accessibility:
   - Explore pathways for backend flexibility to support multiple AD frameworks: JAX, [Legate](https://developer.nvidia.com/legate)
5. Scalable QP layers using ADMM with gradients computed via implicit differentiation of the fixed-point map rather than the full KKT system [[Butler & Kwon (2023)]](https://doi.org/10.1007/s10589-022-00422-7)
6. Explore support for logical/boolean variables 
   - Application to learning to optimize unit commitment
7. Support for delay differential equations
8. Support for external libraries of benchmark problems, e.g.
   - Learning to optimize: 
     - [Open-L2O](https://github.com/VITA-Group/Open-L2O) from [[Chen et al. (2022)]](https://www.jmlr.org/papers/volume23/21-0308/21-0308.pdf)
     - [MATILDA](https://matilda.unimelb.edu.au/matilda/)
   - Learning to model: 
     - [Kroll & Schulte (2014)](https://doi.org/10.1016/j.asoc.2014.08.034)
     - [MOR Wiki](https://modelreduction.org/morwiki/Main_Page)
     - [DaISy](https://homes.esat.kuleuven.be/~smc/daisy/)
   - Learning to control: 
     - [Bridge](https://zenodo.org/records/10966777)
     - [Open RL Benchmark](https://github.com/openrlbenchmark/openrlbenchmark)
     - [RLLAB](https://github.com/rll/rllab) from [[Duan et al. (2016)]](https://proceedings.mlr.press/v48/duan16.pdf)
     - [Koryakovskiy et al. (2017)](https://doi.org/10.1016/j.robot.2017.02.006)

## New examples

1. Tutorial examples:
   - Linear regression, quantile regression
   - Linear system identification
   - Displaying diagnostic information with `torch.utils.tensorboard`
2. Application-specific parametrizations:
   - Convex optimization layers via [cvxpylayers.torch](https://locuslab.github.io/2019-10-28-cvxpylayers/) (cf. the 2D example in [pao.ipynb](./pao.ipynb))
   - Feature-extraction layers via [`kymatio.torch`](https://www.kymat.io/)
3. Learning to optimize:
   - Learning the efficient frontier of a multi-objective optimization problem
   - Comparison of learned vs. Hessian-informed proximal metrics [[King & Drgoňa et al. (2024)]](https://doi.org/10.48550/arXiv.2404.00882)
4. Energy-related case studies:
   - Learning to optimize dynamics-aware economic dispatch [[King & Drgoňa et al. (2022)]](https://doi.org/10.23919/ACC53348.2022.9867379)

## Extra documentation

1. FAQ entries:
   - _What are the roles of  `Node` vs `Variable`?_
   - _What are the roles of `Problem` vs `Loss` vs `System`?_
   - _How do I specify the penalty norm on a `Constraint`?_
   - _How do I specify penalty weights?_
   - _How do I specify multiple objectives?_
   - _How do I extract diagnostics from `Trainer.train`?_
   - _Which dynamic system models are already available?_ (See `neuromancer.dynamics.ode.*`)
