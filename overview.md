Neuromancer: An introduction
============================

# Design overview

Neuromancer is a **PyTorch-based** framework for solving **parametric constrained optimization problems** (PCOPs).
The two emphasized terms are explained below.

## PyTorch in a nutshell

PyTorch is a widely-adopted platform for parametric unconstrained optimization via gradient descent.
It offers:
- High performance on modern GPU accelerators
- A healthy software ecosystem, including `torch.utils.tensorboard`, `cvxpylayers.torch`, and others
- A large community of developers, active on Stack Overflow
- Extensive documentation
- Portability across the major operating systems

Neuromancer inherits these strengths and builds on PyTorch's key abstractions:

| Component                     | Description                                                                                                          |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `torch.nn.Module`             | Stateful representations of trainable functions and their compositions; can be evaluated on a CPU or GPU/accelerator |
| `torch.Tensor`                | Holds numeric inputs/outputs, analogous to `numpy.ndarray`                                                           |
| `torch.autograd.*`            | Functions for propagating tangents/gradients through modules                                                         |
| `torch.optim.Optimizer`       | Gradient descent algorithms and their parameters                                                                     |
| `torch.utils.data.Dataset`    | A `dict`-like container mapping keys/identifiers to data samples, possibly with parallel I/O                         |
| `torch.utils.data.DataLoader` | For extracting data from `Dataset` with `batch_size`, `shuffle`, etc.                                                |

## Parametric constrained optimization in Neuromancer

By adding a layer of capabilities for **constrained** parametric optimization, Neuromancer greatly expands the range of problems that are easy to express in PyTorch:

- Support for **expressing constraints** directly via symbolic expressions (rather than parent-child object composition)
- Support for **aggregating constraints** and objectives into PyTorch-compatible loss functions
- A powerful mechanism for **composing** trainable functions and **injecting data** via keys/identifiers
- An implementation of **projected** gradient descent for constrained optimization, complementing `torch.optim.Optimizer`
- A catalogue of approximation modules for scientific machine learning, compatible with `torch.nn.Module`
- Customizable template for formulating a range of scientific machine learning problems, e.g. `neuromancer.SINDy` for System Identification
- A catalogue of standard dynamic systems (e.g. `VanDerPolControl`) for use in tutorials/demonstrations and benchmarking
- An expanding set of supporting materials, including research papers, tutorials, and presentations

Neuromancer's capabilities are embodied in the following key components:

| Component                        | Description                                                                                                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `neuromancer.Variable`           | Stateless **symbolic** variables and expressions for composing objective and constraint functions                                                                              |
| `neuromancer.Constraint`         | For specification of constraints (and algebraic penalty aggregation)                                                                                                           |
| `neuromancer.Node`               | Endows the ports of a PyTorch-compatible approximation module with **keys/identifiers**, facilitating arbitrarily complex **compositions** of elementary approximation modules |
| `neuromancer.AggregateLoss`      | A penalty-reformulation of a (non-parametric) constrained optimization problem                                                                                                 |
| `neuromancer.Problem`            | A **parametric** optimization problem, pairing an `AggregateLoss` with an approximation module                                                                                 |
| `neuromancer.GradientProjection` | An implementation of gradient descent suitable for **constrained** optimization                                                                                                |

The following mnemonics might be helpful to new users:

> _expression_ = composition of `Variable`s
>
> _approximation architecture (AA)_ = composition of `Node`s
>
> _constrained optimization problem (COP)_ = (objective _expressions_) + (constraint _expressions_)
>
> _parametric constrained optimization problem (PCOP)_ = _AA_ + _COP_ + (sampled data source)

---

# Parametric constrained optimization problems

A PCOP has the following generic form:
$$
\left.
\begin{aligned}
    \min_{q, \theta} \sum_{k} \begin{bmatrix} f_1(x_k, p_k, q) \\ \vdots \\ f_M(x_k, p_k, q) \end{bmatrix}
    \\
    \left.\begin{array}{r}
        g(x_k, p_k, q) \leq 0
        \\
        h(x_k, p_k, q) = 0
        \\
        x_k = \pi(p_k; \theta)
    \end{array}\right\} &  \forall k
    \\
    \text{given} ~ D = \{(p_k, x_k)\}_{k=1}^N &
\end{aligned}
\right\} \quad \text{(PCOP)}
$$

Neuromancer reformulates PCOPs as unconstrained problems by converting hard constraints into differentiable penalty terms:
$$
\begin{aligned}
    \min_{q, \theta} & ~
    \left\{ 
        F(q, \theta):
        x_k = \pi(p_k; \theta) ~ \forall k 
    \right\}
    \\
    F(q, \theta) {}& = \sum_{k} 
        \lambda^\top f(x_k, p_k, q) 
        + \omega_g \operatorname{mean}(\operatorname{relu}(g(x_k, p_k, q))^{\rho_g})
        + \omega_h \operatorname{mean}(\operatorname{abs}(h(x_k, p_k, q))^{\rho_h})
\end{aligned}
$$

The symbols are described as follows:

| Symbol                      | Meaning                     | Class                                       |
| --------------------------- | --------------------------- | ------------------------------------------- |
| $k$                         | scenario/sample index       |                                             |
| $x_k$                       | decision variables          | `neuromancer.Variable`                      |
| $p_k$                       | parameters                  | `neuromancer.Variable`                      |
| $q$                         | auxiliary variables         | `neuromancer.Variable`                      |
| $\pi$                       | approximation architecture  | `List[neuromancer.Node]` representing a DAG |
| $\theta$                    | degrees of freedom of $\pi$ | `neuromancer.Variable`                      |
| $f$                         | objective/cost functions    | `neuromancer.Variable`                      |
| $g$, $h$                    | constraint functions        | `neuromancer.Variable`                      |
| $\lambda$                   | scalarization coefficients  |                                             |
| $\omega$                    | penalty coefficients        |                                             |
| $\rho_\bullet \in \{1, 2\}$ | penalty powers              |                                             |

The roles of these variables vary by use case. For example, $\pi$ may represent a dynamics model or a control policy.

PCOPs offer a more expressive formulation than standard PyTorch pipelines, which typically lack constraints and auxiliary variables: e.g.
$$
\begin{aligned}
    \min_{\theta} \sum_{k} \Vert \pi(p_k; \theta) - x_k \Vert^2
    \quad
    \text{given} ~ D = \{(p_k, x_k)\}_{k=1}^N &
\end{aligned}
$$

This added expressiveness enables direct modeling of a broad range of practical applications. 
We will consider serveral examples in the sections that follow:
1. Learning to optimize (L2O)
2. Learning to model (sytem identification)
3. Learning to control
   - Imitation learning
   - Differential model-predictive control
4. Decision-focused learning

## Learning to optimize (L2O)

The goal of learning to optimize (L2O) is to move the computation of a parametric optimization problem's solution map **offline**, enabling fast evaluation online:
$$
\begin{aligned}
    \hat{x} = \pi(p; \theta) \approx \text{Solution of PCOP} 
\end{aligned}
$$

The approximation can be used directly or as a warm-start solution for online iterations.

## Learning to model (system identification)

The goal in this case is to estimate the state transition function of a dynamic system:
$$
\begin{aligned}
    \arg\min_{x,\theta} ~ 0
    \\
    \left.\begin{array}{r}
        y_{t,k} - f(x_{t,k}) = 0 
        \\
        h(x_{t,k}) \leq 0 
        \\
        x_{t+1,k} = \pi(x_{t,k}, u_{t,k}; \theta)
    \end{array}\right\} & \quad \forall t, k
    \\
    \text{given} ~ D = \{(u_k, y_k)\}_{k=1}^N 
\end{aligned}
$$

| Symbol              | Meaning                   | Analogy in POMP                         |
| ------------------- | ------------------------- | --------------------------------------- |
| $t$                 | discrete time index       |                                         |
| $k$                 | scenario/sample index     | for each entire input-output trajectory |
| $u_{t,k}$           | known input/control       | $p_k$/$x_k$                             |
| $y_{t,k}$           | measurement signal        | $p_k$/$x_k$                             |
| $x_{t,k}$           | unknown state             | $q$                                     |
| $f(x)$              | known measurement model   | $f(x, p)$                               |
| $h(x)$              | encodes state bounds      | $h(x, p)$                               |
| $\pi(x, u; \theta)$ | state transition function | $\pi(x; \theta)$                        |

The trained dynamic model $\pi(x, u; \theta)$ can be employed online in a model-predictive control loop, as described in the next section.

## Learning to control

### Imitation learning

In the simplest case, given state-action pairs $\{x_k, u_k\}_{k=1}^N$ sampled from an expert system, one can estimate a control law $\pi(x; \theta)$ by offline training:
$$
\begin{aligned}
    \arg\min_\theta ~ \sum_k \Vert \pi(x_k; \theta) - u_k\Vert^2
\end{aligned}
$$

| Symbol           | Meaning                    | Analogy in POMP            |
| ---------------- | -------------------------- | -------------------------- |
| $k$              | sample index               | for each state-action pair |
| $x_{k}$          | state sample               | $p_k$                      |
| $u_{k}$          | input/control sample       | $x_k$                      |
| $\pi(x; \theta)$ | controller policy function | $\pi(x; \theta)$           |

The trained policy $\pi(x; \theta)$ can then be deployed for online use.

### Data-driven model-predictive control

When expert demonstrations are unavailable, but a model of the plant dynamics is known or has been learned, a control law can be trained as follows:
$$
\begin{aligned}
    \arg\min_{\theta} ~ \sum_{k=1}^N \sum_{t=0}^M \omega_1 \Vert x_{t,k} \Vert^2 + \omega_2 \Vert u_{t,k} \Vert^2
    \\
    \left.\begin{array}{r}
        x_{t+1,k} - f(x_{t,k}, u_{t,k}) = 0
        \\
        h_1(x_{t,k}) \leq 0 
        \\
        h_2(u_{t,k}) \leq 0
        \\
        u_{t,k} = \pi(x_{t,k}; \theta)
    \end{array}\right\} & \quad \forall t, k
    \\
    \text{given} ~ D = \{(x_{0,k})\}_{k=1}^N 
\end{aligned}
$$

| Symbol           | Meaning                   | Analogy in POMP  |
| ---------------- | ------------------------- | ---------------- |
| $t$              | discrete time index       |                  |
| $u_t$            | input/control             | $x_k$            |
| $x_t$            | state                     | $q$              |
| $f(x, u)$        | state transition model    | $g(x, p)$        |
| $h_1(x)$         | state bounds              | $h(x, p)$        |
| $h_2(u)$         | control bounds            | `"`              |
| $\pi(x; \theta)$ | full state control policy | $\pi(x; \theta)$ |
| $\omega_\bullet$ | penalty coefficients      |

This formulation employs full state feedback for simplicity. A complete formulation is explained in [Drgoňa et al. (2022)](https://www.sciencedirect.com/science/article/pii/S0959152422000981?via%3Dihub).

## Decision-focused learning

Given features and costs $\{x_k, c_{k}\}_{k=1}^N$:
$$
\begin{aligned}
    \arg\min_\theta ~ \frac{1}{N} \sum_k c_k^\top (w^\ast(\hat{c}_k) - w^\ast(c_k))
    \\
     w^\ast(c) = \arg\min_w \{c^\top w: w \in W\}
    \\
    \hat{c}_k = \pi(x_k; \theta)
    \\
    \text{given} ~ D = \{(x_k, c_k)\}_{k=1}^N 
\end{aligned}
$$

| Symbol           | Meaning                           | Analogy in POMP  |
| ---------------- | --------------------------------- | ---------------- |
| $k$              | sample index                      |                  |
| $x_k$            | feature of prediction model       | $p_k$            |
| $c_k$            | cost (target of prediction model) | $x_k$            |
| $\pi(x; \theta)$ | prediction model                  | $\pi(x; \theta)$ |

The inner $\arg\min$ presents an interesting difficulty, which can be handled in Neuromancer via the `cvxpylayers.torch.CvxpyLayer` - part of the PyTorch ecosystem. A demonstration is available in the accompanying script, following the notation of [Tang & Khalil (2024)](https://doi.org/10.1007/s12532-024-00255-x).

---

These examples illustrate the broad applicability of the PCOP formulation to learning-based modeling, optimization, and control.

Neuromancer's symbolic expression of constraints and objectives, composable approximation modules, and other building blocks, provide a powerful foundation for research at the intersection of scientific computing, optimization, and machine learning.

Its seamless PyTorch integration and first-class support for PCOP modeling distinguish Neuromancer within the rapidly evolving scientific machine learning ecosystem.