# Nonlinear solving process

Generate a markdown table:

| Method | Total Number |
|--------|:----------:|
| number of function evaluations | 5 |
| number of jacobian evaluations | 4 |
| number of factorizations | 0 |
| number of linear solves | 3 |
| number of steps taken | 3 |

With sparsity pattern from ADTypes.jacobian_sparsity, we get the following performance stats:

| Method | Total Number |
|--------|:----------:|
| number of function evaluations | 5 |
| number of jacobian evaluations | 3 |
| number of factorizations | 0 |
| number of linear solves | 3 |
| number of steps taken | 3 |

The benefit of using the sparsity pattern is that it reduces the number of jacobian evaluations, which can be expensive for large systems, so when handling with large system, the benefit becomes more significant.