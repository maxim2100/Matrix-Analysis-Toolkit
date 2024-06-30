# Matrix Analysis Toolkit

## Overview

Matrix Analysis Toolkit is a Python-based tool that allows users to perform essential matrix operations such as calculating the inverse of a matrix using elementary row operations, computing matrix norms, and determining the condition number of a matrix.

## Features

- Inverse of a 3x3 matrix using elementary row operations
- Norm calculation of matrices
- Condition number computation

## Usage

1. Clone the repository or download the project files.
2. Ensure you have Python installed on your system.
3. Run the `matrix_analysis_toolkit.py` script.

### Example

```python
# Run the script and follow the prompts to enter a 3x3 matrix.
python matrix_analysis_toolkit.py

Sample Input
```

    Enter the elements of the 3x3 matrix row-wise:
    Enter row 1: 1 2 3
    Enter row 2: 0 1 4
    Enter row 3: 5 6 0

```
Sample Output
```

Matrix A:
1.0000 2.0000 3.0000
0.0000 1.0000 4.0000
5.0000 6.0000 0.0000
Inverse of A:
-24.0000 18.0000 5.0000
20.0000 -15.0000 -4.0000
-5.0000 4.0000 1.0000
Norm of A:
9.5394
Norm of A^-1:
49.6991
Condition number of A:
473.6488

```

```
