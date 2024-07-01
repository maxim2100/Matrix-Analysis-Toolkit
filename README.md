# Matrix Analysis Toolkit

## Overview

Matrix Analysis Toolkit is a Python-based tool that allows users to perform essential matrix operations such as calculating the inverse of a matrix using elementary row operations, computing matrix norms, and determining the condition number of a matrix.

## Features

- Matrix multiplication of a 3x3
- Inverse of a 3x3 matrix using elementary row operations
- Norm calculation of matrices
- Vector norm calculation
- Condition number computation
- Save and load matrix data and calculations

## Basic Assumptions

It is important to note that in this project, I took the basic assumption that a matrix is only square and only of 3x3 and a vector is 3x1.

## Notations

- **A**: The original 3x3 matrix.
- **A^-1**: The inverse of matrix A.
- **||A||**: The norm of matrix A.
- **||A^-1||**: The norm of the inverse matrix A^-1.
- **cond(A)**: The condition number of matrix A, calculated as ||A|| \* ||A^-1||.

## Usage

1. Clone the repository or download the project files.
2. Ensure you have Python installed on your system.
3. Run the `matrix_analysis_toolkit.py` script.

## Menu Options

1. Enter a new matrix: Input a new 3x3 matrix.
2. Display current matrix: Display the current matrix.
3. Find inverse of current matrix: Calculate and display the inverse of the current matrix.
4. Calculate norm of current matrix: Calculate and display the norm of the current matrix.
5. Calculate norm of inverse matrix: Calculate and display the norm of the inverse matrix.
6. Calculate condition number of current matrix: Calculate and display the condition number of the current matrix.
7. Save current matrix and calculations: Save the current matrix and calculations to a file.
8. Load saved matrix and calculations: Load a previously saved matrix and calculations from a file.
9. Exit: Exit the program.

### Example

```python
# Run the script and follow the prompts to enter a 3x3 matrix.
python matrix_analysis_toolkit.py
```

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
