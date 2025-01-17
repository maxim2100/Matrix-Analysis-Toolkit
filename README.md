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

## How to Use

1. **Start the Program**: Run the main script to start the program. You will be presented with a menu of options.
2. **Menu Options**:
   - `1. Enter a new matrix`: Allows you to input a new matrix by specifying its elements row by row.
   - `2. Display saved matrix`: Displays a previously saved matrix.
   - `3. Find inverse of matrix`: Calculates the inverse of a specified matrix, if it exists.
   - `4. Calculate norm of matrix`: Computes the infinity norm of a given matrix.
   - `5. Calculate norm of inverse matrix`: Calculates the infinity norm of the inverse of a specified matrix.
   - `6. Calculate condition number of matrix`: Computes the condition number of a given matrix.
   - `7. Print all matrices`: Displays all matrices stored in the session.
   - `8. Exit`: Exits the program.
3. **Input Data**: Follow the prompts to enter data as required by the chosen operation.

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
