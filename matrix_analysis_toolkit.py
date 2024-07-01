import json


UNIT_MATRIX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
VECTOR_ZIRO = [0, 0, 0]
MATRIX_ZIRO = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]



def vector_1_norm(vector):
    """Calculate the 1-norm of a 3x1 vector."""
    return sum(abs(v) for v in vector)

def vector_inf_norm(vector):
    """Calculate the infinity-norm of a 3x1 vector."""
    return max(abs(v) for v in vector)

def matrix_1_norm(matrix):
    """Calculate the 1-norm of a 3x3 matrix."""
    return max(sum(abs(matrix[i][j]) for i in range(3)) for j in range(3))

def matrix_inf_norm(matrix):
    """Calculate the infinity-norm of a 3x3 matrix."""
    return max(sum(abs(matrix[i][j]) for j in range(3)) for i in range(3))

def condition_number(matrix):
    """Calculate the condition number of the matrix."""
    matrix_inv = find_inverse(matrix)
    norm_A = matrix_inf_norm(matrix)
    norm_A_inv = matrix_inf_norm(matrix_inv)
    cond_A = norm_A * norm_A_inv
    return cond_A

def vector_multiplication(a, b):
    """Multiply two vectors(a 1x3 * b 3x1) return the result as number."""
    if len(a) != 3 or len(b) != 3:
        raise ValueError("Both vectors must be of length 3.")
    for i in range(3):
        result += [a[i] * b[i]]
    return result

def matrix_and_vector_multiplication(A, x):
    """Multiply matrix A 3x3 by vector x 3x1."""
    #aij - The member in the i-th row and the j-th column

    result = [0, 0, 0]
    result1 = [0, 0, 0]
    for i in range(3):
        result1[i] = vector_multiplication(A[i], x)

    for i in range(3):
        for j in range(3):
            result[i] += A[i][j] * x[j]

    if result==result1:
        print(f"The result is {result}")

    return result


def matrix_multiplication(A, B):
    """Multiply two 3x3 matrices A and B."""
    # The member in the i-th row and the j-th column
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += A[i][k] * B[k][j]
    return result

def find_inverse(matrix):
    """Find the inverse of a 3x3 matrix using elementary row operations."""
    n = 3
    identity_matrix = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    augmented_matrix = [matrix[i] + identity_matrix[i] for i in range(n)]

    # Perform Gaussian elimination
    for i in range(n):
        # Make the diagonal contain all 1's
        diag_element = augmented_matrix[i][i]
        augmented_matrix[i] = [elem / diag_element for elem in augmented_matrix[i]]
        
        # Make the other elements in the column contain 0's
        for j in range(n):
            if i != j:
                row_factor = augmented_matrix[j][i]
                augmented_matrix[j] = [augmented_matrix[j][k] - row_factor * augmented_matrix[i][k] for k in range(2 * n)]
                
    inverse_matrix = [row[n:] for row in augmented_matrix]
    return inverse_matrix


def infinty_matrix_norm():
    pass
def one_matrix_norm():
    pass
def print_matrix(matrix):
    """Print the matrix in a readable format."""
    for row in matrix:
        print(" ".join(f"{elem:.4f}" for elem in row))

def save_data(data, filename='matrix_data.json'):
    """Save matrix data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_data(filename='matrix_data.json'):
    """Load matrix data from a JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def input_vector():
    """Get a 3x1 vector from user input."""
    print("Enter the elements of the 3x1 vector:")
    vector = VECTOR_ZIRO
    for i in range(3):
        vector = float(input("Enter element {i+1}: "))
    return vector



def input_matrix():
    input_matrix=[]
    row=[]
    print("Enter a 3x3 matrix input by rows")
    for i in range(3):
        print(f"Enter row {i+1}: ")
        for j in range(3):
            row.append(input())
        input_matrix.append(row)
    return input_matrix

def menu():
    data = load_data()
    
    while True:
        print("\nMatrix Analysis Toolkit")
        print("1. Enter a new matrix")
        print("2. Display current matrix")
        print("3. Find inverse of current matrix")
        print("4. Calculate norm of current matrix")
        print("5. Calculate norm of inverse matrix")
        print("6. Calculate condition number of current matrix")
        print("7. Save current matrix and calculations")
        print("8. Load saved matrix and calculations")
        print("9. Exit")
        
        choice = input("Choose an option: ")
        
        if choice == '1':
            matrix = input_matrix()
            data['matrix'] = matrix
            print("Matrix entered successfully.")
        elif choice == '2':
            if 'matrix' in data:
                print("Current Matrix A:")
                print_matrix(data['matrix'])
            else:
                print("No matrix found. Please enter a matrix first.")
        elif choice == '3':
            if 'matrix' in data:
                matrix_inv = find_inverse(data['matrix'])
                data['inverse'] = matrix_inv
                print("Inverse of A:")
                print_matrix(matrix_inv)
            else:
                print("No matrix found. Please enter a matrix first.")
        elif choice == '4':
            if 'matrix' in data:
                norm_A = matrix_inf_norm(data['matrix'])
                data['norm'] = norm_A
                print(f"Norm of A: {norm_A:.4f}")
            else:
                print("No matrix found. Please enter a matrix first.")
        elif choice == '5':
            if 'inverse' in data:
                norm_A_inv = matrix_inf_norm(data['inverse'])
                data['norm_inverse'] = norm_A_inv
                print(f"Norm of A^-1: {norm_A_inv:.4f}")
            else:
                print("No inverse matrix found. Please calculate the inverse first.")
        elif choice == '6':
            if 'matrix' in data:
                cond_A = condition_number(data['matrix'])
                data['condition_number'] = cond_A
                print(f"Condition number of A: {cond_A:.4f}")
            else:
                print("No matrix found. Please enter a matrix first.")
        elif choice == '7':
            save_data(data)
            print("Data saved successfully.")
        elif choice == '8':
            data = load_data()
            print("Data loaded successfully.")
        elif choice == '9':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    menu()