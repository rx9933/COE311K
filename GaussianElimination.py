# Define a function that swaps two rows of a matrix
def swap_rows(matrix, i, j):
  # Swap the ith and jth rows of the matrix
  matrix[i], matrix[j] = matrix[j], matrix[i]

# Define a function that performs Gaussian elimination on a matrix
def row_ech(matrix):
  # Get the number of rows and columns of the matrix
  rows = len(matrix)
  cols = len(matrix[0])
  # Loop through the columns
  for c in range(cols - 1):
    # Find the pivot row, the one with the largest absolute value in the current column
    pivot = c
    for r in range(c + 1, rows):
      if abs(matrix[r][c]) > abs(matrix[pivot][c]):
        pivot = r
    # Swap the pivot row with the current row
    swap_rows(matrix, c, pivot)
    # Make the pivot element 1 by dividing the row by it
    pivot_element = matrix[c][c]
    for k in range(c, cols):
      matrix[c][k] /= pivot_element
    # Eliminate the elements below the pivot by subtracting multiples of the pivot row
    for r in range(c + 1, rows):
      factor = matrix[r][c] 
      for k in range(c, cols):
        matrix[r][k] -= factor * matrix[c][k]
  # Return the matrix in row echelon form
  return matrix
def gaussian_elimination(matrix):
    row_ech_matrix = row_ech(matrix)

mat = [[3.0, 2.0, -4.0], [2.0, 3.0, 3.0], [5.0, -3, 1.0]]
print(gaussian_elimination(mat))