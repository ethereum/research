import sys
import copy

def mkmatrix(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def print_form(mat):
    return '\n'.join(''.join(str(val) for val in row) for row in mat)

def recover(matrix):
    rows, cols = len(matrix), len(matrix[0])
    matrix = copy.deepcopy(matrix)
    print(print_form(matrix))
    for _round in range(1, rows + cols + 1):
        print(f"\nRound {_round}")
        rows_to_recover = {
            i for i in range(rows) if
            cols <= sum(matrix[i]) * 2 < cols * 2
        }
        cols_to_recover = {
            i for i in range(cols) if
            rows <= sum(matrix[j][i] for j in range(rows)) * 2 < rows * 2
        }
        if rows_to_recover:
            print(f"Recovering rows: {rows_to_recover}")
            for row in rows_to_recover:
                matrix[row] = [1] * rows
        if cols_to_recover:
            print(f"Recovering cols: {cols_to_recover}")
            for col in cols_to_recover:
                for i in range(rows):
                    matrix[i][col] = 1
        print(print_form(matrix))
        if sum(sum(row) for row in matrix) == rows * cols:
            print(f"Finished in {_round} rounds")
            return _round
        if rows_to_recover == cols_to_recover == set():
            print("Recovery failed")
            return None
    raise Exception("wtf happened here ^_^")

# Example input: '110 110 000' or '110\n100\n000'
def parse(text):
    separator = '\n' if '\n' in text else ' '
    rows = text.strip().split(separator)
    return [[int(x) for x in row] for row in rows]

def mk_evil_matrix(n):
    odd_n = n - ((n+1) % 2)
    half_odd_n = odd_n // 2
    o = mkmatrix(n, n)
    for i in range(half_odd_n):
        for j in range(half_odd_n + i, odd_n):
            o[i][j] = 1
    o[half_odd_n][half_odd_n] = 1
    for i in range(1, half_odd_n+1):
        for j in range(i):
            o[half_odd_n + i][j] = 1
    return o

def test(n=12):
    recover(mk_evil_matrix(n))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        test(int(sys.argv[-1]))
    else:
        test()
