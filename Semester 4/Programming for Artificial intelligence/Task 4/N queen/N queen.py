def issafe(board,row,column,n):
    for i in range(column):
        if board[row][i]==1:
            return False
    for i,j in zip(range(row,-1,-1), range(column,-1,-1)):
        if board[i][j]==1:
            return False
    for i,j in zip(range(row,n,1), range(column,-1,-1)):
        if board[i][j]==1:
            return False
    return True
def solve(board,column,n):
    if column>=n:
        return True
    for i in range(n):
        if issafe(board,i,column,n):
            board[i][column]=1
            if solve(board,column+1,n):
                return True
            board[i][column]=0
    return False
def solvequeen(n):
    board=[[0 for _ in range(n)] for _ in range(n)]
    if not solve(board,0,n):
        print("Solution does not exist")
        return False
    printsol(board)
    return True
def printsol(board):
    for row in board:
        print(" ".join("Q" if x==1 else "." for x in row))
n=8
solvequeen(n)