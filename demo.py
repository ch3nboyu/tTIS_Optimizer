import copy
from typing import List

class Solution:
    def gameOfLife(self, board: List[List[int]]) -> None:
        if not board: return board
        m, n = len(board), len(board[0])
        res = copy.deepcopy(board)

        def check(i, j):
            cells = 0
            for row in range(max(i-1,0), min(i+1,m)):
                for col in range(max(j-1,0), min(j+1,n)):
                    print(row, col)
                    cells += board[row,col]
            print(cells)
            exit()
        
        for i in range(m):
            for j in range(n):
                res[i][j] = check(i,j)
        
        board = res

print(Solution().gameOfLife([[0,1,0],[0,0,1],[1,1,1],[0,0,0]]))