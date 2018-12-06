import copy
question = []
#save the soduku in the list
def read_board():
	with open('puzzle.txt') as f:
		data = f.readlines()
		for line in data:
			temp = line.split()
			temp = [int(x) for x in temp]
			question.append(temp)
	pass

def judge(x, y, num):
	for i in range(0,9):
		if question[x][i] == num:
			return 0

	for i in range(0,9):
		if question[i][y] == num:
			return 0

	for i in range(int(x / 3) * 3, (int(x / 3) + 1) * 3):
		for j in range(int(y / 3) * 3, (int(y / 3) + 1) * 3):
			if question[i][j] == num:
				return 0
	return 1
	pass

def forward_checking(x, y, temp, board):
	loc = x * 9 + y
	if board[x][y] != 0:
		if x == 8 and y == 8:
			for i in temp2:
				print(i)
			return
		elif y == 8:
			return forward_checking(x + 1, 0, temp, board)
		else:
			return forward_checking(x, y + 1, temp, board)
	for i in temp[loc]:
		temp1 = copy.deepcopy(temp)
		temp2 = copy.deepcopy(board)
		temp1[loc] = []
		temp2[x][y] = i
		for j in range(9):
			if i in temp1[x * 9 + j] and j != y:
				temp1[x * 9 + j].remove(i)
			if i in temp1[j * 9 + y] and j != x:
				temp1[j * 9 + y].remove(i)
		for m in range(int(x / 3) * 3, (int(x / 3) + 1) * 3):
			for n in range(int(y / 3) * 3, (int(y / 3) + 1) * 3):	
				if i in temp1[m * 9 + n] and x != m and y != m:
					temp1[m * 9 + n].remove(i)
		if x == 8 and y == 8:
			for i in temp2:
				print(i)
		elif y == 8:
			forward_checking(x + 1, 0, temp1, temp2)
		else:
			forward_checking(x, y + 1, temp1, temp2)

read_board()
possibilities = []
for i in range(9):
	for j in range(9):
		temp = []
		if question[i][j] == 0:
			for m in range(1, 10):
				if judge(i, j, m) == 1:
					temp.append(m)
		possibilities.append(temp)
# print(possibilities)
forward_checking(0, 0, possibilities, question)