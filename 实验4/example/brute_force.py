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

def judge(a):
	for x in range(9):
		for y in range(9):
			num = a[x][y]
			a[x][y] = 0
			for i in range(0,9):
				if question[x][i] == num:
					a[x][y] = num
					return 0

			for i in range(0,9):
				if question[i][y] == num:
					a[x][y] = num
					return 0

			for i in range(int(x / 3) * 3, (int(x / 3) + 1) * 3):
				for j in range(int(y / 3) * 3, (int(y / 3) + 1) * 3):
					if question[i][j] == num:
						a[x][y] = num
						return 0
			a[x][y] = num
	return 1
	pass
ans = []
def force(x, y, temp):
	if temp[x][y] == 0:
		for i in range(1, 10):
			temp[x][y] = i
			if x == 8 and y == 8:
				ans.append(temp)
			elif y == 8 and x != 8:
				force(x + 1, 0, temp)
			else:
				force(x, y + 1, temp)
			temp[x][y] = 0
	else:
		if x == 8 and y == 8:
			ans.append(temp)
			return
		elif y == 8 and x != 8:
			force(x + 1, 0, temp)
		else:
			force(x, y + 1, temp)
	return
#brute force
read_board()
force(0, 0, question)
for i in ans:
	if judge(i) == 1:
	 	print(i)