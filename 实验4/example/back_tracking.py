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

#back tracking
def back_tracking(x, y, temp):
	if temp[x][y] != 0:
		if x == 8 and y == 8:
			return 1
		elif y == 8:
			return back_tracking(x + 1, 0, temp)
		else:
			return back_tracking(x, y + 1, temp)

	ans = 0
	for i in range(1, 10):
		if judge(x, y, i) == 1:
			temp[x][y] = i
			if x == 8 and y == 8:
				ans = 1
				for line in temp:
					print(line)
				print()
			elif y == 8 and x != 8:
				ans = back_tracking(x + 1, 0, temp)
			else:
				ans = back_tracking(x, y + 1, temp)
			temp[x][y] = 0
	return ans

read_board()
a = back_tracking(0, 0, question)