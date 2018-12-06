import time

start_time = time.time()
puzzle = []
count = 0

def read_puzzle(file):
	with open(file) as f:
		lines = f.readlines()
		for data in lines:
			# print(data.split())
			puzzle.append([int(x) for x in data.split()])
	return puzzle

def conflict(x,y,num):
	# 判断当前行的元素与新填的元素有无冲突
	for i in range(9):
		if (puzzle[x][i] == num):
			return True
	# 判断当前列的元素与新填的元素有无冲突
	for i in range(9):
		if (puzzle[i][y] == num):
			return True
	# 判断当前九宫格的元素与新填的元素有无冲突
	if (x >= 0 and x <= 2):
		row = 0
	elif (x >= 3 and x <= 5):
		row = 1
	elif (x >= 6 and x <= 8):
		row = 2
	if (y >= 0 and y <= 2):
		col = 0
	elif (y >= 3 and y <= 5):
		col = 1
	elif (y >= 6 and y <= 8):
		col = 2
	for j in range(3):
		for k in range(3):
			if (puzzle[row*3+j][col*3+k] == num):
				return True
	# 没有冲突则返回False
	return False

def write_solution(file, puzzle):
	solution = []
	# 将结果存入solution这个列表
	for lines in puzzle:
		solution.append(lines)
	# 将结果打印出来
	print('solution:')
	for lines in solution:
		print(lines)
	# 将结果写入文件
	with open(file, 'w') as f:
		for lines in solution:
			for data in lines:
				f.write(str(data))
				f.write(' ')
			f.write('\n')

def write_performance(file, start_time, end_time, search_start_time, search_end_time, count):
	print('Total clock time:', end_time - start_time)
	print('Search clock time:', search_end_time - search_start_time)
	print('Number of nodes generated:', count)
	with open(file, 'w') as f:
		f.write('Total clock time: ')
		f.write(str(end_time - start_time))
		f.write('\nSearch clock time: ')
		f.write(str(search_end_time - search_start_time))
		f.write('\nNumber of nodes generated: ')
		f.write(str(count))

# 一行一行地遍历，遇到第一个为0的数便应该访问该位置
def start_point(puzzle):
	for i in range(9):
		for j in range(9):
			if puzzle[i][j] == 0:
				return i,j
	return i,j

search_start_time = time.time()
def bt(puzzle):
	global count
	count += 1
	i,j = start_point(puzzle)
	# 如果i，j均为8且该位置的数不为0，说明已经填完
	if i == 8 and j == 8 and puzzle[8][8]:
		search_end_time = time.time()
		end_time = time.time()
		write_solution('solution_bt.txt', puzzle)
		write_performance('performance_bt.txt', start_time, end_time, search_start_time, search_end_time, count)
		return True

	for value in range(1,10):
		# 判断有无冲突
		if conflict(i, j, value) == 0:
			# 如果当前位置没有冲突，则令puzzle[i][j]=value
			puzzle[i][j] = value 
			if bt(puzzle) == 0:
				# 如果后面的递归搜索不满足要求，令puzzle[i][j] = 0
				puzzle[i][j] = 0 
			else:
				return True
	# 如果该点遍历1-9都不符合要求，则表示上游选值不当，回溯 
	return False 

# read_puzzle('puzzle_pro.txt')
# bt(puzzle)