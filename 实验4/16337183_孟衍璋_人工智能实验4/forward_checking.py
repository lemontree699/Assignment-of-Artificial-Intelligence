import sys
import time

start_time = time.time()

puzzle = []
remaining = [[],[],[],[],[],[],[],[],[]]
solution = []
def read_puzzle(file):
	with open(file) as f:
		lines = f.readlines()
		for data in lines:
			# print(data.split())
			puzzle.append([int(x) for x in data.split()])
	return puzzle

# 判断某一个位置还能填什么数字，返回一个list
def can_fill(x,y,puzzle):
	a = []
	# 如果当前元素不为0，直接返回一个空列表
	if puzzle[x][y] != 0:
		return a

	appear = {1:False, 2:False, 3:False, 4:False, 5:False, 6:False, 7:False, 8:False, 9:False}
	# 当前行出现过的元素都置True
	for i in range(9):
		if(puzzle[x][i] != 0):
			appear[puzzle[x][i]] = True
	# 当前列出现过的元素都置True
	for i in range(9):
		if(puzzle[i][y] != 0):
			appear[puzzle[i][y]] = True
	# 当前九宫格出现过的元素都置True
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
			if (puzzle[row*3+j][col*3+k] != 0):
				appear[puzzle[row*3+j][col*3+k]] = True

	# 将还能填的数字存在一个list中
	for i in appear:
		if(appear[i] == False):
			a.append(i)
	return a 

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

# 选取remaining中可以选择的值最少的地方开始访问
def start_point(remaining,puzzle):
	i = j = 0
	x = y = 0
	smallest = float("inf")
	for i in range(9):
		for j in range(9):
			if len(remaining[i][j]) <= smallest and len(remaining[i][j]) != 0:
				smallest = len(remaining[i][j])
				x = i
				y = j
	return x,y

import copy
count = 0
search_start_time = time.time()
# forward_checking
def fc_mrv(remaining,puzzle):
	global count
	count += 1
	x,y = start_point(remaining,puzzle)
	# 如果所有的空都填上了
	judge = 1
	for i in range(9):
		for j in range(9):
			if puzzle[i][j] == 0:
				judge = 0
	if judge:
		search_end_time = time.time()
		end_time = time.time()
		write_solution('solution_fc.txt', puzzle)
		write_performance('performance_fc.txt', start_time, end_time, search_start_time, search_end_time, count)
		# sys.exit()
		return True

	for i in remaining[x][y]:
		new_remaining = copy.deepcopy(remaining)
		new_puzzle = copy.deepcopy(puzzle) 
		new_remaining[x][y] = []
		new_puzzle[x][y] = i
		# remaining[x][y] = []
		# puzzle[x][y] = i
		# 更新remaining
		for j in range(9):
			if i in new_remaining[x][j] and j != y:
				new_remaining[x][j].remove(i)
			if i in new_remaining[j][y] and j != x:
				new_remaining[j][y].remove(i)
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
				if i in new_remaining[row*3+j][col*3+k] and x != (col*3+j) and y != (col*3+j):
					new_remaining[row*3+j][col*3+k].remove(i)
		
		if fc_mrv(new_remaining,new_puzzle) == 0:
			# 如果后面的递归搜索不满足要求，令new_puzzle[i][j] = 0
			new_puzzle[x][y] == 0
		else:
			return True
	# 如果该点遍历1-9都不符合要求，则表示上游选值不当，回溯 
	return False

# read_puzzle('puzzle.txt')
# print('puzzle:')
# for i in puzzle:
# 	print(i)
# # 将每个位置可以填入的数字存入remaining中
# for j in range(9):
# 	for k in range(9):
# 		remaining[j].append(can_fill(j,k,puzzle))
# fc_mrv(remaining,puzzle)