import itertools
import time
import sys

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

def not_done(puzzle):
    return True in [0 in r for r in puzzle]

def go_around(puzzle):
    ans = []
    global count
    for index_r, r in enumerate(puzzle):
        row = []
        for index_c, c in enumerate(r):
            if 0 == c:
                count += 1
                maybe_ans = can_fill(index_r, index_c, puzzle)
                row.append(maybe_ans[0] if len(maybe_ans) == 1 \
                           else 0)
            else:
                row.append(c)
        ans.append(row)
    return ans

def bf(puzzle):
	search_start_time = time.time()
	while not_done(puzzle):
		halfway = time.time()
		#　如果搜索时间大于30秒，就退出程序
		if halfway - search_start_time >= 30:
			print('Number of nodes generated:', count)
			sys.exit()
		puzzle = go_around(puzzle)
	search_end_time = time.time()
	end_time = time.time()
	write_solution('solution_bf.txt', puzzle)
	write_performance('performance_bf.txt', start_time, end_time, search_start_time, search_end_time, count)