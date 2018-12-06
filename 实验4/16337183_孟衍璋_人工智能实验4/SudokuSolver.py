# 命令行参数:数独谜题文件和算法名称
# BF brute force search
# BT back-tracking
# FC-MRV forward checking with minimum remaining values
# enforce generalized arc consistency(GAC)
# for example:Python SudokuSolver puzzle1.txt BF 
			# Python SudokuSolver puzzle2.txt BT 
			# Python SudokuSolver puzzle3.txt FC-MRV 
			# Python SudokuSolver puzzle4.txt GAC

# 程序输出到屏幕必须遵循：
# 1.显示解决方案
# 2.输出运行时间（毫秒）和生成的节点数
# 3.必须将这两个规范分别保存到两个文件中：解决方案，例如'puzzle1.txt'，'solution1.txt'; 并将时间和节点评估为'performance1.txt'
# Total clock time: 1000.00
# Search clock time: 800.00 
# Number of nodes generated: 500

import sys
import time
import brute_force
import backtracking
import forward_checking
import generalized_arc_consistency

file = sys.argv[1]
method = sys.argv[2]

if len(sys.argv) != 3:
	print('Your input is illegal.')
	sys.exit()

if(method == 'BF'):
	puzzle = brute_force.read_puzzle(file)
	brute_force.bf(puzzle)

elif(method == 'BT'):
	puzzle = backtracking.read_puzzle(file)
	backtracking.bt(puzzle)

elif(method == 'FC-MRV'):
	puzzle = forward_checking.read_puzzle(file)
	remaining = [[],[],[],[],[],[],[],[],[]]
	for j in range(9):
		for k in range(9):
			remaining[j].append(forward_checking.can_fill(j,k,puzzle))
	forward_checking.fc_mrv(remaining,puzzle)

elif(method == 'GAC'):
	puzzle = generalized_arc_consistency.read_puzzle(file)
	remaining = [[],[],[],[],[],[],[],[],[]]
	for j in range(9):
		for k in range(9):
			remaining[j].append(generalized_arc_consistency.can_fill(j,k,puzzle))
	generalized_arc_consistency.gac(remaining,puzzle)

else:
	print('Your input is illegal.')