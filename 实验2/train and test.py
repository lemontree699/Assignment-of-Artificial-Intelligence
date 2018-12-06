from scipy import sparse
from numpy import *
import numpy

# 700 * 2500的矩阵
numTrainDocs = 700
numTokens = 2500

with open('train-features.txt','r') as f:
	# 将文件中的数全部存入一个list中，其中每三个数第一个代表矩阵的行、第二个代表矩阵的列、第三个代表矩阵对应位置的数据
	ele = f.read().split()
	# print(list)
	# 将行单独列出来，从第一个数开始每隔三个数取一个。因为python中数组下标从0开始，所以需要做减1的操作
	row = [int(x) - 1 for x in ele[::3]]
	# 将列单独列出来，从第一个数开始每隔三个数取一个。因为python中数组下标从0开始，所以需要做减1的操作
	col = [int(x) - 1 for x in ele[1::3]]
	# 将数据项单独列出来，即每个单词出现的次数
	data = [int(x) for x in ele[2::3]]
	# 将数据项存入稀疏矩阵中
	spmatrix = sparse.coo_matrix((data, (row, col)), shape = (numTrainDocs, numTokens))
	train_matrix = spmatrix.toarray()
	# print(spmatrix)
	# print(train_matrix)

train_labels = list()
# 每份邮件是否为垃圾邮件
M = numpy.loadtxt('train-labels.txt',dtype = int)
for i in M:
	train_labels.append(i)
# print(train_labels)

# 垃圾邮件与非垃圾邮件的份数
spam_indices = nonspam_indices = 0
for i in train_labels:
	if(i == 1):
		spam_indices += 1
	else:
		nonspam_indices += 1

# 垃圾邮件出现的概率
prob_spam = spam_indices / numTrainDocs

# 每份邮件的单词数
email_lengths = numpy.sum(train_matrix, axis = 1)
# print(email_lengths)

# 垃圾邮件与非垃圾邮件出现的单词的个数
spam_wc = numpy.sum(email_lengths[spam_indices:])
nonspam_wc = numpy.sum(email_lengths[:nonspam_indices])
# print(spam_wc, nonspam_wc)

# Now the k-th entry of prob_tokens_spam represents phi_(k|y=1)
prob_tokens_spam = (numpy.sum(train_matrix[spam_indices:], axis = 0) + 1) / (spam_wc + numTokens)
# Now the k-th entry of prob_tokens_nonspam represents phi_(k|y=0)
prob_tokens_nonspam = (numpy.sum(train_matrix[:nonspam_indices], axis = 0) + 1) / (nonspam_wc + numTokens)
# print(prob_tokens_spam)
# print(prob_tokens_nonspam)


# from scipy import sparse
# import numpy
import math

# 打开文件test-features并将其数据存入矩阵中
with open('test-features.txt', 'r') as f:
	ele = f.read().split()
	row = [int(x) - 1 for x in ele[::3]]
	col = [int(x) - 1 for x in ele[1::3]]
	data = [int(x) for x in ele[2::3]]
	spmatrix = sparse.coo_matrix((data, (row, col)))
	test_matrix = spmatrix.toarray()
	# print(spmatrix)
	# print(test_matrix)

numTestDocs = len(test_matrix)
numTokens = len(test_matrix[0])
# print(numTestDocs,numTokens)

output = list()
log_a = numpy.dot(test_matrix,(list(map(math.log,prob_tokens_spam)))) + math.log(prob_spam)
log_b = numpy.dot(test_matrix,(list(map(math.log,prob_tokens_nonspam)))) + math.log(1 - prob_spam)
for i in range(len(log_a)):
	if(log_a[i] > log_b[i]):
		output.append(1)
	else:
		output.append(0)
# print(output)

# 打开test-labels文件并将其中数据存入test_labels中
with open('test-labels.txt', 'r') as f:
	ele = f.read().split()
	test_labels = list()
	for i in ele:
		test_labels.append(int(i))
# print(test_labels)

# 计算判断错误的邮件的数量和比例
numdocs_wrong = 0
for i in range(len(output)):
	if(output[i] ^ test_labels[i] == 1):
		numdocs_wrong += 1
print('numdocs_wrong =', numdocs_wrong)
fraction_wrong = numdocs_wrong / numTestDocs
print('fraction_wrong =', fraction_wrong)