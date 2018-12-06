observations_head_A = [9,8,7]  # 观察A硬币扔到正面的次数
observations_tail_A = [1,2,3]  # 观察A硬币扔到反面的次数
observations_head_B = [5,4]  # 观察B硬币扔到正面的次数
observations_tail_B = [5,6]  # 观察B硬币扔到反面的次数

count = 0
for i in observations_head_A:
	count += i
probability_A = count / (10 * len(observations_head_A))
count = 0
for i in observations_head_B:
	count += i
probability_B = count / (10 * len(observations_head_B))

print("The probability that coin A will be thrown to the front:", probability_A)
print("The probability that coin B will be thrown to the front:", probability_B)