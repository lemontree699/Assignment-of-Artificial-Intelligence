def em_single(probability_a, probability_b, observations_head, observations_tail):
	new_observations_head_a = 0.0
	new_observations_tail_a = 0.0
	new_observations_head_b = 0.0
	new_observations_tail_b = 0.0
	for i in range(len(observations_head)):
		pa = pow(probability_a, observations_head[i]) * pow(1-probability_a, observations_tail[i])  # 为A的概率
		pb = pow(probability_b, observations_head[i]) * pow(1-probability_b, observations_tail[i])  # 为B的概率
		normalized_pa = pa / (pa + pb)  # 归一化后的P(A)
		normalized_pb = pb / (pa + pb)  # 归一化后的P(B)
		
		new_observations_head_a += normalized_pa * observations_head[i]
		new_observations_tail_a += normalized_pa * observations_tail[i]
		new_observations_head_b += normalized_pb * observations_head[i]
		new_observations_tail_b += normalized_pb * observations_tail[i]

	new_probaliblity_a = new_observations_head_a / (new_observations_head_a + new_observations_tail_a)
	new_probaliblity_b = new_observations_head_b / (new_observations_head_b + new_observations_tail_b)

	return new_probaliblity_a, new_probaliblity_b

def em(origin_a, origin_b, observations_head, observations_tail):
	count = 0
	new_probaliblity_a = origin_a
	new_probaliblity_b = origin_b
	while(1):
		probability_a = new_probaliblity_a
		probability_b = new_probaliblity_b
		new_probaliblity_a, new_probaliblity_b = em_single(new_probaliblity_a, new_probaliblity_b, observations_head, observations_tail)
		delta_a = new_probaliblity_a - probability_a
		delta_b = new_probaliblity_b - probability_b
		if delta_a < 1e-6 and delta_b < 1e-6:
			print("Number of iterations:", count)
			print("The probability that coin A will be thrown to the front:", new_probaliblity_a)
			print("The probability that coin B will be thrown to the front:", new_probaliblity_b)
			return new_probaliblity_a, new_probaliblity_b
		else:
			count += 1
			pass

observations_head = [5,9,8,4,7]  # 观察扔到正面的次数
observations_tail = [5,1,2,6,3]  # 观察扔到反面的次数
origin_a = 0.6  # 初始化硬币A扔正面的概率
origin_b = 0.4 # 初始化硬币B扔正面的概率
em(origin_a, origin_b, observations_head, observations_tail)