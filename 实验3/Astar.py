import numpy

a,b,c,d,e,f,g,h,i,l,m,n,o,p,r,s,t,u,v,z = range(20)

city_map = [
	{s:140, t:118, z:75}, # a
	{f:211, g:90, p:101, u:85}, # b
	{d:120, p:138, r:146}, # c
	{c:120, m:75}, # d
	{h:86}, # e
	{b:211, s:99}, # f
	{b:90}, # g
	{e:86, u:98}, # h
	{n:87, v:92}, # i
	{m:70, t:111}, # l
	{d:75, l:70}, # m
	{i:87}, # n
	{s:151, z:71}, # o
	{b:101, c:138, r:97}, # p
	{c:146, p:97, s:80}, # r
	{a:140, f:99, o:151, r:80}, # s
	{a:118, l:111}, # t
	{b:85, h:98, v:142}, # u
	{i:92, u:142}, # v
	{a:75, o:71}, # z
]

straight_line_distance_to_b = {a:366, b:0, c:160, d:242, e:161, f:178, g:77, h:151, i:226, l:244, m:241, n:234, o:380, p:98, r:193, s:253, t:329, u:80, v:199, z:374}

num_to_point = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h', 8:'i', 9:'l', 10:'m', 11:'n', 12:'o', 13:'p', 14:'r', 15:'s', 16:'t', 17:'u', 18:'v', 19:'z'}

def heuristic(cost_so_far, current_point):
	return cost_so_far + straight_line_distance_to_b[current_point]

start_point = a
end_point = b
cost_so_far = 0
print("start form:", num_to_point[start_point])
current_point = start_point
route = list()

while(current_point != end_point):
	last_point = current_point
	temp_point = current_point
	h = float("inf")
	for i in city_map[current_point]:
		if(i in route):
			pass
		else:
			search = heuristic((cost_so_far + city_map[current_point][i]), i)
			if(search < h):
				h = search
				temp_point = i
	route.append(temp_point)
	current_point = temp_point
	cost_so_far += city_map[last_point][current_point]

for i in route:
	if(i != end_point):
		print("next:", num_to_point[i])
	else:
		print("end:", num_to_point[end_point])