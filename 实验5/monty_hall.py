#!/usr/bin/env python
# -*- coding:utf-8 -*-

from bayesian.bbn import build_bbn

def f_prize_door(prize_door):
	return 0.33333333

def f_guest_door(guest_door):
	return 0.33333333

def f_monty_door(prize_door, guest_door, monty_door):
	if prize_door == guest_door:  # 参赛者选择了藏有汽车的门
		if prize_door == monty_door:
			return 0  # Monty永远不可能选到有汽车的门
		else:
			return 0.5  # 剩下的两个门Monty任选一个
	elif prize_door == monty_door:
		return 0  # Monty永远不会选到有汽车的门
	elif guest_door == monty_door:
		return 0  # Monty永远不会选择与参赛者相同的门
	else:
		return 1  # 参赛者没有选择到有汽车的门，Monty在剩下的两个门当中只能选剩下的没有汽车的门

if __name__ == '__main__':
	g = build_bbn(f_prize_door, f_guest_door, f_monty_door, 
		domains = dict(prize_door = ['A', 'B', 'C'], 
			guest_door = ['A', 'B', 'C'], 
			monty_door = ['A', 'B', 'C']))

