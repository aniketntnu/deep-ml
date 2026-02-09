#import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	reshaped_matrix = [[]]

	r,c = len(a),len(a[0])
	r1,c1= new_shape[0],new_shape[1]

	total= r*c

	#print("\n\t  r:",r,"c:",c," total:",total," new_shape:",new_shape)

	if not total == (new_shape[0]*new_shape[1]):
		return []

	reshaped_matrix = [[None for _ in range(c1)] for _ in range(r1)]
	
	flatten =[]

	for i in range(r):
		for j in range(c):
			flatten.append(a[i][j])

	#print(" \n\t flatten:",flatten)
	totIndx=0
	for i in range(r1):
		for j in range(c1):

			tot1 = c1*(i-1)+c1

			reshaped_matrix[i][j]=flatten[totIndx]
			totIndx+=1
	
	return reshaped_matrix



