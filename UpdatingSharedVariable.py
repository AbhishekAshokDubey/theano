import numpy as np
import theano
import theano.tensor as T


def updateFunc(ps):
	updates = []
	print 'Inside the function'
	for p in ps:
		param_update = theano.shared(0)
		updates.append((p, p+param_update))
		updates.append((param_update, param_update+1))
	return updates

W1 = theano.shared(value=np.array([[1,1],[1,1]]),name='W1')
b1 = theano.shared(value=np.array([1,1]),name='b1')
W2 = theano.shared(value=np.array([[2,2],[2,2]]),name='W2')
b2 = theano.shared(value=np.array([2,2]),name='b2')

l1 = [W1,b1]
l2 = [W2,b2]

paramsList = []
paramsList.append(l1)
paramsList.append(l2)

# paramsList = [[W1, b1], [W2, b2]]

params = []
for param in paramsList:
	params = params+param
# above for loop makes params = [W1,b1,W2,b2]


train = theano.function([],updates=updateFunc(params))
# Before this declaration of train function, params should be defined completely.
# when above declaration is encountered the graph is made for train function using the updateFunc.
# So the updateFunc will be executed even though we have not called the train function yet.

# Most important point to note:
# when the graph for train function is build at line 35 [ train = theano.function([],updates=updateFunc(params)) ]
# at that time depending on the number of elements in params, the equivalent no of param_update are made one for each param in params.
# as they (param_update) are also shared, they will retain their values.
# Note that the updateFunc function and hence setting the values to zero for param_update is executed only once to make the graph,
# so next time on the graph will decide what to do and the updateFunc will not be called.

i = 0
while i < 4:
	print "after call", i
	train()
	print W1.get_value()
	print b1.get_value()
	print W2.get_value()
	print b2.get_value()
	print "-------------------"
	print "-------------------"
	i = i +1
	
