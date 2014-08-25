import numpy as np
import theano
import theano.tensor as T


# general syntax:
#result, updates = theano.scan(fn=lambda S, P, N: operation for each scan,
#                              outputs_info = initial value of the parameters,
#                              non_sequences= Unchanging variables during scan,
#                              sequences = iterate loop for this sequence, passed to "S"
#                              n_steps= no of iteration, when we dun have any sequence to repeat from)

# The general order of function parameters to 'fn' (or the lambda function) is:
# S: sequences (if any), P: prior result(s) [or initialization of these prior result variable] (if needed), N: non-sequences (if any)

# result is mostly the list containing the result of each pass/ iteration of the loop

#############################################################################################################
####################################################################################################
###################################################################################


#
## first "n_steps", without a sequence
#

# A be a matrix and result = unit matrix,
# compute result = A**k
# computing result = result*A

k = T.iscalar("k")
A = T.vector("A")

# Symbolic description of the result
result, updates = theano.scan(fn=lambda prior_result, A: prior_result * A,
                              outputs_info=T.ones_like(A),
                              non_sequences=A,
                              n_steps=k)

# We only care about A**k, but scan has provided us with A**1 through A**k.
# Note: "n_steps=k"
# Discard the values that we don't care about. Scan is smart enough to
# notice this and not waste memory saving them.



# result above is a 3D tensor containing the value of A**k for each step

# compiled function that returns A**k
power = theano.function(inputs=[A,k], outputs=result, updates=updates)

# if we need only the last result, i.e. for step =k, we can use the following version
#final_result = result[-1]
#power = theano.function(inputs=[A,k], outputs=final_result, updates=updates)
# updates give the update rules from the scan

print power(range(10),3)



#############################################################################################################
####################################################################################################
###################################################################################

#
## With sequence
#

# A be a matrix and result = unit matrix,
# compute result = A**k
# computing result = result*A


coefficients = theano.tensor.vector(name="coefficients")
x = T.scalar("x")

max_coefficients_supported = 10000

# Generate the components of the polynomial
components, updates = theano.scan(fn=lambda coefficient, power, free_variable: coefficient * (free_variable ** power),
                                  outputs_info=None,
                                  sequences=[coefficients, theano.tensor.arange(max_coefficients_supported)],
                                  non_sequences=x)
#theano.tensor.arange(max_coefficients_supported) = [1,2,3,....,9999] and assume coefficients = [9,2,5,0,3]
#so for each iteration of above scan will be a element of zip(coefficients,theano.tensor.arange(max_coefficients_supported)).
# i.e. step1 :  coefficient = 9 and power = 1
#step1 :  coefficient = 2 and power = 2
#this will stop when we reach the end of any of the terms in sequence 

#here non_sequences=x implies that the x is will not change in the function and the will be passed as the parameter, when we call the function,
#in above code value of x will be assigned to free_variable

#
# In short:
# at iteraton i, the parameters passed will be
# coefficient = coefficients[i] 
# power = theano.tensor.arange(max_coefficients_supported)[i]
# free_variable = x    [entire x (if that would have been an array or matrix) is passed to free_variable]

# Sum them up
#polynomial = components.sum()
# Compile a function
#calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=polynomial)

# component (a list) retured by scan contain the result of each iteration
calculate_polynomial = theano.function(inputs=[coefficients, x], outputs=components)

# Test
test_coefficients = np.asarray([1, 0, 2], dtype=np.float32)
test_value = 3
print calculate_polynomial(test_coefficients, test_value)
print 1.0 * (3 ** 0) + 0.0 * (3 ** 1) + 2.0 * (3 ** 2)


#############################################################################################################
####################################################################################################
###################################################################################



up_to = T.iscalar("up_to")

# define a named function, rather than using lambda
def accumulate_by_adding(arange_val, sum_to_date):
    return sum_to_date + arange_val
seq = T.arange(up_to)

# An unauthorized implicit downcast from the dtype of 'seq', to that of
# 'T.as_tensor_variable(0)' which is of dtype 'int8' by default would occur
# if this instruction were to be used instead of the next one:
# outputs_info = T.as_tensor_variable(0)

outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype))
#so as seq is int32
#outputs_info will be a scalar variable of type int32

scan_result, scan_updates = theano.scan(fn=accumulate_by_adding,
                                        outputs_info=outputs_info,
                                        sequences=seq)
# pass 0:
# arange_val = seq[0] and sum_to_date = outputs_info
# check the order of parameter passing:
# S: sequences (if any), P: prior result(s) [or initialization of these prior result variable] (if needed), N: non-sequences (if any)
# S: seq, P: outputs_info. P is prior result.

triangular_sequence = theano.function(inputs=[up_to], outputs=scan_result)


# test
some_num = 15
print triangular_sequence(some_num)
print [n * (n + 1) // 2 for n in xrange(some_num)]



# for rest all see:
#http://deeplearning.net/software/theano/library/scan.html
