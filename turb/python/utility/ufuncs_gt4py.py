from gt4py import gtscript


# Return the difference x-y if the result is positive, otherwise return 
# zero
@gtscript.function
def dim(x, y):
    
    diff = x - y
    
    return diff if diff > 0. else 0.
