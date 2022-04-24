import numpy as np

serum_oh = [0,0,0,9,25,0,16,0,0,0,0,0,0,0,0,0,9,25,0,16,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,3,0,0,0,16,0,0,3,0,2,
0,0,0,0,0,2,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,2,2,0,0,3,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,
0,0,0,8,2,0,0,2,2,2,2,2,2,2,2,2,2,0,0,0,0,0,0,6,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,2,2,2,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

diva_oh =[0,2,2,0,8,6,5,2,0,0,0,0,0,2,0,0,3,0,4,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,
0,2,2,2,0,0,0,0,0,0,3,0,2,2,2,0,0,4,8,0,0,0,0,0,0,0,0,4,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,5,0,0,0,0,0,0,0,0,0,2,0,0,0,0,4,6,6,23,0,23,0,23,0,23,0,2,2,2,2,2,2,2,2,2,3,2,0,0,2,
2,2,2,2,2,4,4,4,2,0,23,0,23,0,4,0,0,2,0,23,0,0,5,0,0,23,0,23,0,0,0,2,2,2,4,0,23,0,23,0,
23,0,0,0,2,23,0,23,0,0,0,0,0,5,3,0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
3,0,0,0,0,0,0,0,0,4,5,3,0,0,0,2,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,0,0,0,
0,0,0,0,0,4,0,4,0,6,4,0,0,2,0,0,0,0,0,0,0,0,2,2,0,0,0,0,7,7,2]

tyrell_oh =[0,18,0,18,0,18,0,18,0,8,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,0,0,0,0,0,0,0,4,0,
0,0,0,0,18,0,0,0,0,18,0,0,18,0,0,0,0,6,3,0,0,0,0,0,0,0,2,3,18,0,18,0,0,0,0,18,0,0,18,3,
0,0,0,4,8,0,0,0,0,0,4,8,0,0,0,0,0]

def encode(params,oh_code):
    if len(params) != len(oh_code):
        raise ValueError("make sure the parameters and one  hot code arrays are the same length")

    final_array = []
    for i,p in enumerate(params):
        #just keep the parameter when only one category
        if oh_code[i] <= 1:
            final_array.append(p)
        else:
            #make one hot encoding
            for n in range(oh_code[i]):
                if np.rint(p*(oh_code[i]-1)) == n:
                    final_array.append(1)
                else:
                    final_array.append(0)

    return np.array(final_array)

def decoded(encoded,oh_code):
    final_array = []
    i = 0
    for c in oh_code:
        if c <= 1:
            final_array.append(encoded[i])
            i += 1
        else:
            #decode one hot
            for n in range(c):
                if encoded[i] == 1:
                    final_array.append(n/(c-1))
                i += 1

    return np.array(final_array)

def predict(encoded,oh_code):
    final_array = []
    i = 0
    for c in oh_code:
        if c <= 1:
            final_array.append(encoded[i])
            i += 1
        else:
            #decode one hot
            max = 0
            max_i = 0
            for n in range(c):
                final_array.append(0)
                if encoded[i] >= max:
                    max = encoded[i]
                    max_i = i
                i += 1
            final_array[max_i] = 1
    
    return np.array(final_array)
            


def main():
    # hot_loc = [1,3,1,1,4,5,1]
    # original_code = [0.2,0.5,0.323,.43,0.6667,0.25,0.93]
    # one_hot_code = encode(original_code, hot_loc)
    # d_code =decoded(one_hot_code, hot_loc)
    origin = [.5,.45,0,0.3,0.1,1,0.5,0.9,0.1,0.2,0.1]
    print(origin)
    rounded = predict(origin, [0,0,3,0,5])
    print(rounded)
    print(decoded(rounded,[0,0,3,0,5]))
    en = encode([0.5, 0.45, 1,   1,   0.25], [0,0,3,0,5])
    print(en)


if __name__ == "__main__":
    main()