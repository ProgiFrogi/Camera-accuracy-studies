def MSE(array_1, array_2):
    if array_1 is None or array_2 is None:
        return 0
    if array_1.shape != array_2.shape:
        raise ValueError("Different array len")
    result = 0
    for i in range(len(array_1)):
        result += (array_1[i] - array_2[i]) ** 2
    return result/len(array_1)


def RMSE(array_1, array_2):
    return MSE(array_1, array_2) ** 0.5