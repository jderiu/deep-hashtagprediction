import numpy


def precision_at_k(y_truth,y_pred,k=5):
    assert y_truth.shape[0] == y_pred.shape[0]

    s = numpy.argsort(y_pred.view(numpy.ndarray),axis=1)[:,::-1]
    s = s[::,:k]
    it_counter = 0
    for i in xrange(y_truth.shape[0]):
        s_i = s[i]
        yt_i = y_truth[i]
        c = s_i[s_i == yt_i]
        if len(c) > 0:
            it_counter += 1
    return it_counter/float(y_truth.shape[0])


if __name__ == '__main__':
    pred = numpy.array([[23,3,1,41,5,1,2],[2,3,4,1,5,6,42],[2,32,4,1,2,4,5]])
    y_truth = numpy.array([0,6,1])
    print precision_at_k(y_truth,pred,k=5)
