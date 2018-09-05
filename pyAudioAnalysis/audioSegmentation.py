from __future__ import print_function
import numpy
import os
from pyAudioAnalysis import audioFeatureExtraction as aF
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import audioBasicIO
from scipy.spatial import distance
import os.path

""" General utility functions """


def smoothMovingAvg(inputSignal, windowLen=11):
    windowLen = int(windowLen)
    if inputSignal.ndim != 1:
        raise ValueError("")
    if inputSignal.size < windowLen:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLen < 3:
        return inputSignal
    s = numpy.r_[2*inputSignal[0] - inputSignal[windowLen-1::-1],
                 inputSignal, 2*inputSignal[-1]-inputSignal[-1:-windowLen:-1]]
    w = numpy.ones(windowLen, 'd')
    y = numpy.convolve(w/w.sum(), s, mode='same')
    return y[windowLen:-windowLen+1]


def selfSimilarityMatrix(featureVectors):
    '''
    This function computes the self-similarity matrix for a sequence
    of feature vectors.
    ARGUMENTS:
     - featureVectors:     a numpy matrix (nDims x nVectors) whose i-th column
                           corresponds to the i-th feature vector

    RETURNS:
     - S:                  the self-similarity matrix (nVectors x nVectors)
    '''

    [nDims, nVectors] = featureVectors.shape
    [featureVectors2, MEAN, STD] = aT.normalizeFeatures([featureVectors.T])
    featureVectors2 = featureVectors2[0].T
    S = 1.0 - distance.squareform(distance.pdist(featureVectors2.T, 'cosine'))
    return S


def flags2segs(flags, window):
    '''
    ARGUMENTS:
     - flags:      a sequence of class flags (per time window)
     - window:     window duration (in seconds)

    RETURNS:
     - segs:       a sequence of segment's limits: segs[i,0] is start and
                   segs[i,1] are start and end point of segment i
     - classes:    a sequence of class flags: class[i] is the class ID of
                   the i-th segment
    '''

    preFlag = 0
    cur_flag = 0
    n_segs = 0

    cur_val = flags[cur_flag]
    segsList = []
    classes = []
    while (cur_flag < len(flags) - 1):
        stop = 0
        preFlag = cur_flag
        preVal = cur_val
        while (stop == 0):
            cur_flag = cur_flag + 1
            tempVal = flags[cur_flag]
            if ((tempVal != cur_val) | (cur_flag == len(flags) - 1)):  # stop
                n_segs = n_segs + 1
                stop = 1
                cur_seg = cur_val
                cur_val = flags[cur_flag]
                segsList.append((cur_flag * window))
                classes.append(preVal)
    segs = numpy.zeros((len(segsList), 2))

    for i in range(len(segsList)):
        if i > 0:
            segs[i, 0] = segsList[i-1]
        segs[i, 1] = segsList[i]
    return (segs, classes)


def segs2flags(seg_start, seg_end, seg_label, win_size):
    '''
    This function converts segment endpoints and respective segment
    labels to fix-sized class labels.
    ARGUMENTS:
     - seg_start:    segment start points (in seconds)
     - seg_end:    segment endpoints (in seconds)
     - seg_label:    segment labels
      - win_size:    fix-sized window (in seconds)
    RETURNS:
     - flags:    numpy array of class indices
     - class_names:    list of classnames (strings)
    '''
    flags = []
    class_names = list(set(seg_label))
    curPos = win_size / 2.0
    while curPos < seg_end[-1]:
        for i in range(len(seg_start)):
            if curPos > seg_start[i] and curPos <= seg_end[i]:
                break
        flags.append(class_names.index(seg_label[i]))
        curPos += win_size
    return numpy.array(flags), class_names

def mtFileClassification(input_file, model_name, model_type,
                         plot_results=False, gt_file=""):
    '''
    This function performs mid-term classification of an audio stream.
    Towards this end, supervised knowledge is used, i.e. a pre-trained classifier.
    ARGUMENTS:
        - input_file:        path of the input WAV file
        - model_name:        name of the classification model
        - model_type:        svm or knn depending on the classifier type
        - plot_results:      True if results are to be plotted using
                             matplotlib along with a set of statistics

    RETURNS:
          - segs:           a sequence of segment's endpoints: segs[i] is the
                            endpoint of the i-th segment (in seconds)
          - classes:        a sequence of class flags: class[i] is the
                            class ID of the i-th segment
    '''

    if not os.path.isfile(model_name):
        print("mtFileClassificationError: input model_type not found!")
        return (-1, -1, -1, -1)
    # Load classifier:
    if model_type == "knn":
        [classifier, MEAN, STD, class_names, mt_win, mt_step, st_win, st_step, compute_beat] = \
            aT.load_model_knn(model_name)

    if compute_beat:
        print("Model " + model_name + " contains long-term music features "
                                     "(beat etc) and cannot be used in "
                                     "segmentation")
        return (-1, -1, -1, -1)
    [fs, x] = audioBasicIO.readAudioFile(input_file)       # load input file
    if fs == -1:                                           # could not read file
        return (-1, -1, -1, -1)
    x = audioBasicIO.stereo2mono(x)                        # convert stereo (if) to mono
    duration = len(x) / fs
    # mid-term feature extraction:
    [mt_feats, _, _] = aF.mtFeatureExtraction(x, fs, mt_win * fs,
                                                     mt_step * fs,
                                                     round(fs * st_win),
                                                     round(fs * st_step))
    flags = []
    Ps = []
    flags_ind = []
    for i in range(mt_feats.shape[1]):              # for each feature vector (i.e. for each fix-sized segment):
        cur_fv = (mt_feats[:, i] - MEAN) / STD       # normalize current feature vector
        [res, P] = aT.classifierWrapper(classifier, model_type, cur_fv)    # classify vector
        flags_ind.append(res)
        flags.append(class_names[int(res)])              # update class label matrix
        Ps.append(numpy.max(P))                          # update probability matrix
    flags_ind = numpy.array(flags_ind)

    # 1-window smoothing
    for i in range(1, len(flags_ind) - 1):
        if flags_ind[i-1] == flags_ind[i + 1]:
            flags_ind[i] = flags_ind[i + 1]
    # convert fix-sized flags to segments and classes
    (segs, classes) = flags2segs(flags, mt_step)
    segs[-1] = len(x) / float(fs)
    acc = 0
    cm = []
    flags_ind_gt = numpy.array([])
    return (flags_ind, class_names, acc, cm)

def silenceRemoval(x, fs, st_win, st_step, smoothWindow=0.5, weight=0.5, plot=False):
    '''
    Event Detection (silence removal)
    ARGUMENTS:
         - x:                the input audio signal
         - fs:               sampling freq
         - st_win, st_step:    window size and step in seconds
         - smoothWindow:     (optinal) smooth window (in seconds)
         - weight:           (optinal) weight factor (0 < weight < 1) the higher, the more strict
         - plot:             (optinal) True if results are to be plotted
    RETURNS:
         - seg_limits:    list of segment limits in seconds (e.g [[0.1, 0.9], [1.4, 3.0]] means that
                    the resulting segments are (0.1 - 0.9) seconds and (1.4, 3.0) seconds
    '''

    if weight >= 1:
        weight = 0.99
    if weight <= 0:
        weight = 0.01

    # Step 1: feature extraction
    x = audioBasicIO.stereo2mono(x)
    st_feats, _ = aF.stFeatureExtraction(x, fs, st_win * fs, 
                                                  st_step * fs)

    # Step 2: train binary svm classifier of low vs high energy frames
    # keep only the energy short-term sequence (2nd feature)
    st_energy = st_feats[1, :]
    en = numpy.sort(st_energy)
    # number of 10% of the total short-term windows
    l1 = int(len(en) / 10)
    # compute "lower" 10% energy threshold
    t1 = numpy.mean(en[0:l1]) + 0.000000000000001
    # compute "higher" 10% energy threshold
    t2 = numpy.mean(en[-l1:-1]) + 0.000000000000001
    # get all features that correspond to low energy
    class1 = st_feats[:, numpy.where(st_energy <= t1)[0]]
    # get all features that correspond to high energy
    class2 = st_feats[:, numpy.where(st_energy >= t2)[0]]
    # form the binary classification task and ...
    faets_s = [class1.T, class2.T]
    # normalize and train the respective svm probabilistic model
    # (ONSET vs SILENCE)
    [faets_s_norm, means_s, stds_s] = aT.normalizeFeatures(faets_s)
    svm = aT.trainSVM(faets_s_norm, 1.0)

    # Step 3: compute onset probability based on the trained svm
    prob_on_set = []
    for i in range(st_feats.shape[1]):
        # for each frame
        cur_fv = (st_feats[:, i] - means_s) / stds_s
        # get svm probability (that it belongs to the ONSET class)
        prob_on_set.append(svm.predict_proba(cur_fv.reshape(1,-1))[0][1])
    prob_on_set = numpy.array(prob_on_set)
    # smooth probability:
    prob_on_set = smoothMovingAvg(prob_on_set, smoothWindow / st_step)

    # Step 4A: detect onset frame indices:
    prog_on_set_sort = numpy.sort(prob_on_set)
    # find probability Threshold as a weighted average
    # of top 10% and lower 10% of the values
    Nt = int(prog_on_set_sort.shape[0] / 10)
    T = (numpy.mean((1 - weight) * prog_on_set_sort[0:Nt]) +
         weight * numpy.mean(prog_on_set_sort[-Nt::]))

    max_idx = numpy.where(prob_on_set > T)[0]
    # get the indices of the frames that satisfy the thresholding
    i = 0
    time_clusters = []
    seg_limits = []

    # Step 4B: group frame indices to onset segments
    while i < len(max_idx):
        # for each of the detected onset indices
        cur_cluster = [max_idx[i]]
        if i == len(max_idx)-1:
            break
        while max_idx[i+1] - cur_cluster[-1] <= 2:
            cur_cluster.append(max_idx[i+1])
            i += 1
            if i == len(max_idx)-1:
                break
        i += 1
        time_clusters.append(cur_cluster)
        seg_limits.append([cur_cluster[0] * st_step,
                           cur_cluster[-1] * st_step])

    # Step 5: Post process: remove very small segments:
    min_dur = 0.2
    seg_limits_2 = []
    for s in seg_limits:
        if s[1] - s[0] > min_dur:
            seg_limits_2.append(s)
    seg_limits = seg_limits_2


    return seg_limits



