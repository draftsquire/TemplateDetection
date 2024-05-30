import cv2
import cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

def extract_features(img, feature_extractor=None):
    '''
    Extract features using SIFT by default
    :param feature_extractor: python object of feature extractor class (e.g. SIFT)
    :param img: original image
    :return: kp1, des1 - keypoints and descriptors for each keypoint respectively
    '''
    if feature_extractor is None:
        feature_extractor = cv2.SIFT_create()
    kp1, des1 = feature_extractor.detectAndCompute(img, None)

    return kp1, des1


def get_matches(des1, des2):
    '''
    Get matches between two descriptor lists using k-nearest neighbour matcher and ratio test as per Lowe's paper
    :param des1: descriptors of 1st image
    :param des2: descriptors of 2nd image
    :return: "good" matches, tuple of matches in respect for each descriptor (for debug purposes)
    '''
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(des1, des2, k=2)
    good_points = []
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_points.append(m)
            good_matches.append((m, n))
    return good_points, good_matches

def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    list_of_bboxes = [(0, 0, 1, 1), ]
    return list_of_bboxes


if __name__ == '__main__':
    template_paths = ['train/template_0_0.jpg', 'train/template_0_1.jpg', 'train/template_1.jpg', 'train/template_2.jpg', 'train/template_3.jpg']
    train_paths = ['train/train_0.jpg', 'train/train_1.jpg', 'train/train_2.jpg', 'train/train_3.jpg', 'train/train_extreme.jpg',]
    # initialized a list of images
    templates = []
    trains = []
    for i in range(len(template_paths)):
        templates.append(cv2.imread(template_paths[i]))
        templates[i] = cv2.cvtColor(templates[i], cv2.COLOR_BGR2RGB)

    for i in range(len(train_paths)):
        trains.append(cv2.imread(train_paths[i]))
        trains[i] = cv2.cvtColor(trains[i], cv2.COLOR_BGR2RGB)

    kp_temp, des_temp = extract_features(templates[0])

    kp_train, des_train = extract_features(trains[0])
    # -----------------
    x = np.array([kp_train[0].pt])

    for i in range(1, len(kp_train)):
        x = np.append(x, [kp_train[i].pt], axis=0)
    bandwidth = estimate_bandwidth(x, quantile=0.1, n_samples=500)

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True)
    ms.fit(x)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("number of estimated clusters : %d" % n_clusters_)

    s = [None] * n_clusters_
    for i in range(n_clusters_):
        l = ms.labels_
        d, = np.where(l == i)
        print(d.__len__())
        s[i] = list(kp_train[xx] for xx in d)

    des2_ = des_train
    # -----------------

    good_points, good_matches = get_matches(des_temp, des_train)

    # fig, axes = plt.subplots(1, 1)
    fig = plt.figure()
    fig.canvas.manager.set_window_title("")
    fig.suptitle("", fontsize=20)
    plt.imshow(
        cv2.drawMatchesKnn(templates[0], kp_temp, trains[0], kp_train, good_matches, None, flags=2),
        interpolation="none", norm=None, filternorm=False)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="x", color="red", s=200)

    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
