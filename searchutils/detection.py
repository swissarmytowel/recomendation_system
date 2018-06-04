import warnings
import cv2 as cv
import numpy as np
from skimage.future import graph
from skimage import segmentation, color, io, filters
from searchutils.indices import ImageIndex, IndexFeature

# SKIMAGE WARNINGS DISABLING
warnings.filterwarnings("ignore")


def perform_canny_edge_detection(image):
    """
    Canny edge detection and estimated ROI bounding box calculation
    :param image: image of BGR values
    :return: grayscale image, top left bbox point, bottom right bbox point
    """
    image_copy = image.copy()
    edges_image = cv.Canny(image=image_copy, threshold1=0, threshold2=image_copy.flatten().mean(), apertureSize=3,
                           L2gradient=True)
    y, x = edges_image.nonzero()
    top_left_point = x.min(), y.min()
    bottom_right_point = x.max(), y.max()
    return edges_image, top_left_point, bottom_right_point


def perform_grabcut_segmentation(image, top_left_point, bottom_right_point, iterations=2):
    """
    Perform GrabCut foreground extraction algorithm to get object inside bounding box, annotated by rectangle
    :param image: image in BGR format
    :param top_left_point: top left bbox point
    :param bottom_right_point: bottom right bbox point
    :return: image with extracted object
    """
    resulting_image = image.copy()
    mask = np.zeros(image.shape[:2], np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    bounding_rectangle = (top_left_point[0], top_left_point[1], bottom_right_point[0], bottom_right_point[1])

    cv.grabCut(img=resulting_image.copy(), mask=mask, rect=bounding_rectangle, bgdModel=background, fgdModel=foreground,
               iterCount=iterations, mode=cv.GC_INIT_WITH_RECT)
    filtered_mask = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype(np.uint8)
    resulting_image[filtered_mask == 0] = [255, 255, 255]
    return resulting_image


def find_frontal_face(image, classifier_path, scale=1.1, min_neighbours=5):
    """
    Find frontal (maximum area) face on image
    :param image: image in BGR format
    :param classifier_path: path of pre-trained cascade classifier xml
    :param scale: detection parameter
    :param min_neighbours: detection parameter
    :return: frontal face pixels in BGR format
    """
    image_copy = image.copy()
    grayscale_copy = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
    cascade_classifier = cv.CascadeClassifier(classifier_path)
    encountered_faces_bounding_boxes = cascade_classifier.detectMultiScale(grayscale_copy, scaleFactor=scale,
                                                                           minNeighbors=min_neighbours)
    frontal_face = None
    frontal_face_bbox = None
    if len(encountered_faces_bounding_boxes) > 0:
        faces = []
        for (x, y, w, h) in encountered_faces_bounding_boxes:
            faces.append(image_copy[y:y + h, x:x + w])
        face_sizes = np.asarray([face.size for face in faces])
        frontal_face = faces[face_sizes.argmax()]
        frontal_face_bbox = encountered_faces_bounding_boxes[face_sizes.argmax()]
    return frontal_face, frontal_face_bbox


def get_subtraction_masks_for_skin_and_hair(image, face, segments=2):
    data = face.copy().reshape((-1, 3)).astype(np.float32)
    face_hsv = cv.cvtColor(face.copy(), cv.COLOR_BGR2HSV)
    image_hsv = cv.cvtColor(image.copy(), cv.COLOR_BGR2HSV)

    ret, labels, center = cv.kmeans(data, segments, None, (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    10, cv.KMEANS_PP_CENTERS)
    center = center.astype(np.uint8)
    result = center[labels.flatten()]
    reshaped_result = result.reshape(face.shape)

    skin_label = 1 if np.sum(labels) > labels.size - np.sum(labels) else 0
    hair_label = 1 - skin_label

    skin_mean, skin_std = cv.meanStdDev(face_hsv,
                                        mask=cv.inRange(reshaped_result, center[skin_label], center[skin_label]))
    hair_mean, hair_std = cv.meanStdDev(face_hsv,
                                        mask=cv.inRange(reshaped_result, center[hair_label], center[hair_label]))

    skin_min_thresh = skin_mean - (skin_std * 2)
    skin_max_thresh = skin_mean + (skin_std * 2)
    skin_min_thresh[2] = 0
    skin_max_thresh[2] = 255

    hair_min_thresh = hair_mean - (hair_std * 2)
    hair_max_thresh = hair_mean + (hair_std * 2)

    skin_mask = cv.inRange(image_hsv, skin_min_thresh, skin_max_thresh)
    hair_mask = cv.inRange(image_hsv, hair_min_thresh, hair_max_thresh)

    return cv.morphologyEx(skin_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8)),\
           cv.morphologyEx(hair_mask, cv.MORPH_OPEN, np.ones((3, 3), np.uint8))


def superpixel_extraction(image_i):
    """
    Extracts regions from image and refines segmentation with graph cut
    :param image_i: input image of shape (N, M, 3)
    :return: label matrix of shape (N, M, 1)
    """
    image = image_i.copy()
    image = image[:, :, ::-1]
    edges, top_left, bottom_right = perform_canny_edge_detection(image=image)
    grabbed_object = perform_grabcut_segmentation(image=image, top_left_point=top_left,
                                                  bottom_right_point=bottom_right)

    frontal_face, frontal_face_bbox = find_frontal_face(image=image, classifier_path='data/face.xml')
    if frontal_face is None:
        frontal_face, frontal_face_bbox = find_frontal_face(image=image,
                                                            classifier_path='data/profile.xml')
    if frontal_face is not None:
        skin_mask, hair_mask = get_subtraction_masks_for_skin_and_hair(image=image, face=frontal_face)
        result = grabbed_object.copy()
        result[skin_mask == 255] = [255, 255, 255]
        result[hair_mask == 255] = [255, 255, 255]
        x, y, w, h = frontal_face_bbox
        top_left = list(top_left)
        top_left[1] += h
        result = perform_grabcut_segmentation(image=result.copy(), top_left_point=top_left,
                                              bottom_right_point=bottom_right, iterations=1)
    else:
        result = grabbed_object.copy()
    result = result[:, :, ::-1]

    labels = segmentation.slic(image=result, n_segments=200, convert2lab=True, max_iter=100, min_size_factor=0.01,
                               max_size_factor=3, compactness=100)
    # labels = segmentation.quickshift(image, ratio=1, sigma=0.8, max_dist=20, convert2lab=True, kernel_size=10)
    rag = graph.rag_mean_color(image=color.rgb2lab(result), labels=labels)
    return graph.cut_threshold(labels=labels, rag=rag, thresh=3.5)


def index_image(image, image_name, labels, convert_to_lab=True):
    """
    Extracts feature set from intensity image and labels
    Feature set consists of mean value and variance (standard deviation)
    :param image_name: image name
    :param convert_to_lab: should convert to CIE L*a*b*
    :param image: intensity image of shape (N, M, 3)
    :param labels: labels of shape (N, M).
    :return: FeatureSet object of IndexFeature
    """
    if convert_to_lab:
        try:
            image = color.rgb2lab(image)
        except MemoryError:
            print(image_name)
    return ImageIndex(image=image, labels=labels, image_name=image_name)