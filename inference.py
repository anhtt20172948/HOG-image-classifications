from utils.nms import non_maximum_suppression
import cv2
from utils.utils import *
from skimage.transform import pyramid_gaussian
from feature_extractor.configs import *
from models.SVM import SVM
from feature_extractor.hog import HOG_feature_extractor
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--SVM',  action='store_true', default=True, help='Linear SVM')
    parser.add_argument('--hog', action='store_true', default=True, help='HOG feature extractor')
    parser.add_argument('--weight', type=str, default='checkpoints/SVM/model.pkl', help='path to checkpoint')
    parser.add_argument('--image', type=str, default='images/dogs.jpg', help='path to single image')
    parser.add_argument('--input-size', type=int, default=32)
    parser.add_argument('--nms', type=float, default=0.4)
    parser.add_argument('--conf', type=float, default=0.5)
    parser.add_argument('--downscale', type=float, default=1.25)
    opt = parser.parse_args()
    # load model
    if opt.SVM:
        clf = SVM(opt.weight)

    # init feature extractor
    feature_extractor = HOG_feature_extractor(HOG_CONFIG)
    MIN_WINDOW_SIZE = HOG_CONFIG['min_window_size']
    STEP_SIZE = HOG_CONFIG['step_size']
    ORIENTATIONS = HOG_CONFIG['orientations']
    PIXELS_PER_CELL = HOG_CONFIG['pixel_per_cell']
    CELL_PER_BLOCK = HOG_CONFIG['cell_per_block']

    # hyper parameter
    downscale = opt.downscale
    conf = opt.conf
    nms_threshold = opt.nms
    input_size = opt.input_size

    # read image
    im = cv2.imread(opt.image)

    ori_img = im
    detections = []
    detections_v1 = []
    confidence_score = []
    scale = 0
    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        cd = []
        if im_scaled.shape[0] < MIN_WINDOW_SIZE[1] or im_scaled.shape[1] < MIN_WINDOW_SIZE[0]:
            break

        for (x, y, im_window) in sliding_window(im_scaled, MIN_WINDOW_SIZE, STEP_SIZE):
            if im_window.shape[0] != MIN_WINDOW_SIZE[1] or im_window.shape[1] != MIN_WINDOW_SIZE[0]:
                continue
            raw_im_window = im_window
            im_window = cv2.resize(im_window, (input_size, input_size))
            fd = feature_extractor.get_feature(im_window)
            pred, confidences = clf.inference([fd])
            # hiện tại mới nhận diện chó
            if pred[0] == 5:
                if confidences[0] < 0:
                    continue
                print("Scale ->  {} | Confidence Score {} \n".format(scale, confidences[0]))
                detections.append((x, y, confidences[0],
                                   int(MIN_WINDOW_SIZE[0] * (downscale ** scale)),
                                   int(MIN_WINDOW_SIZE[1] * (downscale ** scale))))
                detections_v1.append((x, y,
                                      x + int(MIN_WINDOW_SIZE[0] * (downscale ** scale)),
                                      y + int(MIN_WINDOW_SIZE[1] * (downscale ** scale))))
                confidence_score.append(round(confidences[0], 3))
                cd.append(detections[-1])
            if True:
                clone = im_scaled.copy()
                for x1, y1, _, _, _ in cd:
                    # Draw the detections at this scale
                    cv2.rectangle(clone, (x1, y1), (x1 + raw_im_window.shape[1], y1 +
                                                    raw_im_window.shape[0]), (0, 0, 0), thickness=2)
                cv2.rectangle(clone, (x, y), (x + raw_im_window.shape[1], y +
                                              raw_im_window.shape[0]), (0, 0, 0), thickness=2)
                cv2.imshow("Sliding Window in Progress", clone)
                cv2.waitKey(10)
        scale += 1

    # clone = im.copy()
    # for (x_tl, y_tl, _, w, h) in detections:
    #     # Draw the detections
    #     cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
    # cv2.imshow("Raw Detections before NMS", im)
    # cv2.waitKey()

    # Perform Non Maxima Suppression

    # detections = nms(detections, 0.99)
    picked_boxes, picked_score = non_maximum_suppression(detections_v1, confidence_score, 0.4)
    # Draw parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
        (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
        cv2.rectangle(ori_img, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
        cv2.rectangle(ori_img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
        cv2.putText(ori_img, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)
    # Display the results after performing NMS
    # for (x_tl, y_tl, _, w, h) in detections:
    #     # Draw the detections
    #     cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 0), thickness=2)
    cv2.imshow("Final Detections after applying NMS", ori_img)
    cv2.waitKey()
