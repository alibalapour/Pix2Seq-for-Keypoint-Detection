import numpy as np
import torch
from itertools import groupby


def calculate_line_distances(points):
    """
    calculates the perimeter of generated shape of corners

    @param points: a list of xy coordinates
    @return: perimeter of corners
    """

    result = 0
    for i in range(len(points) - 1):
        cur_point = points[i]
        next_point = points[i + 1]
        result += np.linalg.norm(abs(cur_point - next_point))
    return result


def merge_point_for_metrics(pred_labels, pred_bboxes, true_labels, true_bboxes):
    """
    gets two sequences of corners and tries to make the number of them equal

    @param pred_labels: predicted labels
    @param pred_bboxes: predicted bboxes
    @param true_labels: true labels
    @param true_bboxes: true bboxes
    @return:
    """
    source_bboxes, source_labels, target_bboxes, target_labels = None, None, None, None

    # do nothing if number of predicted and true points are the same
    if len(pred_labels) == len(true_labels):
        return pred_labels, pred_bboxes, true_labels, true_bboxes

    # initiates source and target variables to make two sequences equal
    elif len(pred_labels) < len(true_labels):
        source_bboxes = true_bboxes
        source_labels = true_labels
        target_bboxes = pred_bboxes
        target_labels = pred_labels

    elif len(pred_labels) > len(true_labels):
        source_bboxes = pred_bboxes
        source_labels = pred_labels
        target_bboxes = true_bboxes
        target_labels = true_labels

    # main loop for equalizing two sequences
    for _ in range(abs(len(source_bboxes) - len(target_bboxes))):
        min_value = np.inf
        min_idx = -1
        for idx in range(len(source_bboxes) - 1):
            dist = np.linalg.norm(source_bboxes[idx + 1] - source_bboxes[idx])
            if dist < min_value:
                min_value = dist
                min_idx = idx
        new_point = (source_bboxes[min_idx] + source_bboxes[min_idx + 1]) / 2
        source_bboxes = np.delete(source_bboxes, min_idx + 1, axis=0)
        source_bboxes = np.delete(source_bboxes, min_idx, axis=0)
        source_bboxes = np.insert(source_bboxes, min_idx, new_point, axis=0)
        source_labels = np.delete(source_labels, min_idx)

    return source_labels, source_bboxes, target_labels, target_bboxes


def decode_predictions(tokens, tokenizer, number_of_corners):
    """
    decodes output of the model to generate bboxes and labels

    @param tokens: generated tokens from the model
    @param tokenizer: an object for tokenizing output of the model
    @param number_of_corners: true number of corners in an image
    @return: processed bboxes, labels, true masks, and predicted number of corners
    """
    # removes paddings
    mask = tokens != tokenizer.PAD_code
    tokens = tokens[mask]
    tokens = tokens[:-1]

    # finds EOS indices
    eos_idxs = np.where(tokens == tokenizer.EOS_code)
    if len(eos_idxs[0]) > 0:
        idx = eos_idxs[0][0]
        tokens = tokens[:idx]

    # find number of predicted points by finding indices of wrong dashes
    point_mask = (torch.where(tokens == tokenizer.DASH_code)[0] + 1) % 4 != 0
    wrong_dash_idxs = (torch.where(tokens == tokenizer.DASH_code)[0] + 1)[point_mask]
    if len(wrong_dash_idxs) != 0:
        number_of_predicted_points = (wrong_dash_idxs[0] / 4) + 1
    else:
        number_of_predicted_points = number_of_corners

    # splits tokens to an array of [x, y, cls]
    sep = tokenizer.DASH_code
    splitted_tokens = [list(items) for key, items in groupby(tokens, lambda x: x == sep) if not key]
    split_lengths = np.array([len(item) for item in splitted_tokens])
    correct_split_mask = split_lengths == 3
    # correct_split_indices = np.where(split_lengths == 3)[0]

    labels = []
    bboxes = []

    # return None if prediction has fewer tokens than true sequence
    if len(correct_split_mask) < number_of_corners:
        return None, None, None, None

    # prepares output bboxes and labels
    for index in range(len(splitted_tokens)):
        if correct_split_mask[index]:
            bboxes.append([int(splitted_tokens[index][0]), int(splitted_tokens[index][1])])
            labels.append(int(splitted_tokens[index][2]))
        else:
            bboxes.append([-1 / tokenizer.width, -1 / tokenizer.height])
            labels.append(-1 + tokenizer.num_bins)
    labels = np.array(labels) - tokenizer.num_bins
    bboxes = np.array(bboxes)
    bboxes = tokenizer.dequantize(bboxes)

    # recalculate width and heights
    if len(bboxes) != 0:
        bboxes[:, 0] = bboxes[:, 0] * tokenizer.width
        bboxes[:, 1] = bboxes[:, 1] * tokenizer.height

    return labels, bboxes, correct_split_mask[:number_of_corners], int(number_of_predicted_points)


def calculate_metric(predictions, trues, tokenizer, dataset_len, logging, normalize=False, img_size=384):
    """
    calculates custom metrics to evaluate corner detection results

    @param predictions: raw output of model
    @param trues: true labels of samples which are sequence in this format: [BOS, x1, y1, cls1, DASH, x1, y1, cls2, ..., EOS]
    @param tokenizer: an object for tokenizing raw output of model
    @param dataset_len: length of dataset
    @param logging: an object for creating logs
    @param normalize: normalize numbers of xy coordinates
    @param img_size: size of images
    @return: calculated metrics such as Accuracy of points with correct dimension, Accuracy of points with in range labels,
                Accuracy of labels classification,
                Regression loss of bounding boxes,
                Regression loss of unopt bounding boxes,
                Accuracy of points with in range xy value,
                Difference of distances of points, and
                Difference of number of predicted points
    """
    # initializes lists for saving metrics
    metric_1_list = []
    metric_2_list = []
    metric_3_list = []
    metric_4_list = []
    metric_4_unopt_list = []
    metric_5_list = []
    metric_6_list = []
    metric_7_list = []
    point_count = 0

    for prediction, true in zip(predictions, trues):
        # convert softmax values to token values
        prediction = prediction.float().argmax(dim=-1)

        # decode true bbox and labels
        true_labels, true_bboxes = tokenizer.decode(true)

        # decode pred bbox and labels
        original_pred_labels, original_pred_bboxes, correct_split_mask, number_of_predicted_points = decode_predictions(
            prediction,
            tokenizer,
            len(true_labels))

        if original_pred_labels is None:
            continue

        # normalize values of bboxes
        if normalize:
            true_bboxes /= img_size
            original_pred_bboxes /= img_size

        # merge bboxes of closed points
        pred_labels, pred_bboxes, true_labels, true_bboxes = merge_point_for_metrics(original_pred_labels,
                                                                                     original_pred_bboxes,
                                                                                     true_labels,
                                                                                     true_bboxes)

        # initializes metric values
        metric_1 = sum(correct_split_mask)
        metric_2 = 0
        metric_3 = 0
        metric_4 = 0
        metric_4_unopt = 0
        metric_5 = 0
        metric_6 = 0
        metric_7 = np.abs(len(true_labels) - number_of_predicted_points)

        # calculates metrics
        if metric_1 != 0:
            pred_class_mask = pred_labels[correct_split_mask] >= 0
            metric_2 = sum(pred_class_mask)
            if metric_2 != 0:
                metric_3 = np.sum(pred_labels[correct_split_mask][pred_class_mask] == true_labels[correct_split_mask][
                    pred_class_mask])
                metric_4 = np.linalg.norm(
                    pred_bboxes[correct_split_mask][pred_class_mask] - true_bboxes[correct_split_mask][pred_class_mask])
                metric_4_unopt = np.linalg.norm(
                    original_pred_bboxes[:len(true_bboxes)][correct_split_mask][pred_class_mask] -
                    true_bboxes[correct_split_mask][
                        pred_class_mask])
                metric_5_temp = np.sum(np.logical_and(pred_bboxes[correct_split_mask][pred_class_mask] >= 0,
                                                      pred_bboxes[correct_split_mask][
                                                          pred_class_mask] < tokenizer.num_bins), axis=1)
                metric_5 = np.sum(metric_5_temp == 2)

                true_distance = calculate_line_distances(true_bboxes[correct_split_mask])
                pred_distance = calculate_line_distances(pred_bboxes[correct_split_mask])
                metric_6 = abs(true_distance - pred_distance)

        # calculates true number of points
        point_count += len(true_labels)

        # appends metrics to corresponding lists
        metric_1_list.append(metric_1)
        metric_2_list.append(metric_2)
        metric_3_list.append(metric_3)
        metric_4_list.append(metric_4)
        metric_4_unopt_list.append(metric_4_unopt)
        metric_5_list.append(metric_5)
        metric_6_list.append(metric_6)
        metric_7_list.append(metric_7)

    # finalizes metrics
    metrics = [
        round((np.sum(metric_1_list) / point_count) * 100, 2),
        round((np.sum(metric_2_list) / point_count) * 100, 3),
        round((np.sum(metric_3_list) / point_count) * 100, 3),
        round((np.sum(metric_4_list) / point_count), 5),
        round((np.sum(metric_4_unopt_list) / point_count), 5),
        round((np.sum(metric_5_list) / point_count) * 100, 2),
        round(np.sum(metric_6_list) / dataset_len, 5),
        round(np.sum(metric_7_list) / dataset_len, 3)
    ]

    # displays metrics
    logging.warning(' ')
    logging.warning('-' * 100)
    logging.warning("Accuracy of points with correct dimension = " + str(metrics[0]) + ' %')
    logging.warning("Accuracy of points with in range labels   = " + str(metrics[1]) + ' %')
    logging.warning("Accuracy of labels classification         = " + str(metrics[2]) + ' %')
    logging.warning("Regression loss of bounding boxes         = " + str(metrics[3]))
    logging.warning("Regression loss of unopt bounding boxes   = " + str(metrics[4]))
    logging.warning("Accuracy of points with in range xy value = " + str(metrics[5]) + ' %')
    logging.warning("Difference of distances of points         = " + str(metrics[6]))
    logging.warning("Difference of number of predicted points  = " + str(metrics[7]))
    logging.warning('-' * 100)

    return metrics
