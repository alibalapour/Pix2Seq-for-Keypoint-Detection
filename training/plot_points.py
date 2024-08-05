import cv2
import json
import pandas as pd
import os


def load_data_json(folder_path):
    data = json.load(open(os.path.join(folder_path, 'points.json')))['points']
    return data


def load_data_csv(folder_path):
    data = pd.read_csv(os.path.join(folder_path, 'annotations.csv'))
    return data


def plot_points_on_imgs_csv(labels_df, folder_path):
    os.makedirs(os.path.join(folder_path, 'labeled_imgs'), exist_ok=True)
    annot_dict = {}
    height_width_dict = {}
    img_dict = {}

    for i, img_labels in labels_df.iterrows():
        img_name = img_labels['id']
        img = cv2.imread(os.path.join(folder_path, 'images', img_name))
        img_dict[img_name] = img
        height_width_dict[img_name] = [img_labels['height'], img_labels['width']]
        try:
            annot_dict[img_name].append([img_labels['x'], img_labels['y']])
        except:
            annot_dict[img_name] = [[img_labels['x'], img_labels['y']]]

    for img_name, floor_point in annot_dict.items():
        previous_xy = None
        for xy in floor_point:
            cv2.circle(img_dict[img_name], (int(xy[0]), int(xy[1])), 3, (200, 0, 0), -1)
            if previous_xy is not None:
                cv2.line(img_dict[img_name], (int(xy[0]), int(xy[1])), (previous_xy[0], previous_xy[1]), (0, 255, 0),
                         thickness=1)
            previous_xy = [int(xy[0]), int(xy[1])]
            cv2.imwrite(os.path.join(folder_path, 'labeled_imgs', img_name), img_dict[img_name])


def plot_points_on_imgs_json(labels, folder_path):
    os.makedirs(os.path.join(folder_path, 'labeled_imgs'), exist_ok=True)
    ceiling_point_color = [0, 0, 255]
    floor_point_color = [0, 255, 0]
    for i, img_labels in enumerate(labels):
        img_name = img_labels['imageName'] + '.jpg'
        img = cv2.imread(os.path.join(folder_path, img_name))
        y, x = img.shape[0], img.shape[1]
        floor_points = img_labels['floorPoints']
        ceiling_points = img_labels['ceilingPoints']
        for floor_point in floor_points:
            x_region = [max(0, int(floor_point['x']) - 5), min(x, int(floor_point['x']) + 5)]
            y_region = [max(0, int(floor_point['y']) - 5), min(y, int(floor_point['y']) + 5)]
            img[y_region[0]:y_region[1], x_region[0]:x_region[1]] = floor_point_color
        for ceiling_point in ceiling_points:
            x_region = [max(0, int(ceiling_point['x']) - 5), min(x, int(ceiling_point['x']) + 5)]
            y_region = [max(0, int(ceiling_point['y']) - 5), min(y, int(ceiling_point['y']) + 5)]
            img[y_region[0]:y_region[1], x_region[0]:x_region[1]] = ceiling_point_color
        cv2.imwrite(os.path.join(folder_path, 'labeled_imgs', img_name), img)


if __name__ == '__main__':
    data_path = '../dataset/New folder/'
    data_type = 'csv'

    if data_type == 'csv':
        points = load_data_csv(data_path)
        plot_points_on_imgs_csv(points, data_path)
    elif data_type == 'json':
        points = load_data_json(data_path)
        plot_points_on_imgs_json(points, data_path)
