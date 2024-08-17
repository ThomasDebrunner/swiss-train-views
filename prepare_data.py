from json import JSONEncoder

import rasterio
import numpy as np
import json
import pickle
import os
import math
from tqdm import tqdm


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return obj.item()  # Convert to Python native types
        return JSONEncoder.default(self, obj)


def horizon_distance(height):
    # Earth's radius in kilometers
    R = 6371000
    return math.sqrt(2 * R * height)


def lv03_to_rowcol(transform, east, north):
    # Convert lat/lon to the coordinate system of the raster
    col, row = ~transform * (east, north)
    return int(row), int(col)


def get_elevation(model, east, north):
    data, transform = model
    row, col = lv03_to_rowcol(transform, east, north)
    elevation = data[row, col]
    return elevation


def read_eleveation_data():
    with rasterio.open('data/ASCII_GRID_1part/dhm25_grid_raster.asc') as dataset:
        transform = dataset.transform
        # Read the elevation data

        if os.path.exists('elevation_data.pkl'):
            print('Reading from cache...')
            with open('elevation_data.pkl', 'rb') as f:
                elevation_data = pickle.load(f)
        else:
            print('Reading from source...')
            elevation_data = dataset.read(1)
            with open('elevation_data.pkl', 'wb') as f:
                pickle.dump(elevation_data, f)

        return elevation_data, transform


def read_train_data():
    with open('data/linienkilometrierung.json', 'r') as f:
        points = json.load(f)

    print(f"Have {len(points)} train points")
    lines_raw = {}
    for point in points:
        if point['linienr'] not in lines_raw:
            lines_raw[point['linienr']] = []
        lines_raw[point['linienr']].append(point)
    print(f"Have {len(lines_raw)} lines")

    lines = []
    for line in lines_raw.values():
        # sort all points by km
        sorted_points = sorted(line, key=lambda x: x['linienr_km'])
        line_name = sorted_points[0]['liniename']
        line_nr = sorted_points[0]['linienr']
        point_coords = np.array([(s['y'] - 2000_000, s['x'] - 1000_000) for s in sorted_points])
        diff_vects = np.diff(point_coords, axis=0)
        # discard the last point, as it has no diff vector
        point_coords = point_coords[0:-1, :]
        # normalize direction vectors
        distances = np.linalg.norm(diff_vects, axis=1)
        normalized_diff_vects = diff_vects / distances[:, None]
        lines.append({
            'name': line_name,
            'nr': line_nr,
            'points': point_coords,
            'vectors': normalized_diff_vects,
        })

    return lines


def seek_terrain_collision(model, point, altitude, vector, viewer_height):
    # vector is in direction of travel
    left_vector = np.array([vector[1], -vector[0]])
    right_vector = -left_vector
    view_altitude = altitude + viewer_height
    horizon = horizon_distance(viewer_height)

    left_distance = horizon
    right_distance = horizon
    # go left
    for distance in range(0, int(horizon), 10):
        left_point = point + left_vector * distance
        left_elevation = get_elevation(model, left_point[0], left_point[1])
        if left_elevation > view_altitude:
            left_distance = distance
            break

    # go right
    for distance in range(0, int(horizon), 5):
        right_point = point + right_vector * distance
        right_elevation = get_elevation(model, right_point[0], right_point[1])
        if right_elevation > view_altitude:
            right_distance = distance
            break

    return left_distance, right_distance


def analyze_altitudes(model, train_data, viewer_height=2):
    for line in tqdm(train_data):
        points = line['points']
        vectors = line['vectors']
        elevations = []
        left_distances = []
        right_distances = []
        for point, vector in zip(points, vectors):
            east, north = point
            elevation = get_elevation(model, east, north)
            elevations.append(elevation)
            left, right = seek_terrain_collision(model, point, elevation, vector, viewer_height)
            left_distances.append(left)
            right_distances.append(right)

        line['elevations'] = elevations
        line['left_distances'] = left_distances
        line['right_distances'] = right_distances

    return train_data


def remove_tunnels(train_data, squared_filtered_threshold=30):
    for line in train_data:
        elevations = line['elevations']
        if len(elevations) < 20:
            # skip short lines, no meaningful filtering possible
            continue

        left_distances = line['left_distances']
        right_distances = line['right_distances']
        # squared elevation differential
        diffs = (np.diff(elevations)) ** 2
        # running average filter
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        diffs_filtered = np.convolve(diffs, kernel, mode='same')

        # where the filtered diffs are below the threshold, we have a tunnel
        # set the distances to 0
        for i, diff in enumerate(diffs_filtered.tolist()):
            if diff > squared_filtered_threshold:
                left_distances[i] = 0
                right_distances[i] = 0

def main():
    east = 599977.858
    north = 198954.5319999999

    model = read_eleveation_data()
    elevation = get_elevation(model, east, north)
    print(f"Elevation at {east}, {north}: {elevation} m")

    train_data = read_train_data()
    # add altitudes to train data
    train_data = analyze_altitudes(model, train_data)

    remove_tunnels(train_data)

    # store as json
    with open('train_data.json', 'w') as f:
        json.dump(train_data, f, cls=NumpyArrayEncoder)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

