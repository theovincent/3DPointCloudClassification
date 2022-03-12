import numpy as np

CENTERS = {
    "MiniLille1": np.array(
        [
            [-28.62239, -23.31253],
            [-26.703415, -25.464874],
            [32.341904, 22.385834],
            [30.010391, 24.857819],
        ],
        dtype=np.float32,
    )
}


ROTATIONS = {"MiniLille1": 0.6877497}


Z_GROUNDS = {
    "MiniLille1": -2.774031,
}

Z_GROUND = -1.703319
MAX_DISTANCE = 48.8722
MAX_HEIGHT = 1.8629856
MIN_DISTANCE = 3.6060606
N_POINTS = 122261

KITTI_NUMBERS_TO_LABELS = {
    0: "unlabeled",
    1: "outlier",
    10: "car",
    11: "bicycle",
    13: "bus",
    15: "motorcycle",
    16: "on-rails",
    18: "truck",
    20: "other-vehicle",
    30: "person",
    31: "bicyclist",
    32: "motorcyclist",
    40: "road",
    44: "parking",
    48: "sidewalk",
    49: "other-ground",
    50: "building",
    51: "fence",
    52: "other-structure",
    60: "lane-marking",
    70: "vegetation",
    71: "trunk",
    72: "terrain",
    80: "pole",
    81: "traffic-sign",
    99: "other-object",
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}

CITY_NUMBERS_TO_LABELS = {
    0: "Unclassified",
    1: "Ground",
    2: "Building",
    3: "Poles",
    4: "Pedestrians",
    5: "Cars",
    6: "Vegetation",
}

KITTI_TO_CITY_LABELS = {
    "unlabeled": "Unclassified",
    "outlier": "Unclassified",
    "car": "Cars",
    "bicycle": "Unclassified",
    "bus": "Cars",
    "motorcycle": "Unclassified",
    "on-rails": "Unclassified",
    "truck": "Cars",
    "other-vehicle": "Cars",
    "person": "Pedestrians",
    "bicyclist": "Pedestrians",
    "motorcyclist": "Pedestrians",
    "road": "Ground",
    "parking": "Ground",
    "sidewalk": "Ground",
    "other-ground": "Ground",
    "building": "Building",
    "fence": "Unclassified",
    "other-structure": "Unclassified",
    "lane-marking": "Ground",
    "vegetation": "Vegetation",
    "trunk": "Cars",
    "terrain": "Ground",
    "pole": "Poles",
    "traffic-sign": "Unclassified",
    "other-object": "Unclassified",
    "moving-car": "Cars",
    "moving-bicyclist": "Pedestrians",
    "moving-person": "Pedestrians",
    "moving-motorcyclist": "Pedestrians",
    "moving-on-rails": "Unclassified",
    "moving-bus": "Cars",
    "moving-truck": "Cars",
    "moving-other-vehicle": "Cars",
}

KITTI_LABELS_TO_NUMBERS = {v: k for k, v in KITTI_NUMBERS_TO_LABELS.items()}
CITY_LABELS_TO_NUMBERS = {v: k for k, v in CITY_NUMBERS_TO_LABELS.items()}

KITTI_TO_CITY_NUMBERS = {}

for kitti_label, city_label in KITTI_TO_CITY_LABELS.items():
    KITTI_TO_CITY_NUMBERS[KITTI_LABELS_TO_NUMBERS[kitti_label]] = CITY_LABELS_TO_NUMBERS[city_label]