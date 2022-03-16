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
    ),
    "MiniLille2_split1": np.array(
        [
            [5.730957, 5.312561],
            [9.944702, 3.635559],
            [27.381958, 42.80591],
            [23.06903, 44.494507],
        ],
        dtype=np.float32,
    ),
    "MiniLille2_split2": np.array(
        [
            [-11.270386, -52.056335],
            [-7.2437744, -53.21167],
            [2.3392334, -16.881104],
            [-1.8076172, -16.075562],
        ],
        dtype=np.float32,
    ),
    "MiniParis1_split1": np.array(
        [
            [-13.398174, 32.048706],
            [8.890404, -3.3701782],
            [11.532715, -1.3626099],
            [-8.367008, 34.48767],
        ],
        dtype=np.float32,
    ),
    "MiniParis1_split2": np.array(
        [
            [-10.885216, 66.33209],
            [-5.750868, 65.864624],
            [-3.6149998, 90.34259],
            [-9.076685, 90.780945],
        ],
        dtype=np.float32,
    ),
    "MiniDijon9": np.array(
        [
            [-52.935486, -22.587769],
            [-49.467407, -28.431564],
            [31.811981, 15.159637],
            [29.539429, 19.625504],
        ],
        dtype=np.float32,
    ),
}

Z_GROUNDS = {
    "MiniLille1": -2.774031,
    "MiniLille2_split1": -2.591594,
    "MiniLille2_split2": -3.23685,
    "MiniParis1_split1": -5.165518,
    "MiniParis1_split2": -6.5578775,
    "MiniDijon9": -4.729243,
}

ROTATIONS = {
    "MiniLille1": 0.6877497,
    "MiniLille2_split1": 1.1541955,
    "MiniLille2_split2": 1.3136249,
    "MiniParis1_split1": 2.132471,
    "MiniParis1_split2": 1.4969587,
    "MiniDijon9": 0.4730679,
}

Z_GROUND = -1.703319
MAX_DISTANCE = 48.8722
MAX_HEIGHT = 1.8629856
MIN_DISTANCE = 2.090909
N_POINTS = 122261
BINS = np.logspace(
    np.log(MIN_DISTANCE) / np.log(10), np.log(MAX_DISTANCE) / np.log(10), 10
)
N_PER_BINS = np.array(
    [6159, 6883, 20252, 24437, 16630, 15982, 14253, 10516, 7144], dtype=np.int32
)

# For the paths
CITY_INFERANCE_FOLDER = {
    "MiniLille1": "00",
    "MiniLille2_split1": "01",
    "MiniLille2_split2": "02",
    "MiniParis1_split1": "03",
    "MiniParis1_split2": "04",
    "MiniDijon9": "11",
}
PATH_INDEXES_TO_KEEP = "data/indexes_to_keep"
PATH_SAMPLES = "data/samples/"

# For the labels
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
    KITTI_TO_CITY_NUMBERS[
        KITTI_LABELS_TO_NUMBERS[kitti_label]
    ] = CITY_LABELS_TO_NUMBERS[city_label]

# The dict is not injective, take opposite listing to have the lowest numbers
CITY_TO_KITTI_NUMBERS = {v: k for k, v in list(KITTI_TO_CITY_NUMBERS.items())[::-1]}
