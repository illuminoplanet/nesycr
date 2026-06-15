import numpy as np

Z_OFFSET = 0.5
FLOOR_AABB = np.array([[0.11, -0.60, -0.01], [0.73, 0.61, 0.50]])

INITIAL_DOF_POSITION = [
    0.00,
    -0.25 * np.pi,
    0.00,
    -0.75 * np.pi,
    0.00,
    0.50 * np.pi,
    0.25 * np.pi,
    0.04,
    0.04,
]

STATE_CHECK_INTERVAL = 30
RECORD_INTERVAL = 20
