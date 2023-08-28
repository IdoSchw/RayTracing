import numpy as np

class Vector:
    def __init__(self, origin, direction):
        self.origin = np.array(origin)
        self.direction = normalize_vector(np.array(direction))


def get_length(start_point, end_point):
    vector = end_point - start_point

    # Computing the length of the vector using the Euclidean norm
    vector_length = np.linalg.norm(vector)
    return vector_length


def normalize_vector(vector: np.array):
    magnitude = np.linalg.norm(vector)
    normalized_vector = vector / magnitude
    return normalized_vector