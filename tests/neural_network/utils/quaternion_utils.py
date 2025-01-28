import numpy as np

def quaternion_conjugate(q):
    """Return the conjugate of a quaternion."""
    q_w, q_x, q_y, q_z = q
    return np.array([q_w, -q_x, -q_y, -q_z])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])


def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize a quaternion."""
    norm = np.linalg.norm(q)
    if norm > 0:
        return q / norm
    return q  # Return the zero quaternion unchanged


def rotate_vector_by_quaternion(v, q):
    q_conj = quaternion_conjugate(q)
    v_rotated = quaternion_multiply(quaternion_multiply(q, [0] + v.tolist()), q_conj)[1:]
    return np.array(v_rotated)

