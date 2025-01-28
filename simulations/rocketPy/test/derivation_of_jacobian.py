import sympy as sp

# Define symbolic variables for the quaternion and vector
q0, q1, q2, q3 = sp.symbols('q0 q1 q2 q3')  # Quaternion components
r0, r1, r2 = sp.symbols('r0 r1 r2')  # Vector components

# Define the quaternion representing the vector
r_quat = sp.Matrix([0, r0, r1, r2])

# Define the quaternion q
q = sp.Matrix([q0, q1, q2, q3])

# Define the quaternion conjugate
q_conjugate = sp.Matrix([q0, -q1, -q2, -q3])

# Function to perform quaternion multiplication
def quaternion_multiply(p, r):
    return sp.Matrix([
        p[0] * r[0] - p[1] * r[1] - p[2] * r[2] - p[3] * r[3],
        p[0] * r[1] + p[1] * r[0] + p[2] * r[3] - p[3] * r[2],
        p[0] * r[2] - p[1] * r[3] + p[2] * r[0] + p[3] * r[1],
        p[0] * r[3] + p[1] * r[2] - p[2] * r[1] + p[3] * r[0]
    ])

# Rotate the reference vector from ENU to sensor frame
m_quat = quaternion_multiply(quaternion_multiply(q, r_quat), q_conjugate)

# Calculate the Jacobian matrix J
# The output is m_quat, which has four components: m0, m1, m2, m3
J = sp.Matrix.zeros(4, 7)  # 4 outputs, 7 inputs (3 from r and 4 from q)

# Compute the Jacobian with respect to r and q
for i in range(4):  # For each component of m_quat
    for j in range(3):  # For components of r (r0, r1, r2)
        J[i, j] = sp.diff(m_quat[i], sp.symbols(f'r{j}'))
    for j in range(4):  # For components of q (q0, q1, q2, q3)
        J[i, j + 3] = sp.diff(m_quat[i], sp.symbols(f'q{j}'))

# Function to format and print the Jacobian matrix nicely
def print_jacobian(J):
    print("Jacobian Matrix:")
    for i in range(J.shape[0]):
        row = " | ".join(f"{J[i, j].simplify()}" for j in range(J.shape[1]))
        print(f"[ {row} ]")

# Print the Jacobian matrix in a readable format
print_jacobian(J)
