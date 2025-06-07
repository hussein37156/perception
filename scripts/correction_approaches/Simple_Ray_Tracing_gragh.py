import numpy as np
import matplotlib.pyplot as plt


disparity_values = np.linspace(70, 5, num=641)
baseline = 12.0  # cm


def line_plane_intersection(Pline, Pplane, Vline, Vplane):
    h = np.dot(Vplane, (Pplane - Pline)) / np.dot(Vplane, Vline)
    P = Pline + h * Vline
    return P

def snells_law(n1, n2, theta1):
    return np.arcsin((n1 / n2) * np.sin(theta1))

def closest_point_between_rays(A2, B2, v1, v2):
    A = np.dot(v1, v1)
    B = np.dot(v1, v2)
    C = np.dot(v2, v2)
    D = np.dot(v1, A2 - B2)
    E = np.dot(v2, A2 - B2)
    denom = A * C - B * B
    if denom == 0:
        return (A2 + B2) / 2
    s1 = (B * E - C * D) / denom
    s2 = (A * E - B * D) / denom
    P1 = A2 + s1 * v1
    P2 = B2 + s2 * v2
    return (P1 + P2) / 2

def compute_depth_for_disparity(disparity):
    D = 2.0
    d = 6.0
    n1 = 1.0
    n2 = 1.4555
    n3 = 1.333333
    T = np.array([baseline, 0, 0])
    R = np.eye(3)
    N = np.array([0, 0, 1])

    Pl_uv = np.array([320 , 180, 1])
    Pr_uv = np.array([320 - disparity, 180, 1])

    Ol = np.array([0, 0, 0])
    Or = T

    #fx_left, fy_left = 699.63/2, 699.63/2
    #cx_left, cy_left = 644.275/2, 332.0365/2
    
    fx_left, fy_left = 700.515/2, 700.515/2
    cx_left, cy_left = 662.935/2, 353.9215/2
    K_left = np.array([[fx_left, 0, cx_left],
                       [0, fy_left, cy_left], 
                       [0, 0, 1]])

    fx_right, fy_right = 700.515/2, 700.515/2
    cx_right, cy_right = 662.935/2, 353.9215/2
    K_right = np.array([[fx_right, 0, cx_right], 
                        [0, fy_right, cy_right], 
                        [0, 0, 1]])

    K_left_inv = np.linalg.inv(K_left)
    K_right_inv = np.linalg.inv(K_right)

    Pl = K_left_inv @ Pl_uv
    Pr = K_right_inv @ Pr_uv
    Pr = R @ Pr + T

    zeta_r_1 = (Pr - Or) / np.linalg.norm(Pr - Or)
    zeta_l_1 = (Pl - Ol) / np.linalg.norm(Pl - Ol)

    C1 = np.array([0, 0, D])  # Plane at depth D
    A1 = line_plane_intersection(Ol, C1, zeta_l_1, N)
    B1 = line_plane_intersection(Or, C1, zeta_r_1, N)

    alpha_1 = np.arccos(np.dot(N, zeta_l_1))
    beta_1 = np.arccos(np.dot(N, zeta_r_1))

    alpha_2 = snells_law(n1, n2, alpha_1)
    beta_2 = snells_law(n1, n2, beta_1)

    zeta_r_2 = ((n1 / n2) * zeta_r_1) - ((n1 / n2) * np.cos(beta_1) - np.cos(beta_2)) * N
    zeta_l_2 = ((n1 / n2) * zeta_l_1) - ((n1 / n2) * np.cos(alpha_1) - np.cos(alpha_2)) * N

    C2 = np.array([0, 0, (D + d)])  # Plane at depth D + d
    A2 = line_plane_intersection(A1, C2, zeta_l_2, N)
    B2 = line_plane_intersection(B1, C2, zeta_r_2, N)

    alpha_4 = snells_law(n2, n3, alpha_2)
    beta_4 = snells_law(n2, n3, beta_2)

    zeta_r_3 = ((n2 / n3) * zeta_r_2) - ((n2 / n3) * np.cos(beta_2) - np.cos(beta_4)) * N
    zeta_l_3 = ((n2 / n3) * zeta_l_2) - ((n2 / n3) * np.cos(alpha_2) - np.cos(alpha_4)) * N
    

    P_final = closest_point_between_rays(A2, B2, zeta_l_3, zeta_r_3)
    return P_final[2]  # Return positive depth

def main():
    depths = []
    for disp in disparity_values:
        try:
            depth = compute_depth_for_disparity(disp)
        except Exception as e:
            print(f"Error at disparity {disp}: {e}")
            depth = np.nan
        depths.append(depth)

    plt.figure(figsize=(10, 6))
    plt.plot(12*350/disparity_values, depths)
    plt.xlabel("Raw Depth (cm)")
    plt.ylabel("Corrected Depth  (cm)")
    plt.title("Raw Depth vs Corrected Depth")
    plt.grid(True)
    #plt.gca().invert_xaxis()  # Optional: to match left-to-right disparity intuition
    plt.show()

if __name__ == '__main__':
    main()
