import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt


eps = 1e-8
# function to convert degrees to radians and vice versa
degrees2radians = lambda degrees: degrees * (pi/180)
radians2degrees = lambda radians: radians * (180/pi)

# # image
# height, width = 50, 50
# image = np.zeros((height, width))
# image[10:40, 10:40] = np.eye(30)
# image[10:40, 40] = 1
# image[10, 10:40] = 1

# image
height, width = 50, 50
image = np.zeros((height, width))
# image[[0, 0, 24, 49, 49], [0, 49, 24, 0, 49]] = 1
image[[20, 30], [20, 32]] = 1

# thetas and rhos
# thetas = degrees2radians(np.arange(-90, 90, 45))
thetas = degrees2radians(np.arange(-90, 90, 3))
max_distance = int(np.ceil((height ** 2 + width ** 2) ** 0.5))
# rhos = np.linspace(-max_distance, max_distance, max_distance * 2)
rhos = np.linspace(-max_distance, max_distance, max_distance * 2)
n_thetas = thetas.size
n_rhos = rhos.size
print(n_rhos)
print(n_thetas)


# hough accumulator and image peaks
hough_accumulator = np.zeros((n_rhos, n_thetas)).astype(np.int32)
ycoords, xcoords = np.where(image != 0)
n_points = xcoords.size

# accumulate
for x, y in zip(xcoords, ycoords):
    for i, theta in enumerate(thetas):
        rho = int(np.round(x * cos(theta) + y * sin(theta)) + max_distance)
        hough_accumulator[rho, i] += 1

# peak detection
# print(np.unique(hough_accumulator))
y_list, x_list = np.where(hough_accumulator >= 0.6*np.max(hough_accumulator))
rho_list = rhos[y_list]
theta_list = thetas[x_list]
print(f"num rhos = {len(rho_list)}")
print(f"rhos={rho_list}")
print(f"thetas={radians2degrees(theta_list)}")

# for rho, theta in zip(rho_list, theta_list):


# line computation
f = lambda rho, theta, x: (rho - x * cos(theta)) / (sin(theta) + eps)
x1 = 0
x2 = width
xcoords = [(x1, x2)]
ycoords = []
for rho, theta in zip(rho_list, theta_list):
    y1 = f(rho, theta, x1)
    y2 = f(rho, theta, x2)
    ycoords.append((y1, y2))
print(xcoords)
print(ycoords)


fig, ax = plt.subplots(nrows=1, ncols=2)
ax1, ax2 = ax.flatten()
ax1.imshow(hough_accumulator, extent=[0, n_thetas, n_rhos, 0])
for ycoord in ycoords:
    ax2.plot(*xcoords, ycoord)
ax2.imshow(image, extent=[0, width, height, 0])

plt.show(block=True)


