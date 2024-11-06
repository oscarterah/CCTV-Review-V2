from filterpy.kalman import KalmanFilter
import numpy as np


def main():
    kf = KalmanFilter(2, 2, 1)
    z = np.array([[1.0], [2.0]])
    kf.update(z)

    print("After update:")
    print("x:", kf.x.T)
    print("P:\n", kf.P)

    u = np.array([[1.0]])
    kf.predict(u)

    print("After predict:")
    print("x:", kf.x.T)
    print("P:\n", kf.P)

if __name__ == "__main__":
	main()
