import numpy as np
import matplotlib.pyplot as plt

g = 9.81
rho = 1.2

def cd_soccer(v, vc=12.19, vs=1.309):
    return 0.155 + 0.346 / (1 + np.exp((v - vc) / vs))

def simulate(
    mode="vacuum",
    m=0.43,
    R=0.11,
    v0=(20, 8, 0),
    pos0=(0, 0, 0),
    omega=(0, 0, 0),
    cL=0.9,
    dt=0.001,
    tmax=10
):
    pos = np.array(pos0, dtype=float)
    vel = np.array(v0, dtype=float)
    omega = np.array(omega, dtype=float)

    xs, ys, zs = [], [], []

    for _ in range(int(tmax / dt)):
        xs.append(pos[0])
        ys.append(pos[1])
        zs.append(pos[2])

        if pos[1] < 0:
            break

        Fg = np.array([0, -m*g, 0])
        F = Fg.copy()

        speed = np.linalg.norm(vel)

        if mode in ["drag", "magnus"]:
            cD = cd_soccer(speed)
            Fd = -0.5 * cD * rho * np.pi * R**2 * speed * vel
            F += Fd

        if mode == "magnus":
            Fm = cL * np.pi * R**3 * rho * np.cross(omega, vel)
            F += Fm

        acc = F / m
        vel += acc * dt
        pos += vel * dt

    return np.array(xs), np.array(ys), np.array(zs)


# 1. 진공 포물선
x, y, z = simulate(mode="vacuum", v0=(20, 10, 0))
plt.plot(x, y, label="Vacuum")

# 2. 공기저항 포함
x, y, z = simulate(mode="drag", v0=(20, 10, 0))
plt.plot(x, y, label="Air drag")

# 3. 스핀 포함: 축구공 커브볼
# omega = (0, 40, 0) 이면 y축 방향 스핀 → z방향으로 휨
x, y, z = simulate(mode="magnus", v0=(20, 5, 2), omega=(0, 46, 0), cL=0.9)
plt.plot(x, y, label="Magnus spin")

plt.xlabel("x position (m)")
plt.ylabel("y height (m)")
plt.legend()
plt.grid()
plt.show()


# 3D 커브 확인용 그래프
x, y, z = simulate(mode="magnus", v0=(20, 5, 2), omega=(0, 46, 0), cL=0.9)

plt.plot(x, z)
plt.xlabel("x position (m)")
plt.ylabel("z side curve (m)")
plt.title("Curveball sideways motion")
plt.grid()
plt.show()


# 4. 바운스 모델
def bounce(v1, omega1, R=0.11, alpha=2/3, ey=0.7, ex=0.3, ez=0.3):
    vx1, vy1, vz1 = v1
    wx1, wy1, wz1 = omega1

    vy2 = -ey * vy1

    vx2 = ((alpha - ex) * vx1 - R * (1 + ex) * wz1) / (1 + alpha)
    wz2 = ((1 - ex) * vx1 + R * (alpha + ex) * wz1) / (R * (1 + alpha))

    vz2 = ((alpha - ez) * vz1 + R * (1 + ez) * wx1) / (1 + alpha)
    wx2 = (-(1 - ez) * vz1 + R * (alpha + ez) * wx1) / (R * (1 + alpha))

    return np.array([vx2, vy2, vz2]), np.array([wx2, wy1, wz2])


v_before = np.array([8, -5, 0])
omega_before = np.array([0, 0, 30])

v_after, omega_after = bounce(v_before, omega_before)

print("바운스 전 속도:", v_before)
print("바운스 후 속도:", v_after)
print("바운스 후 각속도:", omega_after)