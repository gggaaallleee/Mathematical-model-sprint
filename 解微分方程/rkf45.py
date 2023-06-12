'''
Author: gggaaallleee 1293587368@qq.com
Date: 2023-04-24 22:32:27
LastEditors: gggaaallleee 1293587368@qq.com
LastEditTime: 2023-04-26 07:54:10
FilePath: \2023数模准备\微分方程\lgkt_taichi2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
# 初始化 Taichi 环境
ti.init(arch=ti.cpu)
# 定义步长和总步数
dt = 0.0001
n_steps = 200
# 定义变量
y1 = ti.field(dtype=ti.f32, shape=n_steps)
y2 = ti.field(dtype=ti.f32, shape=n_steps)
y3 = ti.field(dtype=ti.f32, shape=n_steps)
x = ti.field(dtype=ti.f32, shape=n_steps)
# 初始条件
y1[0] = 1.0
y2[0] = 1.0
y3[0] = 0.0
x[0] = 0.0


@ti.func
def f1(x, y1, y2, y3):
    df = -0.013 * y1 - 1000 * y1 * y2
    return df


@ti.func
def f2(x, y1, y2, y3):
    df = -2500 * y2 * y3
    return df


@ti.func
def f3(x, y1, y2, y3):
    df = -0.013 * y1 - 1000 * y1 * y2 - 2500 * y2 * y3
    return df
# 定义 kernel 函数


@ti.kernel
def solve():
    for i in range(n_steps - 1):
        # 计算k1
        k1_y1 = dt * f1(x[i], y1[i], y2[i], y3[i])
        k1_y2 = dt * f2(x[i], y1[i], y2[i], y3[i])
        k1_y3 = dt * f3(x[i], y1[i], y2[i], y3[i])
        # 计算k2
        k2_y1 = dt * f1(x[i] + 0.25 * dt, y1[i] + 0.25 * k1_y1,
                        y2[i] + 0.25 * k1_y2, y3[i] + 0.25 * k1_y3)
        k2_y2 = dt * f2(x[i] + 0.25 * dt, y1[i] + 0.25 * k1_y1,
                        y2[i] + 0.25 * k1_y2, y3[i] + 0.25 * k1_y3)
        k2_y3 = dt * f3(x[i] + 0.25 * dt, y1[i] + 0.25 * k1_y1,
                        y2[i] + 0.25 * k1_y2, y3[i] + 0.25 * k1_y3)
        # 计算k3
        k3_y1 = dt * f1(x[i] + 0.375 * dt, y1[i] + (3/32) * k1_y1 + (9/32) * k2_y1,
                        y2[i] + (3/32) * k1_y2 + (9/32) * k2_y2, y3[i] + (3/32) * k1_y3 + (9/32) * k2_y3)
        k3_y2 = dt * f2(x[i] + 0.375 * dt, y1[i] + (3/32) * k1_y1 + (9/32) * k2_y1,
                        y2[i] + (3/32) * k1_y2 + (9/32) * k2_y2, y3[i] + (3/32) * k1_y3 + (9/32) * k2_y3)
        k3_y3 = dt * f3(x[i] + 0.375 * dt, y1[i] + (3/32) * k1_y1 + (9/32) * k2_y1,
                        y2[i] + (3/32) * k1_y2 + (9/32) * k2_y2, y3[i] + (3/32) * k1_y3 + (9/32) * k2_y3)
        # 计算k4
        k4_y1 = dt * f1(x[i] + 0.5 * dt, y1[i] + (1932/2197) * k1_y1 - (7200/2197) * k2_y1 + (7296/2197) * k3_y1,                        y2[i] + (1932/2197)
                        * k1_y2 - (7200/2197) * k2_y2 + (7296/2197) * k3_y2,                        y3[i] + (1932/2197) * k1_y3 - (7200/2197) * k2_y3 + (7296/2197) * k3_y3)
        k4_y2 = dt * f2(x[i] + 0.5 * dt, y1[i] + (1932/2197) * k1_y1 - (7200/2197) * k2_y1 + (7296/2197) * k3_y1,                        y2[i] + (1932/2197)
                        * k1_y2 - (7200/2197) * k2_y2 + (7296/2197) * k3_y2,                        y3[i] + (1932/2197) * k1_y3 - (7200/2197) * k2_y3 + (7296/2197) * k3_y3)
        k4_y3 = dt * f3(x[i] + 0.5 * dt, y1[i] + (1932/2197) * k1_y1 - (7200/2197) * k2_y1 + (7296/2197) * k3_y1,                        y2[i] + (1932/2197)
                        * k1_y2 - (7200/2197) * k2_y2 + (7296/2197) * k3_y2,                        y3[i] + (1932/2197) * k1_y3 - (7200/2197) * k2_y3 + (7296/2197) * k3_y3)
        # 计算k5
        k5_y1 = dt * f1(x[i] + dt, y1[i] + (439/216) * k1_y1 - 8 * k2_y1 + (3680/513) * k3_y1 - (845/4104) * k4_y1,                        y2[i] + (439/216) * k1_y2 -
                        8 * k2_y2 + (3680/513) * k3_y2 - (845/4104) * k4_y2,                        y3[i] + (439/216) * k1_y3 - 8 * k2_y3 + (3680/513) * k3_y3 - (845/4104) * k4_y3)
        k5_y2 = dt * f2(x[i] + dt, y1[i] + (439/216) * k1_y1 - 8 * k2_y1 + (3680/513) * k3_y1 - (845/4104) * k4_y1,                        y2[i] + (439/216) * k1_y2 -
                        8 * k2_y2 + (3680/513) * k3_y2 - (845/4104) * k4_y2,                        y3[i] + (439/216) * k1_y3 - 8 * k2_y3 + (3680/513) * k3_y3 - (845/4104) * k4_y3)
        k5_y3 = dt * f3(x[i] + dt, y1[i] + (439/216) * k1_y1 - 8 * k2_y1 + (3680/513) * k3_y1 - (845/4104) * k4_y1,                        y2[i] + (439/216) * k1_y2 -
                        8 * k2_y2 + (3680/513) * k3_y2 - (845/4104) * k4_y2,                        y3[i] + (439/216) * k1_y3 - 8 * k2_y3 + (3680/513) * k3_y3 - (845/4104) * k4_y3)
        # 更新值
        y1[i + 1] = y1[i] + (25/216) * k1_y1 + (1408/2565) * \
            k3_y1 + (2197/4101) * k4_y1 - (1/5) * k5_y1
        y2[i + 1] = y2[i] + (25/216) * k1_y2 + (1408/2565) * \
            k3_y2 + (2197/4101) * k4_y2 - (1/5) * k5_y2
        y3[i + 1] = y3[i] + (25/216) * k1_y3 + (1408/2565) * \
            k3_y3 + (2197/4101) * k4_y3 - (1/5) * k5_y3
        x[i + 1] = x[i] + dt


# 执行 kernel 函数进行求解
solve()
# 可视化结果
print('y1:', y1.to_numpy())
print('y2:', y2.to_numpy())
print('y3:', y3.to_numpy())
print('x:', x.to_numpy())

np.savetxt('x.txt',x.to_numpy())
np.savetxt('y1.txt',y1.to_numpy())
np.savetxt('y2.txt',y2.to_numpy())
np.savetxt('y3.txt',y3.to_numpy())
