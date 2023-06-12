'''
Author: gggaaallleee 1293587368@qq.com
Date: 2023-04-24 22:32:27
LastEditors: gggaaallleee 1293587368@qq.com
LastEditTime: 2023-04-25 18:09:05
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
        k1_y1 = dt * f1(x[i], y1[i], y2[i], y3[i])
        k1_y2 = dt * f2(x[i], y1[i], y2[i], y3[i])
        k1_y3 = dt * f3(x[i], y1[i], y2[i], y3[i])
        k2_y1 = dt * f1(x[i] + 0.5 * dt, y1[i] + 0.5 * k1_y1,
                        y2[i] + 0.5 * k1_y2, y3[i] + 0.5 * k1_y3)
        k2_y2 = dt * f2(x[i] + 0.5 * dt, y1[i] + 0.5 * k1_y1,
                        y2[i] + 0.5 * k1_y2, y3[i] + 0.5 * k1_y3)
        k2_y3 = dt * f3(x[i] + 0.5 * dt, y1[i] + 0.5 * k1_y1,
                        y2[i] + 0.5 * k1_y2, y3[i] + 0.5 * k1_y3)
        k3_y1 = dt * f1(x[i] + 0.5 * dt, y1[i] + 0.5 * k2_y1,
                        y2[i] + 0.5 * k2_y2, y3[i] + 0.5 * k2_y3)
        k3_y2 = dt * f2(x[i] + 0.5 * dt, y1[i] + 0.5 * k2_y1,
                        y2[i] + 0.5 * k2_y2, y3[i] + 0.5 * k2_y3)
        k3_y3 = dt * f3(x[i] + 0.5 * dt, y1[i] + 0.5 * k2_y1,
                        y2[i] + 0.5 * k2_y2, y3[i] + 0.5 * k2_y3)
        k4_y1 = dt * f1(x[i] + dt, y1[i] + k3_y1, y2[i] + k3_y2, y3[i] + k3_y3)
        k4_y2 = dt * f2(x[i] + dt, y1[i] + k3_y1, y2[i] + k3_y2, y3[i] + k3_y3)
        k4_y3 = dt * f3(x[i] + dt, y1[i] + k3_y1, y2[i] + k3_y2, y3[i] + k3_y3)
        y1[i + 1] = y1[i] + (k1_y1 + 2 * k2_y1 + 2 * k3_y1 + k4_y1) / 6
        y2[i + 1] = y2[i] + (k1_y2 + 2 * k2_y2 + 2 * k3_y2 + k4_y2) / 6
        y3[i + 1] = y3[i] + (k1_y3 + 2 * k2_y3 + 2 * k3_y3 + k4_y3) / 6
        x[i + 1] = x[i] + dt


# 执行 kernel 函数进行求解
solve()
# 输出结果
print('y1:', y1.to_numpy())
print('y2:', y2.to_numpy())
print('y3:', y3.to_numpy())
print('x:', x.to_numpy())

np.savetxt('x.txt',x.to_numpy())
np.savetxt('y1.txt',y1.to_numpy())
np.savetxt('y2.txt',y2.to_numpy())
np.savetxt('y3.txt',y3.to_numpy())

