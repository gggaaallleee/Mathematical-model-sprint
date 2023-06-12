'''
Author: gggaaallleee 1293587368@qq.com
Date: 2023-04-24 22:08:24
LastEditors: gggaaallleee 1293587368@qq.com
LastEditTime: 2023-04-24 22:25:52
FilePath: \2023数模准备\微分方程\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%A
'''
import taichi as ti
# 初始化 Taichi 环境
ti.init(arch=ti.cpu)
# 定义变量
dt = 0.01
t_end = 10.0
n_steps = int(t_end / dt)
m = 1.0
k = 1.0
x0 = 1.0
v0 = 0.0
# 定义数组
x = ti.field(dtype=ti.f32, shape=n_steps)
v = ti.field(dtype=ti.f32, shape=n_steps)
t = ti.field(dtype=ti.f32, shape=n_steps)
# 定义初始条件
x[0] = x0
v[0] = v0
t[0] = 0.0
# 定义 kernel 函数
@ti.kernel
def solve():
    for i in range(n_steps - 1):
        a_i = -k * x[i] / m
        '''
        a_i = -k * x[i] / m - b * v[i] / m
        a_half_i = -k * x_i / m - b * v_half_i / m
        如果你想修改输入的微分方程，需要更改求解微分方程的函数  `solve()`  中的代码。具体来说，你需要更改计算加速度  `a_i`  和  `a_half_i`  的公式。例如，如果你想将弹簧质点的运动改为阻尼运动，可以将公式修改为：
        a_i = -k * x[i] / m - b * v[i] / m
        a_half_i = -k * x_i / m - b * v_half_i / m
        其中， `b`  是阻尼常数。
        '''
        v_half_i = v[i] + 0.5 * a_i * dt
        x_i = x[i] + v_half_i * dt
        a_half_i = -k * x_i / m
        v[i + 1] = v_half_i + 0.5 * a_half_i * dt
        x[i + 1] = x_i + v[i + 1] * dt
        t[i + 1] = t[i] + dt
# 调用 kernel 函数求解微分方程
solve()
# 打印结果
print('x:', x.to_numpy())
print('v:', v.to_numpy())
print('t:', t.to_numpy())
'''这个 Python 代码示例中，通过求解基于龙格库塔4阶的微分方程，求解了一个弹簧质点的运动过程。返回值分别为质点的位置、速度和时间，即：
 -  `x` : 类型为 NumPy 数组，表示质点的位置随时间变化的数组。
-  `v` : 类型为 NumPy 数组，表示质点的速度随时间变化的数组。
-  `t` : 类型为 NumPy 数组，表示时间的数组，其每个元素对应的是  `x`  和  `v`  数组中质点位置和速度的时间点。例如， `t[3]`  对应的是质点在第三个时间步长时的位置和速度。'''