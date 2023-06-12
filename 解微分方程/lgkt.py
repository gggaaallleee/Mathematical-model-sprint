'''
Author: gggaaallleee 1293587368@qq.com
Date: 2023-04-24 21:27:53
LastEditors: gggaaallleee 1293587368@qq.com
LastEditTime: 2023-04-24 21:31:52
FilePath: \2023数模准备\微分方程\lgkt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import math
import matplotlib.pyplot as plt
 
 
def f1(x, y1, y2,y3):
    df = -0.013*y1-1000*y1*y2
    return df
 
def f2(x, y1, y2,y3):
    df = -2500*y2*y3
    return df
 
def f3(x,y1,y2,y3):
    df=-0.013*y1-1000*y1*y2-2500*y2*y3
    return df
 
 
def RK4(x, y1, y2,y3, h):
    """
    :param x: Initial value of X
    :param y1: Initial value of y1
    :param y2: Initial value of y2
    :param y3: Initial value of y3
    :param h: time step
    :return: New iterative solution
    """
    xarray, y1array, y2array,y3array = [], [], [], []
    while x <= 0.02:
        xarray.append(x)
        y1array.append(y1)
        y2array.append(y2)
        y3array.append(y3)
        x += h
 
        K_1 = f1(x, y1, y2,y3)
        L_1 = f2(x, y1, y2,y3)
        M_1 = f3(x, y1, y2,y3)
        K_2 = f1(x + h / 2, y1 + h / 2 * K_1, y2 + h / 2 * L_1 , y3 + h/2 * M_1)
        L_2 = f2(x + h / 2, y1 + h / 2 * K_1, y2 + h / 2 * L_1 , y3 + h/2 * M_1)
        M_2 = f3(x + h / 2, y1 + h / 2 * K_1, y2 + h / 2 * L_1, y3 + h / 2 * M_1)
        K_3 = f1(x + h / 2, y1 + h / 2 * K_2, y2 + h / 2 * L_2 , y3 + h/2 * M_2)
        L_3 = f2(x + h / 2, y1 + h / 2 * K_2, y2 + h / 2 * L_2 , y3 + h/2 * M_2)
        M_3 = f3(x + h / 2, y1 + h / 2 * K_2, y2 + h / 2 * L_2, y3 + h / 2 * M_2)
        K_4 = f1(x + h, y1 + h * K_3, y2 + h * L_3, y3 + h * M_3)
        L_4 = f2(x + h, y1 + h * K_3, y2 + h * L_3, y3 + h * M_3)
        M_4 = f3(x + h, y1 + h * K_3, y2 + h * L_3, y3 + h * M_3)
 
        y1 = y1 + (K_1 + 2 * K_2 + 2 * K_3 + K_4) * h / 6
        y2 = y2 + (L_1 + 2 * L_2 + 2 * L_3 + L_4) * h / 6
        y3 = y3 + (M_1 + 2 * M_2 + 2 * M_3 + M_4) * h / 6
    return xarray, y1array, y2array,y3array
 
 
def main():
    xarray, y1array, y2array,y3array = RK4(0, 1, 1, 0, 0.0001)
    print("Runge Kutta numerical results".center(168))
    print('-' * 420)
    #\t ：表示空4个字符，类似于文档中的缩进功能，相当于按一个Tab键。
    print("object\\time\t", "x=0\t\t", " x=0.001\t\t"," x=0.002\t\t"," x=0.003\t\t", " x=0.004\t\t\t"," x=0.005\t\t","x=0.006\t\t","x=0.007\t\t","x=0.008\t\t","x=0.009\t\t",
          "x=0.010\t\t","x=0.011\t\t","x=0.012\t\t","x=0.0013\t\t","x=0.014\t\t","x=0.015\t\t","x=0.016\t\t","x=0.017\t\t","x=0.018\t\t","x=0.019\t\t","x=0.020\t\t")
    print('-' * 420)
    print("y1:", end='')
    for i in range(len(y1array)):
        if i % 10 == 0:
            print("\t\t", "%8.7f" % y1array[i], end='')
    print('\n', '-' * 420)
    print("y2:", end='')
    for i in range(len(y2array)):
        if i % 10 == 0:
            print("\t\t", "%8.7f" % y2array[i], end='')
    print('\n', '-' * 420)
    print("y3:", end='')
    for i in range(len(y3array)):
        if i % 10 == 0:
            print("\t\t", "%8.7f" % y3array[i], end='')
 
    print('\n', '-' * 420)
    plt.figure('Runge Kutta numerical results')
    plt.subplot(221)
    #plt.plot(xarray, y1array, label='y1_runge_kutta')
    plt.scatter(xarray, y1array, label='y1_scatter', s=1, c='#DC143C', alpha=0.6)
    #plt.y1label('x')
    plt.legend()
    plt.subplot(222)
    #plt.plot(xarray, y2array, label='y2_runge_kutta')
    plt.scatter(xarray, y2array, label='y2_scatter', s=1, c='#DC143C', alpha=0.6)
    #plt.y2label('x')
    plt.legend()
    plt.subplot(223)
    #plt.plot(xarray, y3array, label='y3_runge_kutta')
    plt.scatter(xarray, y3array, label='y3_scatter', s=1, c='#DC143C', alpha=0.6)
    #plt.y3label('x')
    plt.legend()
    plt.show()
 
 
if __name__ == '__main__':
    main()