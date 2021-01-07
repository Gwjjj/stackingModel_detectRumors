# 导入pyplot
from matplotlib import pyplot as plt 
import numpy as np
# from scipy.interpolate import spline
from scipy.interpolate import make_interp_spline
# 数据在X轴的位置，是一个可迭代的对象
x = np.array([2000, 4000, 6000, 8000, 10000, 12000, 14000])

y1 = np.array([0.924, 0.937, 0.942, 0.949, 0.956, 0.959, 0.966])

y2 = np.array([0.916, 0.938, 0.941, 0.950, 0.955, 0.958, 0.961])
y3 = np.array([0.91 , 0.931, 0.942, 0.948, 0.949, 0.954, 0.958])

# y1 = np.array([6.5,16, 24.1, 32, 39, 47.5, 55])

# y2 = np.array([3.5, 5.9, 9, 15.1, 20, 25.5, 31])
# y3 = np.array([3.1 , 5.1, 7, 9, 11, 13.1, 15])
xnew = np.linspace(x.min(),x.max(),300) #300 represents number of points to make between T.min and T.max
y_smooth1 = make_interp_spline(x, y1)(xnew)
y_smooth2 = make_interp_spline(x, y2)(xnew)
y_smooth3 = make_interp_spline(x, y3)(xnew)
# 传入x,y，通过pyplot绘制出折线图
l1, = plt.plot(xnew, y_smooth1,color = 'black')
l2, = plt.plot(xnew, y_smooth2,linestyle='--',color = 'black')
l3, = plt.plot(xnew, y_smooth3,linestyle=':',color = 'black')
plt.legend(handles = [l1, l2, l3], labels = ["120 seq", "100 seq", "80 seq"])
plt.xlabel("Epoch")
# plt.ylabel("Minutes")
plt.ylabel("Accuracy")
# 展示图形
plt.show()