# -*- coding: utf-8 -*-
import random

import matplotlib.pyplot as plt
import numpy as np


class OutlierDetection(object):
    @staticmethod
    def median_absolute_deviation(data, b=1.4826):
        """
        如果一个点大于或小于3倍的绝对中位差，那它就被是视为异常点。中位值作为评估值要
        健壮得多，它是在升序排列的多个观察值中位于中间的观察值，要想彻底改变中位值，要
        替换掉远离中位值的一半观察值，因此有限样本击穿点是50%。
        Args:
            data (list, np.asarray): 原始数据
            b (float): 阈值系数, 影响对异常值的容忍度

        Returns:

        """
        outliers = []
        if isinstance(data, list):
            np.asarray(data)
        median = np.median(data)
        mad = b * np.median(np.abs(data - median))
        mad_upper = median + 3 * mad
        mad_lower = median - 3 * mad

        return mad_upper, mad_lower

    @staticmethod
    def z_score(data, threshold=3):
        """
        标准分数是一个观测或数据点的值高于被观测值或测量值的平均值的标准偏差的符号数。
        """
        mean_d = np.mean(data)
        std_d = np.std(data)
        outliers = []

        for y in data:
            z_score = (y - mean_d) / std_d
            if np.abs(z_score) > threshold:
                outliers.append(y)
        return outliers

    @staticmethod
    def inter_quartile_range(sr):
        """
        四分位距通常是用来构建箱形图，以及对概率分布的简要图表概述。对一个对称性分布数
        据（其中位数必然等于第三四分位数与第一四分位数的算术平均数），二分之一的四分差
        等于绝对中位差（MAD）。中位数是集中趋势的反映。
        公式：IQR = Q3 − Q1
        """
        q1 = sr.quantile(0.25)
        q3 = sr.quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        outliers = sr.loc[(sr < fence_low) | (sr > fence_high)]
        return outliers


if __name__ == '__main__':
    random_number = [x * random.uniform(-1, 1) for x in range(10)]
    test_data = np.asarray(range(10)) * random_number
    print(test_data)
    plt.plot(range(10), test_data)
    plt.show()
    plt.close()
    res = OutlierDetection().median_absolute_deviation(test_data)
    print(res)
