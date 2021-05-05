from os.path import join, dirname

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data file create by module 05
df = pd.read_csv(join(dirname(__file__), "its-alive-zoom-stats.csv"))
df.set_index("frame", verify_integrity=True)

# print(data.head)

y1 = np.exp(df["variance from t-1"][1000:])
y2 = df["mean diff"][1000:]

y1.plot(logy=True, ylabel="variance of percentage of diff from t-1")
y2.plot(secondary_y=True, style="g", ylabel="mean of percentage of diff")
plt.show()

