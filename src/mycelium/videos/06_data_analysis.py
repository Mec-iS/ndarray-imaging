from os.path import join, dirname

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

matplotlib.style.use('seaborn-ticks')

# load data file create by module 052
df = pd.read_csv(join(dirname(__file__), "its-alive-zoom-stats2.csv"))[500:3000]
df.set_index("frame", verify_integrity=True)

df2 = pd.DataFrame({
    "variance from previous frame": np.exp(df["variance from t-1 abs"]),
    "over or under mid range": df["over/under mid range"]},
    index=df["frame"]
)

fig1, fig2 = df2.plot(subplots=True, layout=(2, 1), sharex=True)

fig1[0].xaxis.grid(True)
fig2[0].xaxis.grid(True)

fig1[0].tick_params(which='both', width=3)
fig2[0].tick_params(which='both', width=3)

fig2[0].xaxis.set_major_locator(MultipleLocator(20))
fig2[0].plot(table=True)

plt.tight_layout()
plt.show()

peacks = df.loc[df['over/under mid range'] == 1]
print(peacks)

