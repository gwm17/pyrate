import matplotlib.pyplot as plt
import polars as pl

#Read in the rate data using polars
rate_frame = pl.scan_csv("my_rate.csv")
#Select some subset of the dataset
subset = rate_frame.filter(pl.col("Temperature(GK)") < 3.0).collect()

#Extract to numpy for plotting
rate = subset.select("Rate(cm^3/(mol*s))").to_numpy().flatten()
temperature = subset.select("Temperature(GK)").to_numpy().flatten()

#Plot
fig, ax = plt.subplots(1,1)
ax.plot(temperature, rate, label="rate")
ax.set_ylabel("Rate")
ax.set_xlabel("Temperature")
ax.set_ylim(10.0, 1.0e9)
ax.set_yscale('log')
fig.legend()
fig.tight_layout()
plt.show()