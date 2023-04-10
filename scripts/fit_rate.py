import matplotlib.pyplot as plt
from lmfit import Parameters, minimize, fit_report
import numpy as np
import polars as pl
from typing import Optional

#Set these based on your reaction
RED_MASS: float = 0.0 # Reduced mass of reactants in amu
S0: float = 0.0 # S-factor at E=0

#The normal reaclib rate parameterization
def rate_function(params: Parameters, x: np.ndarray, data: Optional[np.ndarray] = None) -> np.ndarray:
    vals = params.valuesdict()
    expo = np.zeros(len(x)) + vals["a0"]
    for i in range(1, 6):
        par = vals[f"a{i}"]
        expo += par * ((x ** (2.0*float(i)-5.0)) / (3.0))
    expo = np.exp(expo) * (x **(vals["a6"]))
    if data is None:
        return expo
    else:
        return (expo - data)

#The log of the reaclib rate parameterization. This is better for fitting because it 
#reduces bad behavior of reaclib function as well as compensates for many order of magnitude scale of rates.
def log_rate_function(params: Parameters, x: np.ndarray, data: Optional[np.ndarray]=None) -> np.ndarray:
    vals = params.valuesdict()
    expo = np.zeros(len(x)) + vals["a0"]
    for i in range(1, 6):
        par = vals[f"a{i}"]
        expo += par * ((x ** (2.0*float(i)-5.0)) / (3.0))
    expo += np.log(x)*(vals["a6"])
    if data is None:
        return expo
    else:
        return (expo - data)

#Generate our fit parameters with some initial guesses based on reaclib recommendations
def make_parameters() -> Parameters:
    params = Parameters()

    #Recommended parameters for non-resonant charged particle reaction rate.
    #We don't fix any but a1 cause they usually all need to float, but a1 causes instability
    params.add("a0", value=np.log(7.8318e9 * ((4.0*1.0/RED_MASS) ** (1.0/3.0)) * S0), vary=True)
    params.add("a1", 0.0, vary=False)
    params.add("a2", -4.2486*((4.0**2.0 * 1.0 * RED_MASS) ** (1.0/3.0)), vary=True)
    params.add("a3", 0.0)
    params.add("a4", 0.0)
    params.add("a5", 0.0)
    params.add("a6", -2.0/3.0, vary=True)

    return params

def run():
    #Open the rate data to a lazy dataframe
    rate_frame = pl.scan_csv("my_rate.csv")

    #Select the subset of the data to fit to
    subset = rate_frame.filter(pl.col("Temperature(GK)") < 3.0).collect()

    #Convert it to numpy arrays for work
    x = subset.select("Temperature(GK)").to_numpy().flatten() #Flatten just enforces 1-d array result
    rate = subset.select("Rate(cm^3/(mol*s))").to_numpy().flatten()

    log_rate = np.log(rate)
    params = make_parameters()
    #We fit to the log of the rate, to compensate for the many order of magnitude span of rate values.
    out = minimize(log_rate_function, params, args=(x,), kws={"data": log_rate}, nan_policy='raise')
    print(fit_report(out))

    #Plot the results (both log and non-log)
    fig, ax = plt.subplots(1,2)

    ax[0].plot(x, rate, label="data")
    ax[0].plot(x, rate_function(out.params, x), label="fit")
    ax[0].set_ylabel("Rate")
    ax[0].set_xlabel("Temperature")
    ax[0].set_yscale("log")
    ax[0].set_ylim(10.0, 1.0e9)
    ax[0].legend()

    ax[1].plot(x, log_rate, label="lndata")
    ax[1].plot(x, log_rate_function(out.params, x), label="lnfit")
    ax[1].set_ylabel("Ln(Rate)")
    ax[1].set_xlabel("Temperature")
    ax[1].legend()

    fig.tight_layout()
    plt.show()

    with open("fit_result.txt", "w") as outfile:
        outfile.write(fit_report(out))

run()