import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from datetime import date

csv_in_use = "children-born-per-woman.csv"


def read_fetrtility():
    '''
    reads Retirement_Age.csv file and parses the year column as a date and sets it as an index
    '''
    #GET RID OF HARDCODED PATH
    fertility_df = pd.read_csv(csv_in_use, parse_dates=['Year'])

    return fertility_df

fertility_df = read_fetrtility()


#maybe pass top function into this func???
def get_spanish_fertility():

    spain_fertility_df= fertility_df[fertility_df['Entity'] == 'Spain'].copy()
    # Make sure Year is datetime first
    spain_fertility_df["Year"] = pd.to_datetime(spain_fertility_df["Year"], errors="coerce")

    # Extract the year as an integer
    spain_fertility_df["Year"] = spain_fertility_df["Year"].dt.year

    return spain_fertility_df

spain_fertility_df = get_spanish_fertility()


def plot_poly_graph():

    x = spain_fertility_df["Year"]
    y = spain_fertility_df["Fertility rate (period), historical"]


    # Perform linear fit
    coefficients = np.polyfit(x, y, deg=10, )
  

    # Create polynomial function
    p = np.poly1d(coefficients)

    plt.scatter(x, y, label='Data Points')
    plt.plot(x, p(x), label='Linear Fit', color='red')
    plt.legend()
    plt.show()



x = spain_fertility_df["Year"]
y = spain_fertility_df["Fertility rate (period), historical"]

sigma = 0.05

degrees = [1,2,3,4,5,6,7,8,9,10]

chi2_list = []

chi2_reduced_list = []

for d in degrees:
    # Perform linear fit
    coefficients = np.polyfit(x, y, deg = d )

    # Create polynomial function
    p = np.poly1d(coefficients)
    y_pred = p(x)
    resid = y - p(x)
    sigma = 0.05*y

    chi2 = np.sum(((y - y_pred) / sigma)**2)

    dof = len(x) - (d + 1)

    chi2_reduced = chi2/dof

    chi2_list.append(chi2)
    chi2_reduced_list.append(chi2_reduced)


plt.plot(degrees, chi2_reduced_list, marker="o")
plt.xlabel("Polynomial order (n)")
plt.ylabel("Weighted x**2 per degrees of freedom")
plt.title("Model comparison by polynomial order ")
plt.grid(True)
plt.show()


