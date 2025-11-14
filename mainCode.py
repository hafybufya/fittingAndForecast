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

# print(spain_fertility_df)

# def plot_fertility_women_graph():
# #plotting both OECD women and UK same graph
#     plot_fertility_graph= spain_fertility_df.plot(kind='line',y="Fertility rate (period), historical", label='Fertility rate (period), historical',  linestyle='-')
#     plot_fertility_graph.set_title("Fertility rate in Spain  (1970-2018)")
#     plt.show()

# fertility_graph = plot_fertility_women_graph()

max_year = 2015
mask = spain_fertility_df["Year"] <= max_year

x_poly = spain_fertility_df["Year"]
y_poly = spain_fertility_df["Fertility rate (period), historical"]


x_df = spain_fertility_df.loc[mask, "Year"]
y_df = spain_fertility_df.loc[mask, "Fertility rate (period), historical"]


# Perform linear fit
coefficients = np.polyfit(x_poly, y_poly, 10)
print("Linear Fit Coefficients:", coefficients)

# Create polynomial function
p = np.poly1d(coefficients)

plt.scatter(x_df, y_df, label='Data Points')
plt.plot(x_poly, p(x_poly), label='Linear Fit', color='red')
plt.legend()
plt.show()