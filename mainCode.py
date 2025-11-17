
# Importing functions used in the program

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy as np

# ---------------------------------------------------------------------
# Defined CSV file name and columns as well as colors used in program
#  -> make the code flexible if used dataset changed
#  -> or to reuse the same function for a different file.
# ---------------------------------------------------------------------

csv_in_use = "children-born-per-woman.csv"
x_axis = "Year"
y_axis = "Fertility rate (period), historical"
color_1 = "#FF0000"


def read_fertility():
    
    """

    Loads the fertility dataset definied in 'csv_in_use'

    Notes:
    -> Year column is not parsed as a date to allow for scatterfit graphs

    Returns:
    pandas Dataframe -> converts csv to df containing fertility data

    """

    fertility_df = pd.read_csv(csv_in_use, parse_dates=[x_axis])

    return fertility_df

# Calls function so to be used in spanish_fertility()
fertility_df = read_fertility()



def get_spanish_fertility():
    """

    Filter the fertility dataset to include only Spanish data

    Notes:
    -> Apply a mask to select rows in 'Entity' == 'Spain'
    -> Extract the year as an integer for creation of plots

    Returns:
    pandas Dataframe -> Filtered dataframe containg only Spanish fertility data
                         with the 'Year' column as integers.

    """

    spain_fertility_df= fertility_df[fertility_df['Entity'] == 'Spain'].copy()
  
    # Extract the year as an integer
    spain_fertility_df[x_axis] = spain_fertility_df[x_axis].dt.year

    return spain_fertility_df

# Calls function so to be used in scatter plots
spain_fertility_df = get_spanish_fertility()


def plot_prediction_graph(x, y, degree):
    """

    Creates a plot containing both historical data sub-sampled all but the 
    past 10 years and polynomial prediction 10 years into the future

    Notes:
    -> x, y and degree passed into the function
    -> Convert x and y to numpy arrays so mask can be applied
    -> Mask applied to hide the last ten years from the numpy arrays
    -> Polynonimal fit calcualtion preformed 
    -> Both historial data and polynomial fit placed on the same chart to 
       allow comparison and see if fit is good

    Returns:
    matplotlib scatter graph -> historical data and polynomial fit on the same chart

    """

    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)


    #mask applied to subsample all but the last ten years of data
    cutoff_year = x.max() - 10

    mask_sample = x <= cutoff_year
    x_sample = x[mask_sample]
    y_sample = y[mask_sample]


    # Perform polynomial fit
    coefficients = np.polyfit(x, y, degree )
  

    # Create polynomial function
    p = np.poly1d(coefficients)

    plt.scatter(x_sample, y_sample, label='Historial Data')
    plt.plot(x, p(x), label=f'Polynomial order {degree}', color=color_1)
    plt.title("Spanish Fertility Data. Weighed Polynomial fits comparison.")
    plt.xlabel("Year")
    plt.ylabel("Fertility rate (period)")
    plt.legend()
    plt.show()
    

def plot_full_graph(x, y, degree ):
    """

    Creates a plot containing both historical and polynomial data over all years in the dataset

    Notes:
    -> x, y and degree passed into the function
    -> Polynonimal fit calcualtion preformed 
    -> Both historial data and polynomial fit placed on the same chart to 
       allow comparison and see if fit is good

    Returns:
    matplotlib scatter graph -> historical data and polynomial fit on the same chart. 
                                This chart can be compared with plot_prediction_graph
                                to see how good fit is

    """

    # Perform polynomial fit
    coefficients = np.polyfit(x, y, degree)
  
    # Create polynomial function
    p = np.poly1d(coefficients)

    plt.scatter(x, y, label='Historial Data')
    plt.plot(x, p(x), label=f'Polynomial order {degree}', color=color_1)
    plt.title("Spanish Fertility Data. Weighed Polynomial fits comparison.")
    plt.xlabel("Year")
    plt.ylabel("Fertility rate (period)")
    plt.legend()
    plt.show()


def polynomial_best_fit(x , y, sigma):
    """

    Creates model comparison by polynomial order graph comparing polynomial order and reduced chisquared

    Notes:
    -> x, y and sigma passed into the function
    -> Create a list for degrees, chi2_list, and chi2_reduced_list 
    -> Use a for loop to create linear fit and calulate chi_squared, degrees of freedom  and reduced chisquared
       -> chi squared = np.sum(((y - y_pred) / sigma)**2)
       -> degrees of freedom = len(x) - (degrees + 1)
       -> chi squared reduced =  chi_squared / degrees of freedom

    Returns:
    matplotlib scatter graph -> historical data and polynomial fit on the same chart. 
                                This chart can be compared with plot_prediction_graph
                                to see how good fit is

    """
   
   #lists used in for loop

   #for loop to incremeent degrees from 1 to 10 by increments of 0.5
    degrees = [x for x in np.arange(1, 10.5, 0.5)]

    chi2_list = []

    chi2_reduced_list = []

    for d in degrees:
        # Perform linear fit
        coefficients = np.polyfit(x, y, deg = d )

        # Create polynomial function
        p = np.poly1d(coefficients)
        y_pred = p(x)
        

        chi2 = np.sum(((y - y_pred) / sigma)**2)

        dof = len(x) - (d + 1)

        chi2_reduced = chi2/dof

        chi2_list.append(chi2)
        chi2_reduced_list.append(chi2_reduced)


    plt.plot(degrees, chi2_reduced_list, marker="o")
    plt.xlabel("Polynomial order (n)")
    plt.ylabel("Weighted x\u00b2 per degrees of freedom")
    plt.title("Model comparison by polynomial order ")
    plt.grid(True)
    plt.show()


#passed into functions above
x = spain_fertility_df[x_axis]
y = spain_fertility_df[y_axis]




if __name__ == "__main__":

    plot_prediction_graph(x, y, degree= 6)

    plot_full_graph(x, y, degree=6)

    polynomial_best_fit(x , y, 0.05*y)

