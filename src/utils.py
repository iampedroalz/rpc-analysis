import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from lifelines import KaplanMeierFitter
import streamlit as st
from src.data import Data

class StreamlitApp():
    def __init__(self):
        """
        Initializes the StreamlitApp class.

        This constructor sets up the initial state of the application, including
        the data object that will be used for generating trip duration distributions.

        Attributes:
            data (Data): An instance of the Data class used to generate trip data.
        """
        self.data = Data()

    def set_app_title(self):
        """
        Sets the title of the Streamlit application.
        """
        st.title("Data Science Challenge - Distribution Analysis")

    def input_parameters(self):
        """
        This section collects user input parameters for the simulation, including distribution type, mean trip duration, standard deviation, number of trips, shrinkage rate, start date, and years to simulate.
        """
        
        self.distribution_type = st.sidebar.selectbox(
            "Select Distribution Type", ["Normal", "Log-Normal", "Weibull", "Exponential"]
        )
        self.mean_trip_duration = st.sidebar.selectbox(
            "Mean Trip Duration (days)", options=[100], index=0
        )
        self.std_trip_duration = st.sidebar.slider(
            "Standard Deviation of Trip Duration", min_value=0.0, max_value=1.0, value=0.2
        )
        self.number_of_trips = st.sidebar.slider(
            "Number of Trips", min_value=1, max_value=20000, value=10000
        )
        self.shrinkage_rate = st.sidebar.selectbox(
            "Trips that are lost", options=[0.05], index=0
        )
        self.start_date = st.sidebar.date_input("Start Date", value=datetime(2021, 1, 1))
        self.years_to_simulate = st.sidebar.slider(
            "Years to Simulate", min_value=1, max_value=10, value=2
        )
        self.deviation_trip_duration = self.std_trip_duration * self.mean_trip_duration

    def generate_trip_distribution(self):
        """
        Generates the trip duration distribution based on the selected distribution type.
        
        """
        if self.distribution_type == "Normal":
            self.trip_duration_distribution = self.data.generate_normal_distribution(
                mean=self.mean_trip_duration,
                deviation=self.deviation_trip_duration,
                sample_size=self.number_of_trips,
            )
        elif self.distribution_type == "Log-Normal":
            self.trip_duration_distribution = self.data.generate_lognormal_distribution(
                mean=self.mean_trip_duration,
                std=self.std_trip_duration,
                sample_size=self.number_of_trips,
            )
        elif self.distribution_type == "Weibull":
            shape = st.number_input("Weibull Shape Parameter", value=2.5)
            self.trip_duration_distribution = self.data.generate_weibull_distribution(
                mean=self.mean_trip_duration, shape=shape, sample_size=self.number_of_trips
            )
        elif self.distribution_type == "Exponential":
            self.trip_duration_distribution = self.data.generate_exponential_distribution(
                mean=self.mean_trip_duration, sample_size=self.number_of_trips
            )

    def display_generated_data(self):
        """Displays the generated data based on the selected distribution and shows the count of returned trips."""
        st.subheader("Generated Data")
        with st.expander("Data Info", expanded=False):
            st.write("""
                - **trip_id**: Unique identifier for each trip.
                - **rental_date**: The date when the trip was initiated.
                - **return_date**: The date when the trip was completed.
                - **trip_duration**: The duration of the trip in days.
                - **is_returned**: Indicates whether the trip was returned (1) or lost (0).
            """)
        self.df = Data()
        self.df.generate_data(
            self.start_date,
            self.years_to_simulate,
            self.number_of_trips,
            self.trip_duration_distribution,
            self.shrinkage_rate,
        )
        st.write(self.df)
        st.write(f"Returned trips")
        st.dataframe(self.df["is_returned"].value_counts())

    def plot_histogram(self):
        """ Plot histogram in streamlit """
        st.subheader("Histogram of Trip Durations")
        plt.figure(figsize=(10, 5))
        plt.hist(
            self.df[self.df["is_returned"] == 1]["trip_duration"],
            bins=30,
            edgecolor="black",
            alpha=0.7,
            label="Returned Trips",
        )
        plt.hist(
            self.df[self.df["is_returned"] == 0]["trip_duration"],
            bins=30,
            edgecolor="black",
            alpha=0.7,
            label="Lost Trips",
        )
        
        # Calculate mean and standard deviation
        mean_duration = np.mean(self.df["trip_duration"])
        std_duration = np.std(self.df["trip_duration"])

        # Add vertical lines for mean and std
        plt.axvline(mean_duration, color='blue', linestyle='dashed', linewidth=1, label=f'Mean: {mean_duration:.2f}')
        plt.axvline(mean_duration + std_duration, color='green', linestyle='dashed', linewidth=1, label=f'Mean + 1 Std: {mean_duration + std_duration:.2f}')
        plt.axvline(mean_duration - std_duration, color='red', linestyle='dashed', linewidth=1, label=f'Mean - 1 Std: {mean_duration - std_duration:.2f}')

        plt.legend()
        plt.title(f"Histogram of Trip Durations ({self.distribution_type})")
        plt.xlabel("Trip Duration (days)")
        plt.ylabel("Frequency")
        st.pyplot(plt)

    def plot_kmf_curve(self):
        """ Plot Kaplan-Meier curve """
        st.subheader("Kaplan-Meier Survival Curve")
        kmf = KaplanMeierFitter()  # Store kmf in the class
        kmf.fit(durations=self.df["trip_duration"], event_observed=self.df["is_returned"])
        self.kmf = kmf 
        plt.figure(figsize=(10, 5))
        self.kmf.plot_survival_function()
        mean_duration = np.mean(self.trip_duration_distribution)
        mean_kmf_value = self.kmf.survival_function_at_times(mean_duration).values[0]
        plt.axvline(mean_duration, color="r", linestyle="dashed", linewidth=1, label=f"Mean: {mean_kmf_value:.2f}")
        plt.text(
            mean_duration + 0.2 * max(self.df["trip_duration"]),
            mean_kmf_value,
            f"Shrinkage rate at\n 100 days: \n{(1-mean_kmf_value)*100:.2f}%",
            color="r",
            fontsize=12,
            ha="center",
            va="bottom",
        )
        plt.ylim(0, 1)
        plt.title("Kaplan-Meier Survival Function")
        plt.xlabel("Days")
        plt.ylabel("Survival Probability")
        st.pyplot(plt)

    def estimate_pool_size(self):
        """
        Estimate and visualize the pool size over time based on rental rates, return probabilities, and shrinkage.
        """
        st.subheader("Pool Size Over Time")
        
        # Parameters for simulation
        initial_pool_size = st.number_input("Initial Pool Size", min_value=1, max_value=10000, value=1000)
        rental_rate = st.number_input("Rentals at day 0", min_value=1, max_value=1000, value=50)

        # Calculate PDF from trip duration data
        pdf, _, _, max_day = calculate_pdf(self.df)

        # Initialize variables for simulation
        available_assets = initial_pool_size
        assets_out = 0
        assets_lost = 0
        pool_sizes = [available_assets]

        # Day 1 - start simulation, renting assests
        rented_today = rental_rate
        available_assets -= rented_today
        assets_out += rented_today
        pool_sizes.append(available_assets)

        # Simulate pool size over time
        for day in range(1, max_day):
            return_probability_day = pdf[day]
            survival_rate = self.kmf.survival_function_at_times(day).values[0]            
            # Calculate losses and returns
            lost_today = assets_out * (1 - survival_rate) * return_probability_day
            assets_lost += lost_today            
            # Update assets out and available assets
            returned_today = assets_out * return_probability_day * survival_rate
            assets_out -= (lost_today + returned_today)
            available_assets += returned_today
            pool_sizes.append(available_assets)
        
        self.pool_sizes = pool_sizes
    
    def plot_pool_size(self):
        """ Plot pool size """
        plt.figure(figsize=(10, 5))
        plt.plot(self.pool_sizes)
        # add the max pool size as text to the plot
        plt.title("Estimated Pool Size Over Time")
        plt.xlabel("Days")
        plt.ylabel("Pool Size")
        #plt.ylim(0, max(pool_sizes))  # Limit y-axis from 0 to max pool size
        st.pyplot(plt)

    def run(self):
        """ Streamloit app """
        self.set_app_title()
        self.input_parameters()
        self.generate_trip_distribution()
        self.display_generated_data()
        self.plot_histogram()
        self.plot_kmf_curve()
        self.estimate_pool_size()
        self.plot_pool_size()





def calculate_pdf(df):
    """
    Calculate the probability density function (PDF) from the trip duration data.

    Args:
        df (pd.DataFrame): DataFrame containing trip duration data.

    Returns:
        pdf (np.array): The normalized probability density function.
        bin_edges (np.array): The bin edges for the PDF.
        bin_width (float): The width of the bins used in the histogram.
        max_day (int): The maximum day value from the trip durations.
    """
    max_day = int(np.max(df["trip_duration"]))
    H, X1 = np.histogram(df["trip_duration"], bins=max_day, density=True)
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H) * dx
    pdf = np.gradient(F1, dx)
    pdf /= np.sum(pdf)
    return pdf, X1[1:], dx, max_day
