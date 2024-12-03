import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.special import gamma

class Data(pd.DataFrame):
    """
    Data class to generate data for the survival analysis - inherits from pd.DataFrame

    Attributes:
        df (pd.DataFrame): DataFrame with the data
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return super().__repr__()

    def generate_data(self, start_date: datetime, years_to_simulate: int, number_of_trips: int, trip_duration_distribution: np.array, shrinkage_rate: float):
        """
        Generate data from a distribution
        """
        rental_dates = [start_date + timedelta(days=np.random.randint(0, 365 * years_to_simulate)) for _ in range(number_of_trips)]
        return_dates = [rental_date + timedelta(days=trip_duration) for rental_date, trip_duration in zip(rental_dates, trip_duration_distribution)]
        trip_returned = self.generate_binomial_distribution(1, 1 - shrinkage_rate, number_of_trips)
        trip_durations = np.clip(trip_duration_distribution, a_min=0, a_max=None) 
        #trip_durations[trip_returned == 0] = trip_durations.max()
        #trip_durations = np.where(np.isnan(trip_durations), np.nanmax(trip_durations), trip_durations)
        self['trip_id'] = range(number_of_trips)
        self['rental_date'] = rental_dates
        self['return_date'] = return_dates
        self['trip_duration'] = trip_durations
        self['is_returned'] = trip_returned
    
    @staticmethod
    def generate_normal_distribution(mean: float, deviation: float, sample_size: int):
        """
        Generate data from a normal distribution
        """
        return np.random.normal(mean, deviation, sample_size)

    @staticmethod
    def generate_lognormal_distribution(mean: float, std: float, sample_size: int):
        """
        Generate data from a lognormal distribution
        """
        # Calculate mu for the log-normal distribution
        mu = np.log(mean) - (std ** 2) / 2
        # Create a log-normal distribution
        trip_duration_distribution = np.random.lognormal(mu, std, sample_size)
        return trip_duration_distribution
    
    @staticmethod
    def generate_weibull_distribution( mean: float, shape: float, sample_size: int):
        """
        Generate data from a Weibull distribution
        """
        # Calculate scale parameter (lambda)
        lambda_ = mean / (gamma(1 + 1 / shape))
        # Generate a Weibull distribution
        trip_duration_distribution = np.random.weibull(shape, sample_size) * lambda_
        return trip_duration_distribution
    @staticmethod
    def generate_exponential_distribution(mean: float, sample_size: int):
        """
        Generate data from an exponential distribution
        """
        return np.random.exponential(mean, sample_size)
    
    @staticmethod
    def generate_poisson_distribution(mean: float, sample_size: int):
        """
        Generate data from a Poisson distribution
        """
        return np.random.poisson(mean, sample_size)
    
    @staticmethod
    def generate_binomial_distribution(n_trials: int, shrinkage_rate: float, sample_size: int):
        """
        Generate data from a binomial distribution
        """
        return np.random.binomial(n_trials, shrinkage_rate, sample_size)

    
        

