"""
Feature Engineering Module for Airline Customer Churn Prediction
Engineers 25+ features from flight activity and loyalty history data.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_raw_data(data_dir: Path = None):
    """Load raw CSV files."""
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data"

    flight_activity = pd.read_csv(data_dir / "Customer_Flight_Activity.csv")
    loyalty_history = pd.read_csv(data_dir / "Customer_Loyalty_History.csv")

    return flight_activity, loyalty_history


def create_month_index(row):
    """Convert Year/Month to month index (Jan 2017 = 1, Dec 2018 = 24)."""
    if row['Year'] == 2017:
        return row['Month']
    else:  # 2018
        return row['Month'] + 12


def engineer_flight_features(flight_activity: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate flight activity per customer (all 17 engineered features from spec).
    """
    # Create month index for recency calculation
    flight_activity = flight_activity.copy()
    flight_activity['month_index'] = flight_activity.apply(create_month_index, axis=1)

    # Aggregations per customer
    agg_features = flight_activity.groupby('Loyalty Number').agg({
        'Total Flights': ['sum', 'mean', 'std'],
        'Flights Booked': 'sum',
        'Flights with Companions': 'sum',
        'Distance': 'sum',
        'Points Accumulated': 'sum',
        'Points Redeemed': 'sum',
        'Dollar Cost Points Redeemed': 'sum',
    }).reset_index()

    # Flatten column names
    agg_features.columns = [
        'Loyalty Number',
        'total_flights_24m', 'avg_monthly_flights', 'flight_consistency',
        'total_flights_booked',
        'total_companion_flights',
        'total_distance_24m',
        'points_accumulated_24m',
        'points_redeemed_24m',
        'dollar_points_redeemed_24m'
    ]

    # Fill NaN for std (customers with constant flights)
    agg_features['flight_consistency'] = agg_features['flight_consistency'].fillna(0)

    # Calculate flight_trend_yoy (THE MOST IMPORTANT FEATURE)
    yearly_avg = flight_activity.groupby(['Loyalty Number', 'Year'])['Total Flights'].mean().unstack(fill_value=0)
    yearly_avg.columns = ['avg_2017', 'avg_2018']
    yearly_avg['flight_trend_yoy'] = yearly_avg['avg_2018'] - yearly_avg['avg_2017']
    yearly_avg = yearly_avg.reset_index()[['Loyalty Number', 'flight_trend_yoy']]

    agg_features = agg_features.merge(yearly_avg, on='Loyalty Number', how='left')

    # Companion ratio
    agg_features['companion_ratio'] = np.where(
        agg_features['total_flights_24m'] > 0,
        agg_features['total_companion_flights'] / agg_features['total_flights_24m'],
        0
    )

    # Average distance per flight
    agg_features['avg_distance_per_flight'] = np.where(
        agg_features['total_flights_24m'] > 0,
        agg_features['total_distance_24m'] / agg_features['total_flights_24m'],
        0
    )

    # Points utilization (capped at 1.0)
    agg_features['points_utilization'] = np.where(
        agg_features['points_accumulated_24m'] > 0,
        np.minimum(agg_features['points_redeemed_24m'] / agg_features['points_accumulated_24m'], 1.0),
        0
    )

    # Months active (months with flights > 0)
    months_active = flight_activity[flight_activity['Total Flights'] > 0].groupby('Loyalty Number').size()
    months_active = months_active.reset_index(name='months_active')
    agg_features = agg_features.merge(months_active, on='Loyalty Number', how='left')
    agg_features['months_active'] = agg_features['months_active'].fillna(0)

    # Recency of last flight (24 - max month_index where Total Flights > 0)
    last_flight = flight_activity[flight_activity['Total Flights'] > 0].groupby('Loyalty Number')['month_index'].max()
    last_flight = last_flight.reset_index(name='last_flight_month')
    agg_features = agg_features.merge(last_flight, on='Loyalty Number', how='left')
    agg_features['recency_last_flight'] = 24 - agg_features['last_flight_month'].fillna(0)
    agg_features.drop('last_flight_month', axis=1, inplace=True)

    # Redemption frequency
    redemption_months = flight_activity[flight_activity['Points Redeemed'] > 0].groupby('Loyalty Number').size()
    redemption_months = redemption_months.reset_index(name='redemption_months')
    agg_features = agg_features.merge(redemption_months, on='Loyalty Number', how='left')
    agg_features['redemption_months'] = agg_features['redemption_months'].fillna(0)
    agg_features['redemption_frequency'] = agg_features['redemption_months'] / 24
    agg_features.drop('redemption_months', axis=1, inplace=True)

    # Seasonal peak ratio (Jun, Jul, Aug, Dec flights / total)
    peak_months = [6, 7, 8, 12]
    peak_flights = flight_activity[flight_activity['Month'].isin(peak_months)].groupby('Loyalty Number')['Total Flights'].sum()
    peak_flights = peak_flights.reset_index(name='peak_flights')
    agg_features = agg_features.merge(peak_flights, on='Loyalty Number', how='left')
    agg_features['peak_flights'] = agg_features['peak_flights'].fillna(0)
    agg_features['seasonal_peak_ratio'] = np.where(
        agg_features['total_flights_24m'] > 0,
        agg_features['peak_flights'] / agg_features['total_flights_24m'],
        0
    )
    agg_features.drop('peak_flights', axis=1, inplace=True)

    # Drop intermediate column
    agg_features.drop('total_companion_flights', axis=1, inplace=True)

    return agg_features


def engineer_loyalty_features(loyalty_history: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from loyalty history.
    """
    df = loyalty_history.copy()

    # Create churn label (Cancellation Year is NOT NaN -> churned = 1)
    df['churn'] = df['Cancellation Year'].notna().astype(int)

    # Tenure in months (from enrollment to end of observation Dec 2018)
    df['tenure_months'] = (2018 - df['Enrollment Year']) * 12 + (12 - df['Enrollment Month']) + 12

    # Loyalty tier numeric (Star=1, Nova=2, Aurora=3)
    tier_map = {'Star': 1, 'Nova': 2, 'Aurora': 3}
    df['loyalty_tier_numeric'] = df['Loyalty Card'].map(tier_map)

    # Is promotion enrollee
    df['is_promotion_enrollee'] = (df['Enrollment Type'] == '2018 Promotion').astype(int)

    # CLV per year (annualized CLV)
    df['clv_per_year'] = np.where(
        df['tenure_months'] > 0,
        df['CLV'] / (df['tenure_months'] / 12),
        0
    )

    # Gender encoding (Male=1, Female=0)
    df['gender_encoded'] = (df['Gender'] == 'Male').astype(int)

    # Education ordinal encoding
    education_map = {
        'High School or Below': 1,
        'College': 2,
        'Bachelor': 3,
        'Master': 4,
        'Doctor': 5
    }
    df['education_encoded'] = df['Education'].map(education_map)

    # One-hot encode Marital Status (drop_first=True)
    marital_dummies = pd.get_dummies(df['Marital Status'], prefix='marital', drop_first=True)
    df = pd.concat([df, marital_dummies], axis=1)

    # One-hot encode Country (drop_first=True)
    country_dummies = pd.get_dummies(df['Country'], prefix='country', drop_first=True)
    df = pd.concat([df, country_dummies], axis=1)

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between flight activity and loyalty history.
    """
    # CLV per flight
    df['clv_per_flight'] = np.where(
        df['total_flights_24m'] > 0,
        df['CLV'] / df['total_flights_24m'],
        0
    )

    # Distance per salary (travel intensity relative to income)
    df['distance_per_salary'] = np.where(
        df['Salary'] > 0,
        df['total_distance_24m'] / df['Salary'],
        0
    )

    # Points per flight
    df['points_per_flight'] = np.where(
        df['total_flights_24m'] > 0,
        df['points_accumulated_24m'] / df['total_flights_24m'],
        0
    )

    # Miles per dollar CLV
    df['miles_per_dollar_clv'] = np.where(
        df['CLV'] > 0,
        df['total_distance_24m'] / df['CLV'],
        0
    )

    return df


def get_feature_columns():
    """Return the list of feature columns used in the model."""
    return [
        # Flight activity features
        'total_flights_24m',
        'avg_monthly_flights',
        'flight_consistency',
        'total_flights_booked',
        'total_distance_24m',
        'points_accumulated_24m',
        'points_redeemed_24m',
        'dollar_points_redeemed_24m',
        'flight_trend_yoy',
        'companion_ratio',
        'avg_distance_per_flight',
        'points_utilization',
        'months_active',
        'recency_last_flight',
        'redemption_frequency',
        'seasonal_peak_ratio',
        # Loyalty history features
        'tenure_months',
        'loyalty_tier_numeric',
        'is_promotion_enrollee',
        'clv_per_year',
        'gender_encoded',
        'education_encoded',
        'Salary',
        'CLV',
        # Marital status dummies
        'marital_Married',
        'marital_Single',
        # Country dummies
        'country_United States',
        # Interaction features
        'clv_per_flight',
        'distance_per_salary',
        'points_per_flight',
        'miles_per_dollar_clv',
    ]


def engineer_features(flight_activity: pd.DataFrame, loyalty_history: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Returns a DataFrame with all engineered features and the churn label.
    """
    # Engineer flight features
    flight_features = engineer_flight_features(flight_activity)

    # Engineer loyalty features
    loyalty_features = engineer_loyalty_features(loyalty_history)

    # Merge on Loyalty Number
    df = loyalty_features.merge(flight_features, on='Loyalty Number', how='left')

    # Fill any NaN values from merge (customers with no flight activity)
    flight_cols = flight_features.columns.drop('Loyalty Number')
    df[flight_cols] = df[flight_cols].fillna(0)

    # Create interaction features
    df = create_interaction_features(df)

    return df


def prepare_model_data(df: pd.DataFrame):
    """
    Prepare data for model training.
    Returns X (features), y (target), and feature names.
    """
    feature_cols = get_feature_columns()

    # Check which columns exist
    available_cols = [col for col in feature_cols if col in df.columns]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        # Add missing columns with 0
        for col in missing_cols:
            df[col] = 0

    X = df[feature_cols].copy()
    y = df['churn'].copy()

    # Handle any remaining NaN
    X = X.fillna(0)

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)

    return X, y, feature_cols


def main():
    """Run feature engineering and save processed data."""
    print("Loading raw data...")
    data_dir = Path(__file__).parent.parent / "data"
    flight_activity, loyalty_history = load_raw_data(data_dir)

    print(f"Flight Activity: {flight_activity.shape}")
    print(f"Loyalty History: {loyalty_history.shape}")

    print("\nEngineering features...")
    df = engineer_features(flight_activity, loyalty_history)

    print(f"\nProcessed data shape: {df.shape}")
    print(f"Churn rate: {df['churn'].mean():.2%}")

    # Save processed features
    output_path = data_dir / "processed_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved processed features to {output_path}")

    # Prepare model data
    X, y, feature_names = prepare_model_data(df)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")

    return df, X, y, feature_names


if __name__ == "__main__":
    main()
