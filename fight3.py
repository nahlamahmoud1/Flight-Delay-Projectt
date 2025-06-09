import json
import numpy as np
import pandas as pd
import multiprocessing
from numpy import random
import datetime
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler, LabelEncoder, normalize
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# -------------------- Read data --------------------
data = pd.read_csv("DelayedFlights.csv")

# -------------------- Lambda Filtering --------------------
filtered_data = data.loc[lambda df: (df['CarrierDelay'].notna()) & (df['CarrierDelay'] > 100)]

# -------------------- Slicing --------------------
sliced_data = data[1:10]

# -------------------- Create and Manipulate Array --------------------
delays = np.array(data['CarrierDelay'].fillna(0))
shape = delays.shape
dimension = delays.ndim

# -------------------- Filtering --------------------
w = np.where(delays <= 10)
w2 = delays[delays >= 100]

# -------------------- Sorting --------------------
sorted_delays = np.sort(delays)

# -------------------- Series --------------------
flight_nums = pd.Series(data['FlightNum'])
iloc_flight = flight_nums.iloc[0]
head_flights = flight_nums.head(2)
tail_flights = flight_nums.tail(2)
flight_index = flight_nums.index

# -------------------- DataFrame manipulations --------------------
df = pd.DataFrame(data)
delay_columns = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
df['TotalDelay'] = df[delay_columns].sum(axis=1, skipna=True)

# -------------------- Categorical --------------------
df['Cancelled'] = pd.Categorical(df['Cancelled'], categories=[0, 1], ordered=True)

# -------------------- Mapping --------------------
df['LongDelay'] = df['CarrierDelay'].map(lambda x: 1 if pd.notna(x) and x >= 60 else 0)

# -------------------- DateTime --------------------
t = datetime.datetime(2008, 1, 1, 0, 0)
td2 = pd.Timedelta(minutes=15)
dr = pd.date_range(t, periods=len(data), freq=td2)
data['schedule'] = dr
td3 = pd.Timedelta(hours=1)
data['schedule'] = data['schedule'] + td3

# -------------------- Pipeline for Preprocessing --------------------
numeric_features = ['CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
categorical_features = ['Origin'] if 'Origin' in data.columns else []

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Apply pipeline
processed_data = pipeline.fit_transform(data)

# Optional: Convert to DataFrame
processed_df = pd.DataFrame(processed_data.toarray() if hasattr(processed_data, "toarray") else processed_data)

# Add normalized total delay to df
delay_data = df[delay_columns].fillna(0)
normalized_delays = normalize(delay_data, norm='l2')
df['NormalizedTotalDelay'] = np.sum(normalized_delays, axis=1)

# -------------------- Label Encoding --------------------
if 'Origin' in data.columns:
    data['Origin'] = data['Origin'].astype(str)
    encoder = LabelEncoder()
    data['OriginEncoded'] = encoder.fit_transform(data['Origin'])

# -------------------- JSON Save and Read ----
json_path = "DelayedFlights.json"
data.to_json(json_path, orient="records")
json_data = pd.read_json(json_path)

# -------------------- Multiprocessing --------------------
def process_data(part, queue):
    column_sum = part['CarrierDelay'].fillna(0).sum()
    print(f"Processed part:\n{part[['CarrierDelay']].head()}\nSum of 'CarrierDelay': {column_sum}")
    queue.put(column_sum)

if __name__ == "__main__":
    data = pd.read_csv("DelayedFlights.csv")
    mid_index = len(data) // 2
    part1 = data.iloc[:mid_index]
    part2 = data.iloc[mid_index:]

    queue = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=process_data, args=(part1, queue))
    p2 = multiprocessing.Process(target=process_data, args=(part2, queue))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    result1 = queue.get()
    result2 = queue.get()

    print(f"Sum from part 1: {result1}")
    print(f"Sum from part 2: {result2}")
    print(f"Total CarrierDelay Sum: {result1 + result2}")
    print("Done!")