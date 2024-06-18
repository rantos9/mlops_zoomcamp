import boto3
import pickle
import argparse
import pandas as pd

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

parser = argparse.ArgumentParser()
parser.add_argument('year')
parser.add_argument('month')
args = parser.parse_args()

year = int(args.year)
month = int(args.month)

# year = 2023
# month = 3
categorical = ['PULocationID', 'DOLocationID']

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

df['y_pred'] = y_pred

output_file = f'df_result_{year:04d}_{month:02d}.parquet'
df_result = df[['ride_id', 'y_pred']]

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)

print(round(df_result['y_pred'].mean(), 2))

boto3.client('s3').upload_file(output_file, 'mlops-zoomcamp-renan-9999', output_file)