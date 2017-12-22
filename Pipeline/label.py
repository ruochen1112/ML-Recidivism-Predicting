import pandas as pd
from ..setup_environment import set_connection

conn = set_connection('config.json')

def get_date_range(df,label, year):
    grouped = df.groupby('mni_no')
    for name, group in grouped:
        pre_release = 0
        pre_index = 0
        for index, row in group.iterrows():
            if pre_release != 0:
                cur_booking = row['booking_date']
                if (cur_booking-pre_release).days < 365:
                    if(pre_release.year == year):
                        df.set_value(pre_index,label, 1)
            pre_release = row['release_date']
            pre_index = index

def main():
    df = pd.read_sql("select * from booking_temp", conn)
    df['booking_date'] = pd.to_datetime(df['booking_date'])
    df['release_date'] = pd.to_datetime(df['release_date'])
    df = df.sort_values(by='booking_date')
    df['recidivism2010'] = 0
    df['recidivism2011'] = 0
    df['recidivism2012'] = 0
    df['recidivism2013'] = 0
    df['recidivism2014'] = 0
    df['recidivism2015'] = 0
    get_date_range(df, 'recidivism2010', 2010)
    get_date_range(df, 'recidivism2011', 2011)
    get_date_range(df, 'recidivism2012', 2012)
    get_date_range(df, 'recidivism2013', 2013)
    get_date_range(df, 'recidivism2014', 2014)
    get_date_range(df, 'recidivism2015', 2015)
    df.to_csv('out.csv')
                
if __name__ == "__main__":
   main()


