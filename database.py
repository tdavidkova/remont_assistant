import pandas as pd
import sqlite3

def create_database_from_csv(csv_file, db_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)
    # Convert 'Date' column to date format and remove hour part
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y').dt.date

    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_file)

    # Write the data to a new table in the database
    data.to_sql('remont', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

def connect_to_database(db_file):
    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_file)
    return conn

def query_data(conn):
    # Query data from the table
    cursor = conn.cursor()
    cursor.execute('SELECT min(Date) FROM remont')
    rows = cursor.fetchall()
    for row in rows:
        print(row)

ddl = """
The column 'Date' is formatted 'MM/DD/YYYY" and denotes the date when the cost was paid
The column 'Activity' is the type of the cost - high-level aggregation used - e.g. demolition, furniture, etc.
The column 'Subactivity' provides some fine-grained aggregation compared to 'Activity'
The column 'Payee' gives the name of the person/entity/shop who received the paiments related to the cost.
The column 'Payer' provides the name of the person who paid for the activity.
"""

description = """
The database consists of a single table that contains information on the costs related to the reconstruction works in an apartment.
The language used is Bulgarian. The currency is Bulgarian Lev (BGN).

{ddl}
"""


if __name__ == "__main__":
    create_database_from_csv('remont.csv', 'db.db')
    conn = connect_to_database('db.db')
    query_data(conn)
    conn.close()

    