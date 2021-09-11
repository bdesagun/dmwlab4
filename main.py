# from scraper.sc_scrapper import main

# for i in range(1000):
#     main()

#%%


import pandas as pd
import re
from sqlalchemy import exc
from sqlalchemy import create_engine
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb


def parsed_cases(query):
    server = 'scdecsisions.cvd0q1suowjz.ap-southeast-1.rds.amazonaws.com'
    db = 'sc_decsisions'
    account = 'msds2022'
    password = 'Rocksalt1'

    engine = create_engine(
        f"mysql+mysqldb://{account}:{password}@{server}/{db}?charset=utf8mb4")
    with  engine.connect().execution_options(autocommit=True) as conn:
        df_cases = pd.read_sql(query,conn)


    data = [pd.read_pickle(x.replace('html','pkl'))
                for x in df_cases.local.values]
    
    out = pd.concat(data).reset_index(drop=True)
    out['body'].replace('\\n','\n', inplace=True)
    return out

query = """
    select local
    from cases
    where regexp_like(title,'(?i)labor')
    limit 10
    """
df = parsed_cases(query)

# %%


from pathlib import Path
import pandas as pd


path = './cases/1996/Apr'
pattern_filter = '(?i).*family code.*'
# pattern_filter = '(?i).*petitioners.*'

out = [] 
for f in Path(path).glob('**/*.pkl'):
    df = pd.read_pickle(f)
    # Do Something
    dfout = df[df['body'].str.match(pattern_filter)]
    if dfout.shape[0] >0:
        out.append(dfout)

dfdata = pd.concat(out).reset_index(drop=True)
dfdata


# %%
