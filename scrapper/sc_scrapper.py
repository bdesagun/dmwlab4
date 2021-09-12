from requests.packages.urllib3.exceptions import InsecureRequestWarning
from time import sleep
from pathlib import Path
from datetime import datetime
import sqlite3
from time import strptime
import os
import pickle
from bs4 import BeautifulSoup
import re
from math import ceil
import numpy as np
import pandas as pd
import time
import requests
import MySQLdb
from tqdm import tqdm
import pymysql
import getpass
from sqlalchemy import exc
from sqlalchemy import create_engine
pymysql.install_as_MySQLdb()

requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


np.random.seed(0)


server = 'scdecsisions.cvd0q1suowjz.ap-southeast-1.rds.amazonaws.com'
db = 'sc_decsisions'
account = 'msds2022'
password = 'Rocksalt1'


def get_soup(url_):
    """Return BeautifulSoup object from input url."""
    res = requests.get(url_,
                       headers={'User-agent':
                                ('Mozilla/5.0 (Windows NT 6.1; WOW64) '
                                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/47.0.2526.111 Safari/537.36')},
                       verify=False)
    # print(res.content)
    return BeautifulSoup(res.content, features='lxml', from_encoding='utf-8')


def get_month_links(conn):
    """Returns DataFrame of unprocessed SC month-year links"""
    who = getpass.getuser()
    query = f"""
        select * from months
        where date =     
        (select min(date)
        from months
        where status =0 or
        (status = 99 and who = '{who}'))
        """

    try:
        out = pd.read_sql(query, conn)

    except exc.ProgrammingError:

        months_source = 'https://elibrary.judiciary.gov.ph/'
        soup = get_soup(months_source)
        month_urls = soup.find_all(href=True)
        month_urls = [e['href'] for e in month_urls]
        month_urls = [re.findall('.*docmonth.*', e)[0]
                    or e in month_urls if re.findall('.*docmonth.*', e) != []]

        def month_f(x): return strptime(x.split('/')[5], '%b').tm_mon
        def year_f(x): return x.split('/')[6]
        months_df = pd.DataFrame({'url': month_urls})
        months_df['month'] = months_df['url'].apply(month_f)
        months_df['year'] = months_df['url'].apply(year_f)

        months_df['date'] = pd.to_datetime(months_df['month'].astype(
            str) + '-1-' + months_df['year'].astype(str), )
        months_df['status'] = 0
        months_df['who'] = ''
        months_df = months_df.sort_values(by=['year', 'month'])
        months_df.to_sql('months', conn, index=False, if_exists='replace')
        out = pd.read_sql(query, conn)

    mlink = out['url'].values[0]
    update_q = f"""
    update months
    set status = 99
        ,who = '{who}'
    where url = '{mlink}'
    """
    conn.execute(update_q)

    return out


def extract_cases(conn):
    """Download SC cases to a html file."""
    months_df = get_month_links(conn)
    url = months_df.iloc[0, 0]
    # create output directory
    file_out_dir = f"cases/{('/').join(url.split('/')[5:7][::-1])}/"
    Path(file_out_dir).mkdir(parents=True, exist_ok=True)

    soup = get_soup(url)
    cases = soup.select('#container_title ul li')
    out_list = []
    t = tqdm(cases, desc='Setting things up')
    for case in t:
        c_link = case.select('a')[0]['href']
        c_number = case.select('strong')[0].text
        c_title = case.select('small')[0].text
        c_dt = re.findall(r'\w\w\w+ \d{2}, \d{4}', case.text)[0]
        c_dt_object = datetime.strptime(c_dt, '%B %d, %Y')

        file_z = re.sub('\.', '', c_number)
        file_y = re.sub('[^0-9a-zA-Z]+', '_', file_z)
        file = re.sub('_+', '_', file_y).upper() + '.html'
        full_file = f'{file_out_dir}{file}'

        # Random Sleep time max 2 Seconds
        sleep(np.random.rand() * 2)
        # Get full case and save to local
        t.set_description(f"Extacting {full_file}")
        with open(full_file, 'w+', encoding="utf-8") as f:
            f.write(get_soup(c_link).prettify())

        out_dict = {
            'parent': url,
            'url': c_link,
            'number': c_number,
            'title': c_title,
            'date': c_dt_object,
            'local': full_file,
            'extact_dt': datetime.now(),
            'extract_by': getpass.getuser()
        }
        out_list.append(out_dict)

    df_sql = pd.DataFrame(out_list)
    df_sql.to_sql('cases', conn, index=False, if_exists='append')

    update_q = f"""
            update months
            set status = 200
            where url = '{url}'
            """
    conn.execute(update_q)


def main():
    """Execute extract_cases function."""
    engine = create_engine(
        f"mysql+mysqldb://{account}:{password}@{server}/{db}?charset=utf8mb4")
    conn = engine.connect().execution_options(autocommit=True)

    extract_cases(conn)

    conn.invalidate()
    conn.close()
