from time import sleep
from pathlib import Path
from collections import Counter
import datetime
from bs4 import BeautifulSoup
import MySQLdb
import pandas as pd
import re
from sqlalchemy import exc
from sqlalchemy import create_engine
import pymysql
pymysql.install_as_MySQLdb()


def to_df_flg(url):
    server = 'scdecsisions.cvd0q1suowjz.ap-southeast-1.rds.amazonaws.com'
    db = 'sc_decsisions'
    account = 'msds2022'
    password = 'Rocksalt1'

    query = f"""
        update cases
        set to_df=200
        where local = '{url}'
    """

    engine = create_engine(
        f"mysql+mysqldb://{account}:{password}@{server}/{db}?charset=utf8mb4")
    with engine.connect().execution_options(autocommit=True) as conn:
        # df.to_sql('case_content',conn, index=False, if_exists='append')
        conn.execute(query)


server = 'scdecsisions.cvd0q1suowjz.ap-southeast-1.rds.amazonaws.com'
db = 'sc_decsisions'
account = 'msds2022'
password = 'Rocksalt1'

q = """
select url, local, date, number, title
from cases
where to_df=0
"""

engine = create_engine(
    f"mysql+mysqldb://{account}:{password}@{server}/{db}?charset=utf8mb4")
with engine.connect().execution_options(autocommit=True) as conn:
    # df.to_sql('case_content',conn, index=False, if_exists='append')
    df_cases = pd.read_sql(q, conn)


# url = 'cases/1996/Jan/GR_NO_114333.html'
# u=url
df_cols = ['date_yyyymmdd', 'division', 'case no', 'case title', 'petitioner',
           'respondent', 'ponente', 'body', 'dispositive portion',
           'disposition', 'granted', 'denied', 'dismissed',
           'remanded_referred', 'other justices', 'law', 'provision',
           'footnote', 'url']

ctr = 1
for i, r in df_cases.iterrows():
    url = r['local']
    u = r['url']
    date = r['date']
    case_no = r['number']
    case_no = case_no.strip().lower()
    case_title = r['title']
    case_title = case_title.strip().lower()

    print(f'Proessing {url}')
    all_text_df = pd.DataFrame(columns=df_cols)
    with open(f"../{url}", 'r+', encoding="ISO-8859-1") as f:
        x = f.read()
    case_soup = BeautifulSoup(x, features='lxml')

    # Getting the date and division
    header = case_soup.find_all('h2')
    header = [h.text for h in header]

    # date = header[1]
    # date = date[date.find(',')+2:-2]
    # date = re.sub(r',|[^\w ]', '', date).strip()
    # date = datetime.datetime.strptime(date, '%B %d %Y')

    division = header[0].strip()

    # Getting the case number and title

    case_details = case_soup.find('title').text
    case_details = re.sub(r'(?i)\s*D\s*E\s*C\s*I\s*S\s*I\s*O\s*N.*|'
                          '\s*R\s*E\s*S\s*O\s*L\s*U\s*T\s*I\s*O\s*N.*', '',
                          case_details)
    # removing excess whitespaces
    case_details = re.sub(r'\s+', ' ', case_details)
    # removing standardizing v and vs
    case_details = re.sub(r', V\.', ', VS.', case_details)
    case_details = case_details.split(' - ', maxsplit=1)
    # case_no = case_details[0].strip().lower()
    # case_title = case_details[1].strip().lower()
    split_case_title = case_title

    try:
        petitioner, respondent = split_case_title.split('vs.')[0:2]
        petitioner = petitioner.strip()
        respondent = respondent.strip()
        if petitioner.find('petitioner') == -1:
            petitioner, respondent = respondent, petitioner
        elif respondent.find('petitioner') == -1:
            pass
        else:
            petitioner = respondent = None
    except:
        petitioner, respondent = None, None

    try:
        ponente = case_soup.find('strong').text
        ponente = re.sub('J\.*:$', 'J.', ponente)
        ponente = ponente.lower().strip()
    except:
        ponente = None

    # Getting the entire body (should start from the top,
    # right after the name of the ponente, until the final "SO ORDERED")
    full_body = case_soup.find_all(True, {'align': ['justify', 'JUSTIFY']})
    full_body = [b.text for b in full_body]
    full_body = [re.sub('\r\n', ' ', b) for b in full_body]
    full_body = [re.sub(r'\[', ' [', b) for b in full_body]
    full_body = [re.sub(r'  ', ' ', b) for b in full_body]
    full_body = [b.lower() for b in full_body]
    if len(full_body) > 1:  # merging the body in case na-split while extracting
        full_body = [' '.join(full_body)]
    full_body = str(full_body)

    # Getting the index (marker) of the end of the case
    # see if the next word is a SC justice
    so_ordered = full_body.rfind('so ordered')
    # case_soup.select('hr')
    # Getting the main body based on the marker
    main_body = full_body[2:so_ordered+11]  # we start at 2 to remove ['

    # Getting the opinion of other justices and the footnotes
    full_footnote = full_body[so_ordered+11:]
    full_footnote = full_footnote.split('\\n')
    other_j = full_footnote[0]
    footnote_dump = '\n'.join(full_footnote[1:])

    # Getting the dispositive portion based on the marker
    dispositive = main_body[-3000:]
    if dispositive.find('wherefore') != -1:
        dispositive = dispositive[dispositive.find('wherefore'):]
    elif dispositive.find('in view whereof') != -1:
        dispositive = dispositive[dispositive.find('in view whereof'):]
    elif dispositive.find('premises considered') != -1:
        dispositive = dispositive[dispositive.find('premises considered'):]
    elif dispositive.find('in view of the foregoing premises') != -1:
        dispositive = dispositive[dispositive.find(
            'in view of the foregoing premises'):]
    elif dispositive.find('considering the foregoing disquisitions') != -1:
        dispositive = dispositive[dispositive.find(
            'considering the foregoing disquisitions'):]
    elif main_body[-1000:].find('accordingly, the petition is'):
        dispositive = dispositive[dispositive.find(
            'accordingly, the petition is'):]
    elif main_body[-1000:].find('accordingly, in line with the '
                                'foregoing disquisition, the petition is'):
        dispositive = dispositive[dispositive.find(
            'accordingly, in line with the foregoing disquisition, '
            'the petition is'):]
    elif main_body[-1000:].find('in light of the all the foregoing'):
        dispositive = dispositive[dispositive.find(
            'in light of the all the foregoing,'):]
    elif main_body[-1000:].find('considering the foregoing'):
        dispositive = dispositive[dispositive.find(
            'considering the foregoing,'):]

    # appeal|petition|decision|court of appeals|\w+ trial court
    # Getting the disposition based on the marker
    disposition = re.findall(
        'affirm|affirms|affirmed|deny|denies|denied|dismiss|dismissed|grant'
        '|granted|modify|modified|modification|remand|remanded|refer|referred'
        '|set aside|sets aside|reverses|reversed|reinstate|reinstated',
        dispositive)
    disposition_kw = {'affirm': 'affirmed', 'affirms': 'affirmed',
                      'deny': 'denied', 'denies': 'denied',
                      'dismiss': 'dismissed',
                      'grant': 'granted', 'modify': 'modified',
                      'modification': 'modified', 'remand': 'remanded',
                      'refer': 'referred', 'reverses': 'reversed',
                      'reinstate': 'reinstated', 'sets aside': 'set aside'}
    disposition = [disposition_kw[d]
                   if d in disposition_kw.keys() else d for d in disposition]

    #Granted, Denied
    try:
        if disposition[0] == 'dismissed':
            granted, denied, dismissed, remanded_referred = False, False, \
                True, False
        if disposition[0] == 'granted' or disposition[0] == 'set aside':
            granted, denied, dismissed, remanded_referred = True, False, \
                False, False
        elif disposition[0] == 'denied' or disposition[0] == 'deny' or \
                disposition[0] == 'denies' or disposition[0] == 'affirmed':
            granted, denied, dismissed, remanded_referred = False, True,\
                False, False

    except Exception as e:
        granted, denied, dismissed, remanded_referred = None, None, None, None

    # Getting the laws cited
    law = Counter(re.findall('(?i)family code|domestic adoption act|'
                             'inter-country-adoption act|'
                             'intercountry adoption act', main_body))

    # Getting the provisions cited in the case
    provision = Counter(re.findall('(?i) article \d+| art. \d+| art \d+|'
                                   ' section \d+| sec. \d+| sec \d+',
                                   main_body))

    #     #To Brian: Is there a way to track kung anong provision yung kadikit
    #     #ng `law` na cinite? Iba-iba ata format so baka hindi kaya hehe
    #     #so baka ok nang ganito kiwalay kunin tapos domain knowledge
    #     #na lang yung magmemake sense nung entries sa `provision

    case_text_df = pd.DataFrame(dict(zip(df_cols, [date, division, case_no,
                                         case_title, petitioner, respondent,
                                         ponente, main_body, dispositive, [
                                             disposition], granted, denied,
        dismissed, remanded_referred,
        other_j, [law], [provision],
        footnote_dump, u])))
    case_text_df.to_pickle(f"../{url.replace('.html','.pkl')}")
    if not (ctr % 200):
        display(case_text_df)
    to_df_flg(url)
    ctr += 1

# %%


def get_cases_from_pkl(path, pattern_filter):
    out = []
    for f in Path(path).glob('**/*.pkl'):
        df = pd.read_pickle(f)
        # Do Something

        df['body'] = df['body'].str.replace(
            r'\\(.)', r' ', regex=True).str.strip()
        df['footnote'] = df['footnote'].str.replace(
            r'\\(.)', r' ', regex=True).str.strip()
        df['dispositive portion'] = df['dispositive portion'].str.replace(
            r'\\(.)', r' ', regex=True).str.strip()
        df['ponente'] = df['ponente'].str.replace(
            r':', r'', regex=True).str.strip()
        dfout = df[df['body'].str.match(pattern_filter)]

        if dfout.shape[0] > 0:
            out.append(dfout)

    dfdata_labor = pd.concat(out).reset_index(drop=True)
    return dfdata_labor


def main():
    path = '/Users/brian/Documents/dmw_final/cases'
    # pattern_filter = '(?i).*(family code|inter[-]?country
    # #adoption|domeI stic adoption).*'
    pattern_filter = r'(?i).*labor code of the philippines.*'
    df_labor = get_cases_from_pkl(path, pattern_filter)
    df_labor.head()

    outcols = ['Date of decision', 'Case number', 'Case title',
               'Deciding division', 'Ponente', 'Body', 'Provision(s) cited',
               'Case URL']
    test_df = df_labor.loc[:, ['date_yyyymmdd', 'body']]
    exp_str = (r'(?i)((?:article \d+| art. \d+| art \d+|section'
               ' \d+| sec. \d+| sec \d+)(?:\s*of\s*the\s*labor\s*code))')
    exp_str2 = (r'(?i)((?:labor code,? ?)(?:article \d+|'
                ' art. \d+| art \d+|section \d+| sec. \d+| sec \d+))')

    provisions = []
    for i, r in test_df[['date_yyyymmdd', 'body']].iterrows():
        year = r['date_yyyymmdd'].year
        if year > 2015:
            tag = 'New'
        else:
            tag = 'Old'
        f = re.findall(exp_str, r['body'])
        f2 = re.findall(exp_str2, r['body'])
        all_f = f + f2
        provisions.append(Counter(['{tag} Article {prov}' .format(tag=tag,
                             prov=re.findall(r'\d+', x)[0]) for x in all_f]))

    df_labor['prov'] = provisions

    incols = ['date_yyyymmdd', 'case no', 'case title',
              'division', 'ponente', 'body', 'prov', 'url']

    df_labor_out = df_labor.rename(columns=dict(zip(incols, outcols)))[outcols]

    df_labor_out.to_pickle('../df_labor_new.pkl')
