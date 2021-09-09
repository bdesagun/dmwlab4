import re
from wordcloud import WordCloud
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import linkage, dendrogram
import scipy.cluster.hierarchy as sch


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


# path = './cases'
# pattern_filter = '(?i).*(family code|inter[-]?country adoption|domeI stic adoption).*'
# # pattern_filter = '(?i).*(labor code).*'
# df_labor = get_cases_from_pkl(path, pattern_filter)
# df_labor.head()


# df_labor = df_labor[df_labor['case no'].str.contains('g.r.')].reset_index(drop=True)
# df_labor.head()


def figure1(df_labor):
    df_labor["year"] = df_labor["date_yyyymmdd"].dt.year
    year_count_df = pd.DataFrame(df_labor["year"].value_counts()).reset_index()\
        .rename({"index": "year", "year": "counts"}, axis=1)

    bar_colors = '#5F9EA0'

    fig1 = px.bar(x=year_count_df["year"],
                  y=year_count_df["counts"],
                  color_discrete_sequence=[bar_colors])

    fig1.update_xaxes(tickangle=-30, tickmode='linear', title_text=None)
    fig1.update_yaxes(title_text=None)

    fig1.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    fig1.update_layout(title={
        'text': ('Labor Related Case Decisions Over Years'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
        showlegend=False)
    fig1.show()


def figure2(df_labor):
    df_donut = pd.DataFrame(df_labor["division"].value_counts()).reset_index()
    df_donut.rename({
        "index": "division_",
        "division": "counts"
    },
        axis=1,
        inplace=True)

    night_colors = [
        '#008B8B', '#66CDAA', '#3CB371', '#20B2AA', '#40E0D0', '#1E90FF'
    ]

    # Removed Invalid Divisions
    df_donut = df_donut[~df_donut.division_.isin(['', '[SYLLABUS]'])]

    fig2 = go.Figure(data=[
        go.Pie(labels=df_donut["division_"],
               values=df_donut["counts"],
               hole=.5,
               title="Division",
               marker_colors=night_colors)
    ])

    fig2.update_layout(
        title="Supreme Court Division Distribution",
        title_x=0.5,
    )
    fig2.show()


# def figure3(df_labor):
#     law = df_labor.law.apply(pd.Series).count(axis=0).to_frame()
#     law = (law.reset_index().rename(columns={'index': 'law', 0: 'count'}))
#     bar_colors = '#5F9EA0'
#     fig3 = px.bar(law.sort_values(by='count', ascending=True),
#                   x='count',
#                   y='law',
#                   color_discrete_sequence=[bar_colors])

#     fig3.update_xaxes(title_text=None)
#     fig3.update_yaxes(title_text=None)
#     fig3.update_layout(title={
#         'text': ('Covered Philippine Laws'),
#         'y': 0.95,
#         'x': 0.5,
#         'xanchor': 'center',
#         'yanchor': 'top'
#     },
#                        showlegend=False)
#     fig3.update_layout({
#         'plot_bgcolor': 'rgba(0, 0, 0, 0)',
#         'paper_bgcolor': 'rgba(0, 0, 0, 0)'
#     })

#     fig3.show()

def figure3(df_labor):
    df = df_labor.provision.apply(pd.Series).count(
        axis=0).to_frame().reset_index()
    df.rename(columns={"index": 'article', 0: 'Count'}, inplace=True)

    def f(x): return x.replace('art.', 'article').replace('sec.', 'section')
    df['article'] = df['article'].apply(f)
    dfplot = df.groupby('article').sum().sort_values(
        by='Count', ascending=False)

    articles = (dfplot.iloc[:20, ].reset_index().rename(columns={
        'index': 'article',
        'Count': 'count'
    }))
    bar_colors = '#5F9EA0'
    fig = px.bar(articles.sort_values(by='count', ascending=True),
                 x='count',
                 y='article',
                 color_discrete_sequence=[bar_colors])

    fig.update_xaxes(title_text=None)
    fig.update_yaxes(title_text=None)
    fig.update_layout(title={
        'text': ('Top 20 Mentioned Articles and Sections'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
        showlegend=False)
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    fig.show()


en_stopwords = ['br', 'the', 'i', 'my', 'we', 'our', 'ours', 'ourselves',
                'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
                "she's", 'her', 'hers', 'herself', 'it', "it's", 'its',
                'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
                'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be'
                'been', 'being', 'have', 'has', 'had', 'having', 'did',
                'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'to',
                'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
                'all', 'any', 's', 't', 'can', 'will', 'just', 'don', "don't",
                'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
                'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
                "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',
                "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan',
                "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
                "weren't", 'won', "won't", 'wouldn', "wouldn't", "do", "does",
                'in', 'on', 'for', 'of',
                'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'sept',
                'oct', 'nov', 'dec', 'january', 'february', 'march', 'april', 'june', 'july',
                'august', 'september', 'october', 'november', 'december',
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                'clock']

ph_legal_stop_words = ['private respondent', 'respondent', 'respondents',
                       'plaintiff', 'appellant', 'appellee', 'petitioner',
                       'accused', 'accused appellant', 'private complainant',
                       'rollo', 'id', 'x', 'xx', 'supra',  'j.', 'phil.', 'no.',
                       'tsn', 'transcript of stenographic notes', 'g.r.',
                       'republic v', 'republic vs', 'people', 'v.', ' v ',
                       'vs', 'civil case', 'case', 'ca', "cv", 'gr',
                       'motion', 'dismiss', 'decision', 'resolution',
                       'order', 'so ordered', 'ordered', 'judgment',
                       'notice', 'appeal', 'appealed', 'from',
                       'judicial', 'proceeding', 'guilty', 'doubt', 'resonable'
                       'trial court', 'court appeals', 'appeals', 'rtc',
                       'regional', 'supreme', 'court', 'article', 'art',
                       'family code', 'rule', 'rules', 'revised penal code',
                       'penal code', 'revised', 'shall', 'via', 'law',
                       'dated', 'only', 'under', 'whether', 'et al', 'et', 'al']


def dual_wordcloud(wc, title):
    """Display 2 wordclouds side by side.

    Parameters
    ----------
    wc : list
        2 wordcloud.WordCould objects
    title : list
        WordCloud titles
    """
    fig, ax = plt.subplots(figsize=(15, 15), facecolor=None)
    ax1 = plt.subplot(121)
    ax1.set_title(title[0])
    ax1.imshow(wc[0])
    ax1.axis("off")

    ax2 = plt.subplot(122)
    ax2.set_title(title[1])
    ax2.imshow(wc[1])
    ax2.axis("off")
    # fig.suptitle(title)
    plt.show()


class Lab4:
    """"Class used for lab4 notebook."""

    def __init__(self, df, category=[], stop_words=[]):
        """Initialize Lab3 class.

        Parameters
        ----------
        df : Series

        category : list
            list of strings of categories to use.

        stop_words : list
            list of strings of additional stop_words
        """
        self.stopwords = set(en_stopwords + ph_legal_stop_words + stop_words)
        self.df = df
        self.tfidf_vectorizer, self.bow_ng = self.tfidf_wordcloud(df)

    def display_ve(self):
        """Display variance explained by running svd_plot_varex."""
        q_ng, s_ng, self.p_ng, nssd_ng, self.idx_90 = self.svd_plot_varex(
            self.bow_ng)

    def display_lsa(self):
        """Display LSA charts by running plot_lsa."""
        self.plot_lsa(self.tfidf_vectorizer, self.p_ng)

    def get_wordcloud(self):
        """

        Returns
        -------
        wordcloud
            worldcloud.Wordcloud object of phrases from amazon reviews.
        """
        idf_score = self.tfidf_vectorizer.idf_
        feature_names = self.tfidf_vectorizer.get_feature_names()
        # reverse score to give more weight on more common words
        idf_score_corrected = (max(idf_score) + 1 - idf_score) * 2
        idfscore_feat_d = dict(zip(feature_names, idf_score_corrected))

        wordcloud = WordCloud(width=800, height=800,
                              background_color='white',
                              min_font_size=10)\
            .generate_from_frequencies(idfscore_feat_d)
        return wordcloud

    def truncated_svd(self, X):
        """Return q, sigma, p and NSSD from the origin.

        Parameters
        ----------
        X : np.ndarray
            Input matrix

        Returns
        -------
        q, s, p.T, nssd 
            np.ndarray    
        """
        q, e, p = np.linalg.svd(X)
        e = e.astype(float)
        s = np.diag(e)
        nssd = e**2 / np.sum(e**2)
        return q, s, p.T, nssd

    def decontracted(self, phrase):
        """Remove word contractions from input phrases.

        Parameters
        ----------
        phrase : string

        Returns
        -------
        string
            decontracted phrase
        """
        phrase = re.sub(r"(?i)won't", "will not", phrase)
        phrase = re.sub(r"(?i)can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"(?i)n ?\'t", " not", phrase)
        phrase = re.sub(r"(?i)\'re", " are", phrase)
        phrase = re.sub(r"(?i)\'s", " is", phrase)
        phrase = re.sub(r"(?i)\'d", " would", phrase)
        phrase = re.sub(r"(?i)\'ll", " will", phrase)
        phrase = re.sub(r"(?i)\'t", " not", phrase)
        phrase = re.sub(r"(?i)\'ve", " have", phrase)
        phrase = re.sub(r"(?i)\'m", " am", phrase)
        #     phrase = re.sub(r"not ", "not_", phrase)
        return phrase

    def project_svd(self, q, s, k):
        """Return project_svd of based on k."""
        return q[:, :k] @ s[:k, :k]

    def tfidf_wordcloud(self, df, ngram_up=3):
        """Create WorldCloud and return Tfid vectorized arrays.

        Parameters
        ----------
        df : DataFrame
        ngram_up : integer (optional)

        Returns
        -------
        TfidfVectorizer, fitted TfidfVectorizer

        """
        preprocessed_reviews = []
        for sentence in df.values:
            sentence = self.decontracted(sentence)
            sentence = re.sub("\S*\d\S*", "", sentence).strip()
#             sentence = re.sub('[^A-Za-z_]+', ' ', sentence)
            sentence = ' '.join(e.lower() for e in sentence.split()
                                if e.lower() not in self.stopwords)
            preprocessed_reviews.append(sentence.strip())

        data = preprocessed_reviews

        tfidf_vectorizer = TfidfVectorizer(ngram_range=(2, ngram_up),
                                           #                                            token_pattern=r'[a-z-]+',
                                           min_df=5,
                                           #                                               strip_accent='ascii'
                                           max_df=.8)
        bow_ng = tfidf_vectorizer.fit_transform(data)
        return tfidf_vectorizer, bow_ng

    def svd_plot_varex(self, bow_ng):
        """Plot variable explain and return truncated svd.

        Parameters
        ----------
        bow_ng : np.ndarray

        Returns
        -------
        truncated_svd
        """
        q_ng, s_ng, p_ng, nssd_ng = self.truncated_svd(bow_ng.toarray())
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, len(nssd_ng) + 1), nssd_ng, '-', label='individual',
                color='#FF5A5F')
        ax.set_xlim(0, len(nssd_ng) + 1)
        ax.set_xlabel('SV')
        ax.set_ylabel('variance explained')
        ax = ax.twinx()
        ax.plot(range(1, len(nssd_ng) + 1),
                nssd_ng.cumsum(), 'o-', label='cumulative', color='#5F9EA0')
        ve_cumsum = nssd_ng.cumsum()
        idx_90 = ve_cumsum[ve_cumsum < .9].shape[0]
        ax.axhline(0.9, ls='--', color='#484848')
        ax.axvline(idx_90, ls='--', color='#484848')
        ax.set_ylabel('cumulative variance explained')
        plt.title('Variance Explained per SV\n'
                  f'90% Threshold at SV {idx_90}')
        plt.show()
        return q_ng, s_ng, p_ng, nssd_ng, idx_90

    def plot_lsa(self, tfidf_vectorizer, p_ng):
        """Plot LSA from truncated svd.

        Parameters
        ----------
        tfidf_vectorizer : TfidfVectorizer
        n_ng : np.ndarray
        """
        feature_names = tfidf_vectorizer.get_feature_names()
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        for i, sub_ax in enumerate(fig.axes):
            order = np.argsort(np.abs(p_ng[:, i]))[-10:]
            sub_ax.barh([feature_names[o] for o in order], p_ng[order, i],
                        color='#5F9EA0')
            sub_ax.set_title(f'SV{i+1}')
        plt.subplots_adjust(wspace=0.5)
        plt.suptitle('LSA Dimensions')
        plt.show()


def plot_dendogram(Z, best_t=0):
    """Return a truncated dendrogram plot."""
    fig, ax = plt.subplots(figsize=(15, 10))
    dendrogram = sch.dendrogram(
        Z, truncate_mode='level', p=50, color_threshold=best_t)
    plt.ylabel(r'$\Delta$')
    if best_t > 0:
        plt.axhline(best_t, linestyle='--', color='gray')
    plt.title("Labor related Supreme Court decisions "
              "Clustering using Ward's method dendogram plot")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.show()


def plot_scatter(X_ng_new, mapped_df, y_predict_ng_wards):
    """Return a truncated scatter plot."""
    fig = plt.figure(figsize=(8, 6))
    plt.title("Labor related Supreme Court decisions "
              "Clustering using Ward's method scatter plot")
    plt.scatter(X_ng_new[:, 0], X_ng_new[:, 1], c=y_predict_ng_wards)
    plt.tick_params(
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    labelleft=False,)  # labels along the bottom edge are off

    plt.show()

    mapped_df_wards = pd.merge(mapped_df, pd.Series(
        y_predict_ng_wards, name='y_predict'), left_index=True,
        right_index=True)

    return mapped_df_wards

def cluster_wordcloud(sc_labor, mapped_df_wards, cluster, color):
    df_bow = pd.DataFrame(sc_labor.bow_ng.toarray(
    ), columns=sc_labor.tfidf_vectorizer.get_feature_names())
    c1_idx_w = mapped_df_wards[mapped_df_wards.y_predict == 1].index.tolist()

    c1_bow_w = df_bow.loc[c1_idx_w].sum(axis=0)
    c1_bow_w = c1_bow_w[c1_bow_w > 0]

    c1_wordcloud_w = WordCloud(width=1000,
                               height=1000,
                               mode='RGBA',
                               colormap=None,
                               background_color=None,
                               min_font_size=16,
                               max_words=150,
                               color_func=lambda *args, **kwargs: "black"
                               )\
        .generate_from_frequencies(c1_bow_w)
    return c1_wordcloud_w