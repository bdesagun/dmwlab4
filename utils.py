from collections import Counter
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


def figure3(df_labor):
    """Display figure3 for Lab4 report.
    
    Parameters
    ----------
    df_labor : pd.DataFrame
    """
    df_labor["year"] = df_labor["Date of decision"].dt.year
    year_count_df = pd.DataFrame(df_labor["year"].value_counts())\
        .reset_index()\
        .rename({"index": "year", "year": "counts"}, axis=1)

    bar_colors = '#5F9EA0'

    mean_counts = year_count_df['counts'].mean()
    min_yr = min(year_count_df['year'])
    max_yr = max(year_count_df['year'])    

    fig = px.bar(x=year_count_df["year"],
                  y=year_count_df["counts"],
                  color_discrete_sequence=[bar_colors])

    fig.update_xaxes(tickangle=-30, tickmode='linear', title_text=None)
    fig.update_yaxes(title_text=None)
    
    fig.add_shape(type='line',
                    x0=min_yr,
                    y0=mean_counts,
                    x1=max_yr,
                    y1=mean_counts,
                    line=dict(color='Red',),
                    xref='x',
                    yref='y'
    )

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    fig.update_layout(title={
        'text': ('Labor Related Case Decisions Over Years'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
        showlegend=False)
    fig.show()


def figure4(df_labor):
    """Display figure4 for Lab4 report.
    
    Parameters
    ----------
    df_labor : pd.DataFrame
    """
    def change_div(
        x): return 'DIVISION' if 'division' in x.lower() else 'EN BANC'
    df_labor["Deciding division"] = df_labor["Deciding division"].\
        apply(change_div)

    df_donut = pd.DataFrame(df_labor["Deciding division"].value_counts()).\
        reset_index()
    df_donut.rename({
        "index": "division_",
        "Deciding division": "counts"
    },
        axis=1,
        inplace=True)

    night_colors = [
        '#008B8B', '#66CDAA', '#3CB371', '#20B2AA', '#40E0D0', '#1E90FF'
    ]

    # Removed Invalid Divisions
    df_donut = df_donut[~df_donut.division_.isin(['', '[SYLLABUS]'])]

    fig = go.Figure(data=[
        go.Pie(labels=df_donut["division_"],
               values=df_donut["counts"],
               hole=.5,
               title="Division",
               marker_colors=night_colors)
    ])

    fig.update_layout(
        title="Supreme Court Division Distribution",
        title_x=0.5,
    )
    fig.show()



def figure5(df_labor):
    """Display figure5 for Lab4 report.
    
    Parameters
    ----------
    df_labor : pd.DataFrame
    """
    df = df_labor['Provision(s) cited'].apply(pd.Series).count(
        axis=0).to_frame().reset_index()
    df.rename(columns={"index": 'article', 0: 'Count'}, inplace=True)

    def f(x): return x.replace('art.', 'article').replace('sec.', 'section')
    df['article'] = df['article'].apply(f)
    dfplot = df.groupby('article').sum().sort_values(
        by='Count', ascending=False)

    articles = (dfplot.iloc[:5, ].reset_index().rename(columns={
        'index': 'article',
        'Count': 'count'
    }))
    bar_colors = '#5F9EA0'
    fig = px.bar(articles.sort_values(by='count', ascending=True),
                 x='count',
                 y='article',
                 color_discrete_sequence=[bar_colors])

    fig.update_xaxes(title_text=None,visible=False)
    fig.update_yaxes(title_text=None)
    fig.update_layout(title={
        'text': ('Top 20 Cited Labor Code Provisions'),
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
                'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep',
                'sept', 'oct', 'nov', 'dec', 'january', 'february', 'march',
                'april', 'june', 'july',
                'august', 'september', 'october', 'november', 'december',
                'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                'saturday', 'sunday',
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
        self.tfidf_vectorizer, self.bow_lc = self.get_tfidf(df)

    def display_ve(self):
        """Display variance explained by running svd_plot_varex."""
        q_lc, s_lc, self.p_lc, nssd_lc, self.idx_90 = self.svd_plot_varex(
            self.bow_lc)

    def display_lsa(self):
        """Display LSA charts by running plot_lsa."""
        self.plot_lsa(self.tfidf_vectorizer, self.p_lc)

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

    def get_tfidf(self, df, ngram_up=3):
        """Return Tfid vectorized arrays.

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
        bow_lc = tfidf_vectorizer.fit_transform(data)
        return tfidf_vectorizer, bow_lc

    def svd_plot_varex(self, bow_lc):
        """Plot variable explain and return truncated svd.

        Parameters
        ----------
        bow_lc : np.ndarray

        Returns
        -------
        truncated_svd
        """
        q_lc, s_lc, p_lc, nssd_lc = self.truncated_svd(bow_lc.toarray())
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, len(nssd_lc) + 1), nssd_lc, '-', label='individual',
                color='#FF5A5F')
        ax.set_xlim(0, len(nssd_lc) + 1)
        ax.set_xlabel('SV')
        ax.set_ylabel('variance explained')
        ax = ax.twinx()
        ax.plot(range(1, len(nssd_lc) + 1),
                nssd_lc.cumsum(), 'o-', label='cumulative', color='#5F9EA0')
        ve_cumsum = nssd_lc.cumsum()
        idx_90 = ve_cumsum[ve_cumsum < .9].shape[0]
        ax.axhline(0.9, ls='--', color='#484848')
        ax.axvline(idx_90, ls='--', color='#484848')
        ax.set_ylabel('cumulative variance explained')
        plt.title('Variance Explained per SV\n'
                  f'90% Threshold at SV {idx_90}')
        plt.show()
        return q_lc, s_lc, p_lc, nssd_lc, idx_90

    def plot_lsa(self, tfidf_vectorizer, p_lc):
        """Plot LSA from truncated svd.

        Parameters
        ----------
        tfidf_vectorizer : TfidfVectorizer
        n_lc : np.ndarray
        """
        feature_names = tfidf_vectorizer.get_feature_names()
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))
        for i, sub_ax in enumerate(fig.axes):
            order = np.argsort(np.abs(p_lc[:, i]))[-10:]
            sub_ax.barh([feature_names[o] for o in order], p_lc[order, i],
                        color='#5F9EA0')
            sub_ax.set_title(f'SV{i+1}')
        plt.subplots_adjust(wspace=0.5)
        plt.suptitle('LSA Dimensions')
        plt.show()


def plot_dendrogram(Z, best_t=0):
    """Return a truncated dendrogram plot."""
    fig, ax = plt.subplots(figsize=(15, 10))
    dendrogram = sch.dendrogram(
        Z, truncate_mode='level', p=50, color_threshold=best_t)
    plt.ylabel(r'$\Delta$')
    if best_t > 0:
        plt.axhline(best_t, linestyle='--', color='gray')
    plt.title("Labor related Supreme Court decisions "
              "Clustering using Ward's method dendrogram plot")
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)  # labels along the bottom edge are off
    plt.show()


def plot_scatter(X_lc_new, df_combined, y_predict_lc_wards):
    """Return a truncated scatter plot.
    
    Parameters
    ----------
    X_lc_new : np.ndarray
        Truncated SVD
    df_combined : pd.DataFrame
        Labor Dataframe
    y_predict_lc_wards : np.ndarray
        Wards Clustering labels

    Returns
    -------
    df_combined_wards
    """
    fig = plt.figure(figsize=(8, 6))
    plt.title("Labor related Supreme Court decisions "
              "Clustering using Ward's method scatter plot")
    plt.scatter(X_lc_new[:, 0], X_lc_new[:, 1], c=y_predict_lc_wards)
    plt.tick_params(
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        left=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False,)  # labels along the bottom edge are off

    plt.show()

    df_combined_wards = pd.merge(df_combined, pd.Series(
        y_predict_lc_wards, name='y_predict'), left_index=True,
        right_index=True)

    return df_combined_wards


def cluster_wordcloud(sc_labor, df_combined_wards, cluster, color):
    """Return a worldcloud instance base on cluster result."""
    df_bow = pd.DataFrame(sc_labor.bow_lc.toarray(
    ), columns=sc_labor.tfidf_vectorizer.get_feature_names())
    c_idx_w = df_combined_wards[df_combined_wards.y_predict ==
                               cluster].index.tolist()

    c_bow_w = df_bow.loc[c_idx_w].sum(axis=0)
    c_bow_w = c_bow_w[c_bow_w > 0]

    c_wordcloud_w = WordCloud(width=1000,
                               height=1000,
                               mode='RGBA',
                               colormap=None,
                               background_color=None,
                               min_font_size=16,
                               max_words=150,
                               color_func=lambda *args, **kwargs: color
                               )\
        .generate_from_frequencies(c_bow_w)
    return c_wordcloud_w
