from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def get_cases_from_pkl(path, pattern_filter):
    out = [] 
    for f in Path(path).glob('**/*.pkl'):
        df = pd.read_pickle(f)
        # Do Something

        df['body'] = df['body'].str.replace(r'\\(.)',r' ',regex=True).str.strip()
        df['footnote'] = df['footnote'].str.replace(r'\\(.)',r' ',regex=True).str.strip()
        df['dispositive portion'] = df['dispositive portion'].str.replace(r'\\(.)',r' ',regex=True).str.strip()
        df['ponente'] = df['ponente'].str.replace(r':',r'',regex=True).str.strip()
        dfout = df[df['body'].str.match(pattern_filter)]
        
        if dfout.shape[0] >0:
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
            .rename({"index": "year", "year": "counts"}, axis = 1)

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



def figure3(df_labor):
    law = df_labor.law.apply(pd.Series).count(axis=0).to_frame()
    law = (law.reset_index().rename(columns={'index': 'law', 0: 'count'}))
    bar_colors = '#5F9EA0'
    fig3 = px.bar(law.sort_values(by='count', ascending=True),
                  x='count',
                  y='law',
                  color_discrete_sequence=[bar_colors])

    fig3.update_xaxes(title_text=None)
    fig3.update_yaxes(title_text=None)
    fig3.update_layout(title={
        'text': ('Covered Philippine Laws'),
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    },
                       showlegend=False)
    fig3.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)'
    })

    fig3.show()
