import urllib
# import talib
import streamlit as st
import pandas as pd
from sklearn.preprocessing import minmax_scale
# from pykalman import KalmanFilter
import altair as alt
import matplotlib.pyplot as plt
from numpy import nan as Nan
import numpy as np
import os
import time, datetime
import zipfile
import statsmodels.api as sm

# todo prevalence ratio to calc true infections. Then calc asymptomatic and infectious
# todo states where cases and deaths are most and least correlated

TTL = 60 * 60 * 3  # 3 hours

st.set_page_config(page_title='Interactive Covid-19 Forecast and Correlation Explorer', layout='centered',
                   initial_sidebar_state='expanded')


def download_data():
    # Periodically download data
    last_mod = os.path.getmtime('daily.csv')
    last_mod = datetime.datetime.utcfromtimestamp(last_mod)
    dif = datetime.datetime.now() - last_mod
    if dif < datetime.timedelta(hours=12): return

    with st.spinner("Fetching latest data..."):
        urllib.request.urlretrieve('https://api.covidtracking.com/v1/us/daily.csv', 'daily.csv')
        urllib.request.urlretrieve('https://api.covidtracking.com/v1/states/daily.csv', 'states_daily.csv')
        # todo rt
        # https://d14wlfuexuxgcm.cloudfront.net/covid/rt.csv
        # urllib.request.urlretrieve('https://d14wlfuexuxgcm.cloudfront.net/covid/rt.csv','rt.csv')

    # mobility google
    with st.spinner("Fetching Google mobility data..."):
        urllib.request.urlretrieve('https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip',
                                   'Region_Mobility_Report_CSVs.zip')
    with st.spinner("Extracting Google mobility data..."):
        with zipfile.ZipFile('Region_Mobility_Report_CSVs.zip', 'r') as zip_ref:
            zip_ref.extractall('Region_Mobility_Report_CSVs')


@st.cache(ttl=TTL, suppress_st_warning=True)
def process_data(all_states, state):
    # Data
    if all_states:
        df = pd.read_csv('daily.csv').sort_values('date', ascending=True).reset_index()
    else:
        df = pd.read_csv('states_daily.csv').sort_values('date', ascending=True).reset_index().query(
            'state=="{}"'.format(state))

    df = df.query('date >= 20200301')
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)

    # Rolling means
    df['positiveIncrease'] = df['positiveIncrease'].rolling(7).mean()
    df['deathIncrease'] = df['deathIncrease'].rolling(7).mean()
    df['hospitalizedCurrently'] = df['hospitalizedCurrently'].rolling(7).mean()
    df['totalTestResultsIncrease'] = df['totalTestResultsIncrease'].rolling(7).mean()
    df['death'] = df['death'].rolling(7).mean()

    # New features
    df['percentPositive'] = (df['positiveIncrease'] / df['totalTestResultsIncrease']).rolling(7).mean()
    df['Case Fatality Rate'] = (df['death'] / df['positive']) * 100

    df = calc_prevalence_ratio(df)

    df['Infection Fatality Rate'] = (df['death'] / (df['positive'] * df['prevalence_ratio'])) * 100

    # Mobility data ------------------------------------------------------------------------------------------
    mobility = pd.read_csv("Region_Mobility_Report_CSVs/2020_US_Region_Mobility_Report.csv",
                           usecols=[0, 1, 2, 5, 7, 8, 9, 10, 11, 12, 13])
    if all_states:
        mobility = mobility.iloc[mobility.isna().query('sub_region_1==True').index]
    else:
        mobility = mobility.iloc[mobility.query('iso_3166_2_code=="US-{}"'.format(state)).index]
    # Set date index to match df
    mobility['date'] = pd.to_datetime(mobility['date'], format='%Y-%m-%d')
    mobility = mobility.query('date>="2020-03-01"')
    mobility.set_index('date', inplace=True)
    # Concat with df
    df = pd.concat([df, mobility], axis='columns')
    # Smooth
    for c in ['retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
              'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
              'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']:
        df[c] = df[c].rolling(7).mean()
    # Most recent mobility data is empty, project forward
    df.interpolate(inplace=True)
    # End mobility data processing ---------------------------------------------------------------------------

    if np.inf in df.values:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def calc_prevalence_ratio(df):
    # prevalence_ratio(day_i) = (1250 / (day_i + 25)) * (positivity_rate(day_i))^(0.5) + 2, where day_i is the number of days since February 12, 2020.
    # https://covid19-projections.com/estimating-true-infections-revisited/
    import math
    days_since = df.index - datetime.datetime(year=2020, month=2, day=12)
    df['days_since_feb12'] = days_since.days.values
    p_r_list = []
    for i, row in df.iterrows():
        try:
            prevalence_ratio = (1250 / (row['days_since_feb12'] + 25)) * math.pow(row['percentPositive'], 0.5) + 2
        except:
            prevalence_ratio = p_r_list[-1]
        p_r_list.append(prevalence_ratio)
        # st.write(prevalence_ratio)
    df['prevalence_ratio'] = p_r_list
    return df


# def write_trends(cols):
#     import talib
#     up = []
#     down = []
#     for metric in cols:
#         if df[metric].iloc[-1] > talib.SAR(df[metric], df[metric])[-1]:
#             # st.write("{} is rising.".format(metric))
#             up.append(metric)
#         else:
#             down.append(metric)
#             # st.write("{} is declining.".format(metric))
#     st.subheader('Trending Up')
#     st.write(', '.join(up))
#     st.subheader('Trending Down')
#     st.write(', '.join(down))

@st.cache(ttl=TTL)
def find_max_correlation(col, col2):
    best_cor = -1
    best_i = 0
    for i in range(len(col) // 4):
        col1 = col.shift(i)
        correl = col1.corr(col2)
        if correl > best_cor:
            best_cor = correl
            best_i = i

    return best_cor, best_i


def plot_cor(col, col2, best_i, best_cor):
    # st.line_chart({col.name: col.shift(best_i), col2.name: col2})
    st.write("{} shifted {} days ahead is correlated with {}. $r={}$".format(col.name, best_i, col2.name, best_cor))

    # altair chart
    src = pd.DataFrame({col.name: col.shift(best_i), col2.name: col2}).reset_index()
    base = alt.Chart(src).encode(
        alt.X('date:T', axis=alt.Axis(title=None))
    )

    line = base.mark_line(stroke='orange').encode(
        alt.Y(col.name,
              axis=alt.Axis(title=col.name, titleColor='orange'))
    )

    line2 = base.mark_line(stroke='#5276A7').encode(
        alt.Y(col2.name,
              axis=alt.Axis(title=col2.name, titleColor='#5276A7'))
    )

    chrt = alt.layer(line, line2).resolve_scale(
        y='independent'
    )
    st.altair_chart(chrt, use_container_width=True)


# @st.cache(ttl=TTL)
def get_shifted_correlations(df, cols):
    a = st.selectbox("Does this", cols, index=3)
    b = st.selectbox("Correlate with this?", cols, index=2)
    lb = st.slider('How far back should we look for correlations?', min_value=0, max_value=len(df), value=len(df) - 90,
                   step=10, format="%d days", key='window2')
    # st.write(lb / 30, 'months back')

    cor, shift = find_max_correlation(df[a].iloc[-lb:], df[b].iloc[-lb:])
    col1, col2 = df[a].iloc[-lb:], df[b].iloc[-lb:]
    plot_cor(df[a].iloc[-lb:], df[b].iloc[-lb:], shift, cor)

    return cols, a, b, lb


@st.cache(ttl=TTL)
def get_cor_table(cols, lb, df):
    # Find max
    shifted_cors = pd.DataFrame(columns=['a', 'b', 'r', 'shift'])
    for i in cols:
        for j in cols:
            if i == j: continue
            cor, shift_temp = find_max_correlation(df[i].iloc[-lb:], df[j].iloc[-lb:])
            shifted_cors = shifted_cors.append({'a': i, 'b': j, 'r': cor, 'shift': shift_temp}, ignore_index=True)
    return shifted_cors


def forecast_ui(cors_df):
    # st.header('Forecast Based on Shifted Correlations')

    cors_df = cors_df.query("r >0.5 and shift >0")
    if len(cors_df) < 2:
        cors_df = cors_df.query("r >0.0 and shift >=0")
        st.warning("Few strong correlations found for forecasting. Try adjusting lookback window.")

    # forecast_len = int(np.mean(cors_df['r'].values * cors_df['shift'].values))
    # st.write("Forecast Length = average shift weighted by average correlation = ", forecast_len)
    days_back = -st.slider("See how past forecasts did:", 0, lookback, 0, format="%d days back") - 1
    return days_back


@st.cache(ttl=TTL)
def compute_weighted_forecast(days_back, b, shifted_cors):
    cors_df = shifted_cors.query("b == '{}' and r >0.5 and shift >0".format(b))
    if len(cors_df) < 1:
        cors_df = shifted_cors.query("b == '{}' and r >0.0 and shift >=0".format(b))
        # st.warning("No strong correlations found for forecasting. Try adjusting lookback window.")
        # st.stop()
    cols = cors_df['a'].values

    # scale to predicted val
    df[cols] = minmax_scale(df[cols], (df[b].min(), df[b].max()))
    for i, row in cors_df.iterrows():
        col = row['a']
        # weight by cor
        df[col] = df[col] * row['r']
        # shift on x axis
        df[col] = df[col].shift(row['shift'])
        # OLS
        model = sm.OLS(df[b].interpolate(limit_direction='both'),
                       df[col].interpolate(limit_direction='both'))  # Y,X or X,Y ?
        results = model.fit()
        df[col] = df[col] * results.params[0]

    forecast_len = int(np.mean(cors_df['r'].values * cors_df['shift'].values))
    forecast = df[cols].mean(axis=1)

    # todo ML forecast
    # forecast = ml_regression(df[cols], df[b],7)
    # df['forecast'] = forecast
    # forecast = df['forecast']

    # OLS
    df['forecast'] = forecast
    model = sm.OLS(df[b].interpolate(limit_direction='both'),
                   df['forecast'].interpolate(limit_direction='both'))  # Y,X or X,Y ?
    results = model.fit()
    # st.write('OLS Beta =', results.params)
    forecast = forecast * results.params[0]

    # Align on Y axis
    dif = df[b].iloc[days_back] - forecast.iloc[-forecast_len + days_back]
    forecast += dif

    # only plot forward forecast
    forecast.iloc[:-forecast_len + days_back] = np.NAN
    forecast.iloc[days_back:] = np.NAN

    df['forecast'] = forecast

    lines = {
        b: df[b].append(pd.Series([Nan for i in range(forecast_len)]), ignore_index=True),
        "Forecast": df['forecast'].append(pd.Series([Nan for i in range(forecast_len)]),
                                          ignore_index=True).shift(forecast_len),
    }
    return lines, cors_df


def plot_forecast(lines, cors_table):
    idx = pd.date_range(start=df.index[0], periods=len(lines[b]))
    df2 = pd.DataFrame(lines).set_index(idx)
    st.line_chart(df2, width=800, height=400, use_container_width=False)
    # st.line_chart(df2,use_container_width=True)
    # plt.style.use('bmh')
    # st.write(df2.plot().get_figure())


@st.cache(ttl=TTL)
def ml_regression(X, y, lookahead=7):
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    y = y.shift(lookahead)
    X.fillna(0, inplace=True)
    y.fillna(0, inplace=True)
    # X.interpolate(inplace=True, limit_direction='both')
    # y.interpolate(inplace=True, limit_direction='both')
    from sklearn.preprocessing import normalize
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=0)
    reg = GradientBoostingRegressor(random_state=0, verbose=True)
    # reg = RandomForestRegressor(random_state=0, verbose=True)
    reg.fit(X_train, y_train)

    pred = reg.predict(X_test)

    score = reg.score(X_test, y_test)
    reg.fit(X, y)
    return reg.predict(X)


# Unused functions below. May use in future. ---------------------------------------------------------------------------

def matplotlib_charts(df, a='deathIncrease', b='positiveIncrease'):
    plt.style.use('seaborn')
    st.pyplot(df[[a, b]].plot.line().get_figure())
    # st.pyplot(df[[a,b]].plot(subplots=True,layout=(2,2)))
    plots = df[[a, b, 'percentPositive', 'hospitalizedCurrently']].plot.line(subplots=True)
    st.pyplot(plots[0].get_figure())

    # plt.style.use('fivethirtyeight')
    plots = df[[a, b, 'percentPositive', 'hospitalizedCurrently']].plot(subplots=True, layout=(2, 2))
    st.pyplot(plots[0][0].get_figure())


def interactive_plot(df):
    st.title('Interactive Chart')
    cols = ['inIcuCurrently', 'hospitalizedCurrently', 'deathIncrease', 'positiveIncrease', 'percentPositive',
            'totalTestResultsIncrease', 'Case Fatality Rate', 'Infection Fatality Rate']
    cols = st.multiselect('Metrics', cols, default=['positiveIncrease', 'percentPositive', 'deathIncrease'])
    scale = st.checkbox('Scale all Data Equally', value=False)
    scaled_df = df.copy()
    if scale: scaled_df[cols] = minmax_scale(df[cols])
    # st.line_chart(minmax_scale(df[cols]) if scale else df[cols], width=w, height=h, use_container_width=False)
    st.line_chart(scaled_df[cols], width=w, height=h, use_container_width=False)


def get_correlations(df, cols):
    st.header("Correlations")
    df = df[cols]
    cor_table = df.corr(method='pearson', min_periods=30)
    st.write(cor_table)
    max_r = 0
    max_idx = None
    seen = []
    cors = pd.DataFrame(columns=['a', 'b', 'r'])
    for i in cor_table.index:
        for j in cor_table.index:
            if i == j or i == 'index' or j == 'index': continue
            if cor_table.loc[i, j] == 1: continue
            if cor_table.loc[i, j] > max_r:
                max_idx = (i, j)
                max_r = max(cor_table.loc[i, j], max_r)
            if (j, i) not in seen:
                cors = cors.append({'a': i, 'b': j, 'r': cor_table.loc[i, j]}, ignore_index=True)
                seen.append((i, j))
    st.write(max_idx, max_r)
    st.write(cors.sort_values('r', ascending=False).reset_index(drop=True))


# todo normalize timeseries for each state by population and then cluster and tsne. Or instead of by population, minmax scale.
def tsne_plot():
    states = pd.read_csv('states_daily.csv')['state'].unique()
    df = pd.read_csv('states_daily.csv').sort_values('date', ascending=True).reset_index().query(
        'state=="{}"'.format(state))

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df.set_index('date', inplace=True)


if __name__ == '__main__':
    download_data()
    w, h, = 900, 400
    states = pd.read_csv('states_daily.csv')['state'].unique()

    with st.sidebar:
        all_states = st.checkbox("All States", True)
        state = st.selectbox("State", states)

    st.markdown(
        """
                    # Interactive Covid-19 Forecast and Correlation Explorer
        """
    )

    # https://docs.streamlit.io/en/stable/troubleshooting/caching_issues.html#how-to-fix-the-cached-object-mutated-warning
    import copy

    df = copy.deepcopy(process_data(all_states, state))

    cols = ['inIcuCurrently', 'hospitalizedCurrently', 'deathIncrease', 'positiveIncrease', 'percentPositive',
            'totalTestResultsIncrease', 'Case Fatality Rate', 'Infection Fatality Rate']
    cols.extend(
        ['retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline',
         'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline',
         'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline'])
    b = st.selectbox("Plot this:", cols, index=2)
    lookback = st.slider('How far back should we look for correlations?', min_value=0, max_value=len(df),
                         value=len(df) - 70,
                         step=10, format="%d days")
    cors_df = get_cor_table(cols, lookback, df)

    days_back = forecast_ui(cors_df)
    lines, cors_table = compute_weighted_forecast(days_back, b, cors_df)
    plot_forecast(lines, cors_table)

    st.markdown("""
    ## How is this forecast made?

    This forecast is a weighted average of variables from the table below. $shift$ is the number of days $a$ is shifted forward, and $r$ is the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between shifted $a$ and $b$.
    """)
    st.write(cors_table)
    st.write("Below you can choose two variables and see if they are correlated.")

    st.title("Interactive Correlation Explorer")
    cols, a, b, lookback = get_shifted_correlations(df, cols)

    st.markdown(""" ## Further Explanation
    The model searches every combination of $a$, $b$, and $shift$ for the highest $r$ values. Only correlations $>0.5$ are used. $r$ is used to weight each component of the forecast, and each component is scaled and aligned to the forecasted variable $b$. The forecast length is the average $shift$ weighted by the average $r$.
    
    Ordinary Least Squares regression is also used to scale each series from the *a* column as well as the final forecast.
    """)

    st.header("Sources and References")
    st.markdown(
        "Infection fatality rate is calculated using the formula described by https://covid19-projections.com/estimating-true-infections-revisited:")
    st.latex("prevalenceRatio({day_{i}}) = (1250/(day_i+25)) * positivityRate^{0.5}+2")
    st.markdown(
        "Data is pulled daily from https://covidtracking.com. Mobility data is from [google.com/covid19/mobility](https://www.google.com/covid19/mobility/)")
    st.markdown(
        '''
        ## To Do

        - Score forecasts with MSE or other metric

        - Feed correlated variables into ML model for forecasting. LSTM, RF, XGBoost

        - ~~Add Google mobility data~~

        - Add data from https://rt.live

        - Try using cointegration instead of correlation

        - Cleanup code

        - PCA, cluster, and TSNE plot different states

        - Intra-state correlations

        '''

    )
