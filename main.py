import copy
import datetime
import math
import os
import urllib
import zipfile

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
from numpy import nan as Nan
from scipy.signal import find_peaks
from sklearn.preprocessing import minmax_scale, normalize
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import ReducedRegressionForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import smape_loss
from sktime.utils.plotting import plot_series

# todo states where cases and deaths are most and least correlated


st.set_page_config(
    page_title="Interactive Covid-19 Forecast and Correlation Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def download_data(wait_hours=6):
    """
    Periodically download data to csv
    """

    # Download new data when last mod time was > x hours
    filepath = "vaccine.csv"  # 'daily.csv'
    last_mod = os.path.getmtime(filepath)
    last_mod = datetime.datetime.utcfromtimestamp(last_mod)
    dif = datetime.datetime.now() - last_mod
    if dif < datetime.timedelta(hours=wait_hours) and os.path.exists(
        "Region_Mobility_Report_CSVs"
    ):
        return

    # Clear cache if we have new data
    st.caching.clear_cache()

    with st.spinner("Fetching latest data..."):
        os.remove("daily.csv")
        os.remove("states_daily.csv")
        urllib.request.urlretrieve(
            "https://api.covidtracking.com/v1/us/daily.csv", "daily.csv"
        )
        urllib.request.urlretrieve(
            "https://api.covidtracking.com/v1/states/daily.csv", "states_daily.csv"
        )
        # todo rt
        # https://d14wlfuexuxgcm.cloudfront.net/covid/rt.csv
        # urllib.request.urlretrieve('https://d14wlfuexuxgcm.cloudfront.net/covid/rt.csv','rt.csv')

    # mobility google
    with st.spinner("Fetching Google mobility data..."):
        urllib.request.urlretrieve(
            "https://www.gstatic.com/covid19/mobility/Region_Mobility_Report_CSVs.zip",
            "Region_Mobility_Report_CSVs.zip",
        )
    with st.spinner("Extracting Google mobility data..."):
        with zipfile.ZipFile("Region_Mobility_Report_CSVs.zip", "r") as zip_ref:
            zip_ref.extractall("Region_Mobility_Report_CSVs")
        os.remove("Region_Mobility_Report_CSVs.zip")

    with st.spinner("Downloading vaccination data..."):
        os.remove("vaccine.csv")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/youyanggu/covid19-cdc-vaccination-data/main/aggregated_adjusted.csv",
            "vaccine.csv",
        )


@st.cache(suppress_st_warning=True)
def process_data(all_states, state):
    """
    Process CSVs. Smooth and compute new series.

    :param all_states: Boolean if "all states" is checked
    :param state: Selected US state
    :return: Dataframe
    """
    # Data
    if all_states:
        df = pd.read_csv("daily.csv").sort_values("date", ascending=True).reset_index()
    else:
        df = (
            pd.read_csv("states_daily.csv")
            .sort_values("date", ascending=True)
            .reset_index()
            .query('state=="{}"'.format(state))
        )

    df = df.query("date >= 20200301")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df.set_index("date", inplace=True)

    # Rolling means
    df["positiveIncrease"] = df["positiveIncrease"].rolling(7).mean()
    df["deathIncrease"] = df["deathIncrease"].rolling(7).mean()
    df["hospitalizedCurrently"] = df["hospitalizedCurrently"].rolling(7).mean()
    df["totalTestResultsIncrease"] = df["totalTestResultsIncrease"].rolling(7).mean()
    # df['death'] = df['death'].rolling(7).mean()

    # New features
    df["percentPositive"] = (
        (df["positiveIncrease"] / df["totalTestResultsIncrease"]).rolling(7).mean()
    )
    df["Case Fatality Rate"] = (df["death"] / df["positive"]) * 100

    df = calc_prevalence_ratio(df)

    df["Infection Fatality Rate"] = (
        df["death"] / (df["positive"] * df["prevalence_ratio"])
    ) * 100
    df["percentPositive"] = df["percentPositive"] * 100
    df["Cumulative Recovered Infections Estimate"] = (
        df["positive"] * df["prevalence_ratio"] - df["death"]
    )

    # Mobility data ------------------------------------------------------------------------------------------
    mobility_cols = [
        "country_region_code",
        "country_region",
        "sub_region_1",
        "iso_3166_2_code",
        "date",
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]
    mobility = pd.read_csv(
        "Region_Mobility_Report_CSVs/2020_US_Region_Mobility_Report.csv",
        usecols=mobility_cols,
    )

    if all_states:
        mobility = mobility.iloc[mobility.isna().query("sub_region_1==True").index]
    else:
        mobility = mobility.iloc[
            mobility.query('iso_3166_2_code=="US-{}"'.format(state)).index
        ]
    # Set date index to match df
    mobility["date"] = pd.to_datetime(mobility["date"], format="%Y-%m-%d")
    mobility = mobility.query('date>="2020-03-01"')
    mobility.set_index("date", inplace=True)
    # Concat with df
    df = pd.concat([df, mobility], axis="columns")
    # Smooth
    for c in [
        "retail_and_recreation_percent_change_from_baseline",
        "grocery_and_pharmacy_percent_change_from_baseline",
        "parks_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
        "residential_percent_change_from_baseline",
    ]:
        df[c] = df[c].rolling(7).mean()
    # Most recent mobility data is empty, project forward
    df.interpolate(inplace=True)
    # End mobility data processing ---------------------------------------------------------------------------

    # Vaccination data
    vaccine = pd.read_csv("vaccine.csv")
    vaccine["Date"] = pd.to_datetime(vaccine["Date"], format="%Y-%m-%d")
    vaccine = vaccine.query('Date>="2020-03-01"')
    vaccine.set_index("Date", inplace=True)
    # vaccine = vaccine.fillna(0)

    if all_states:
        us_total = []
        for s in states:
            vac_state = vaccine.query('Location=="{}"'.format(s))
            if len(us_total) == 0:
                us_total = vac_state
            else:
                us_total += vac_state
        vaccine = us_total

    else:
        vaccine = vaccine.query('Location=="{}"'.format(state))  # [vac_col]
    # Fill empty dates
    for vac_col in vaccine.columns.values:
        df[vac_col] = vaccine[vac_col]

    # Fill missing
    # df.fillna(method='pad', inplace=True)
    df = df.interpolate(limit_direction="forward")
    df["Administered_Dose2"] = df["Administered_Dose2"].fillna(0)
    df["Administered_Dose1"] = df["Administered_Dose1"].fillna(0)
    # df['administered_dose2_adj'] = df['administered_dose2_adj'].fillna(0)
    df["First Doses Administered"] = df["administered_dose1_adj"].fillna(0)
    df = df.bfill()
    # End Vaccination data

    if np.inf in df.values:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def calc_prevalence_ratio(df):
    """
    Calculate prevalence ratio
    prevalence_ratio(day_i) = (1250 / (day_i + 25)) * (positivity_rate(day_i))^(0.5) + 2, where day_i is the number of days since February 12, 2020.
    https://covid19-projections.com/estimating-true-infections-revisited/

    :param df: Dataframe from process_data()
    :return: Dataframe with prevalence_ratio column
    """

    days_since = df.index - datetime.datetime(year=2020, month=2, day=12)
    df["days_since_feb12"] = days_since.days.values
    p_r_list = []
    for i, row in df.iterrows():
        try:
            # prevalence_ratio = (1250 / (row['days_since_feb12'] + 25)) * math.pow(row['percentPositive'], 0.5) + 2
            # prevalence_ratio = (1500 / (row["days_since_feb12"] + 50)) * math.pow(
            #     row["percentPositive"], 0.5
            # ) + 2
            prevalence_ratio = (1000 / (row["days_since_feb12"] + 10)) * math.pow(
                row["percentPositive"], 0.5
            ) + 2
            # st.write(prevalence_ratio)
        except:
            prevalence_ratio = p_r_list[-1]
        p_r_list.append(prevalence_ratio)
        # st.write(prevalence_ratio)
    df["prevalence_ratio"] = p_r_list
    return df


@st.cache()
def find_max_correlation(col, col2):
    """
    Take two series and test all alignments for maximum correlation.
    :param col: Column 1
    :param col2: Column 2
    :return: Best r, best shift
    """
    best_cor = -1
    best_i = 0
    for i in range(len(col) // 5):
        col1 = col.shift(i)
        correl = col1.corr(col2)
        if correl > best_cor:
            best_cor = correl
            best_i = i

    return best_cor, best_i


def plot_cor(col, col2, best_i, best_cor):
    """
    Plot interactive chart showing correlation between two shifted series.

    :param col:
    :param col2:
    :param best_i:
    :param best_cor:
    """
    # st.line_chart({col.name: col.shift(best_i), col2.name: col2})
    st.write(
        "{} shifted {} days ahead is correlated with {}. $r={}$".format(
            col.name, best_i, col2.name, round(best_cor, 2)
        )
    )

    # altair chart
    src = pd.DataFrame({col.name: col.shift(best_i), col2.name: col2}).reset_index()
    base = alt.Chart(src).encode(alt.X("date:T", axis=alt.Axis(title=None)))

    line = base.mark_line(stroke="orange").encode(
        alt.Y(col.name, axis=alt.Axis(title=col.name, titleColor="orange"))
    )

    line2 = base.mark_line(stroke="#5276A7").encode(
        alt.Y(col2.name, axis=alt.Axis(title=col2.name, titleColor="#5276A7"))
    )

    chrt = alt.layer(line, line2).resolve_scale(y="independent")
    st.altair_chart(chrt, use_container_width=True)


# @st.cache(ttl=TTL)
def get_shifted_correlations(df, cols):
    """
    Interactive correlation explorer. For two series, finds the alignment that maximizes correlation.
    :param df:
    :param cols:
    :return:
    """
    a = st.selectbox("Does this", cols, index=3)
    b = st.selectbox("Correlate with this?", cols, index=2)
    lb = st.slider(
        "How far back should we look for correlations?",
        min_value=0,
        max_value=len(df),
        value=len(df) - 90,
        step=10,
        format="%d days",
        key="window2",
    )

    cor, shift = find_max_correlation(df[a].iloc[-lb:], df[b].iloc[-lb:])
    col1, col2 = df[a].iloc[-lb:], df[b].iloc[-lb:]
    plot_cor(df[a].iloc[-lb:], df[b].iloc[-lb:], shift, cor)

    return cols, a, b, lb


@st.cache()
def get_cor_table(cols, lb, df):
    """
    Generates dataframe of correlated series and alignments for all given columns.
    :param cols:
    :param lb: Lookback for correlation coefficent
    :param df:
    :return:
    """
    # Find max
    shifted_cors = pd.DataFrame(columns=["a", "b", "r", "shift"])
    for i in cols:
        for j in cols:
            if i == j:
                continue
            cor, shift_temp = find_max_correlation(df[i].iloc[-lb:], df[j].iloc[-lb:])
            shifted_cors = shifted_cors.append(
                {"a": i, "b": j, "r": cor, "shift": shift_temp}, ignore_index=True
            )
    return shifted_cors


def forecast_ui(cors_df, lookback):
    """
    Gets user input for correlation forecast
    :param cors_df: Correlations table
    :return:
    """
    # st.header('Forecast Based on Shifted Correlations')

    cors_df = cors_df.query("r >0.5 and shift >0")
    if len(cors_df) < 2:
        cors_df = cors_df.query("r >0.0 and shift >=0")
        st.warning(
            "Few strong correlations found for forecasting. Try adjusting lookback window."
        )

    # forecast_len = int(np.mean(cors_df['r'].values * cors_df['shift'].values))
    # st.write("Forecast Length = average shift weighted by average correlation = ", forecast_len)
    days_back = (
        -st.slider(
            "See how past forecasts did:", 0, lookback // 2, 0, format="%d days back"
        )
        - 1
    )
    return days_back


@st.cache()
def compute_weighted_forecast(days_back, b, shifted_cors):
    """
    Computes a weighted average of all series that correlate with column B when shifted into the future.
    The weighted average is scaled and aligned to the target column b.

    :param days_back: How far back to start forecasting.
    :param b: Target column to forecast.
    :param shifted_cors: Table of correlated series and shifts
    :return:
    """
    cors_df = shifted_cors.query("b == '{}' and r >0.5 and shift >0".format(b))
    if len(cors_df) < 3:
        cors_df = shifted_cors.query("b == '{}' and r >0.0 and shift >0".format(b))
        # st.warning("No strong correlations found for forecasting. Try adjusting lookback window.")
        # st.stop()
    cols = cors_df["a"].values

    # scale to predicted val
    df[cols] = minmax_scale(df[cols], (df[b].min(), df[b].max()))
    for i, row in cors_df.iterrows():
        col = row["a"]
        # weight by cor
        df[col] = df[col] * row["r"]
        # shift on x axis
        df[col] = df[col].shift(row["shift"])
        # OLS
        model = sm.OLS(
            df[b].interpolate(limit_direction="both"),
            df[col].interpolate(limit_direction="both"),
        )  # Y,X or X,Y ?
        results = model.fit()
        df[col] = df[col] * results.params[0]

    forecast_len = int(np.mean(cors_df["r"].values * cors_df["shift"].values))
    forecast = df[cols].mean(axis=1)

    # ML forecast
    # forecast = ml_regression(df[cols], df[b],7)
    # df['forecast'] = forecast
    # forecast = df['forecast']

    # OLS
    df["forecast"] = forecast
    model = sm.OLS(
        df[b].interpolate(limit_direction="both"),
        df["forecast"].interpolate(limit_direction="both"),
    )  # Y,X or X,Y ?
    results = model.fit()
    # st.write('OLS Beta =', results.params)
    forecast = forecast * results.params[0]

    # Align on Y axis
    dif = df[b].iloc[days_back] - forecast.iloc[-forecast_len + days_back]
    forecast += dif

    # only plot forward forecast
    forecast.iloc[: -forecast_len + days_back] = np.NAN
    forecast.iloc[days_back:] = np.NAN

    df["forecast"] = forecast

    lines = {
        b: df[b].append(
            pd.Series([Nan for i in range(forecast_len)]), ignore_index=True
        ),
        "Forecast": df["forecast"]
        .append(pd.Series([Nan for i in range(forecast_len)]), ignore_index=True)
        .shift(forecast_len),
    }
    return lines, cors_df


def plot_forecast(lines, cors_table):
    """
    Plots output from compute_weighted_forecast()
    :param lines: Dict with forecast and target variable.
    :param cors_table: Table of correlated series and shifts
    """
    idx = pd.date_range(start=df.index[0], periods=len(lines[b]))
    df2 = pd.DataFrame(lines).set_index(idx)
    st.line_chart(df2, use_container_width=True)
    # plt.style.use('bmh')
    # st.write(df2.plot().get_figure())


@st.cache()
def compute_arima(df, colname, days, oos):
    """
    Must do computation in separate function for streamlit caching.

    :param df:
    :param colname:
    :param days:
    :param oos: Out of sample forecast.
    :return:
    """
    y = df[colname].dropna()
    if oos:
        # Forecast OOS
        range = pd.date_range(
            start=y.index[-1] + datetime.timedelta(days=1),
            end=y.index[-1] + datetime.timedelta(days=days),
        )
        fh = ForecastingHorizon(range, is_relative=False)
        forecaster = AutoARIMA(suppress_warnings=True)
        forecaster.fit(y)
        alpha = 0.05  # 95% prediction intervals
        y_pred, pred_ints = forecaster.predict(fh, return_pred_int=True, alpha=alpha)
        return [y, y_pred], ["y", "y_pred"], pred_ints, alpha
    else:
        y_train, y_test = temporal_train_test_split(y, test_size=days)
        fh = ForecastingHorizon(y_test.index, is_relative=False)
        forecaster = AutoARIMA(suppress_warnings=True)
        forecaster.fit(y_train)
        alpha = 0.05  # 95% prediction intervals
        y_pred, pred_ints = forecaster.predict(fh, return_pred_int=True, alpha=alpha)
        return (
            [y_train, y_test, y_pred],
            ["y_train", "y_test", "y_pred"],
            pred_ints,
            alpha,
        )


def timeseries_forecast(df, colname, days=14):
    """
    ARIMA forecast wrapper

    :param df: Dataframe from process_data()
    :param colname: Name of forecasted variable
    :param days_back: Lookback when validating, and lookahead for out of sample forecast.
    """
    st.subheader("Past Performance")
    sktime_plot(*compute_arima(df, colname, days, False))

    st.subheader("Forecast")
    sktime_plot(*compute_arima(df, colname, days, True))
    # y_pred, _, pred_ints, _ = compute_arima(df, colname, days, True)
    # st.line_chart(pd.DataFrame(y_pred).transpose())


def sktime_plot(series, labels, pred_ints, alpha):
    """
    Plot forecasts using sktime plot_series
    https://docs.streamlit.io/en/stable/deploy_streamlit_app.html#limitations-and-known-issues

    :param series:
    :param labels:
    :param pred_ints:
    :param alpha:
    """

    _lock = RendererAgg.lock

    with _lock:
        fig, ax = plot_series(*series, labels=labels)
        # Plot with intervals
        ax.fill_between(
            ax.get_lines()[-1].get_xdata(),
            pred_ints["lower"],
            pred_ints["upper"],
            alpha=0.2,
            color=ax.get_lines()[-1].get_c(),
            label=f"{1 - alpha}% prediction intervals",
        )
        ax.legend()
        st.pyplot(fig)


def arima_ui(df, cols, default=3, max_len=20):
    st.title("ARIMA Forecast")
    st.write(
        "This forecast uses a method called [ARIMA](https://www.reddit.com/r/statistics/comments/k9m9wy/question_arima_in_laymans_terms/gf5a9ub?utm_source=share&utm_medium=web2x&context=3)."
    )
    length = st.slider("Forecast length", 1, max_len, value=14)

    b = st.selectbox("Forecast this:", cols, index=default)
    timeseries_forecast(df, b, length)


def pop_immunity(df):
    # Population Immunity Threshold
    st.markdown("# Population Immunity and Vaccination Progress for the US")
    # st.markdown("### Tracking The Return To Normal")
    # vac_col = "First Doses Administered"
    vac_col = st.selectbox("Count first dose or second?",["First Doses Administered","Administered_Dose2"])

    recovered_frac = st.slider(
        "Fraction of vaccinations that go to recovered infections:", 0.0, 0.5, 0.2
    )
    if recovered_frac:
        # Recovered infections can get vaccinated
        df["Cumulative Recovered Infections Estimate"] -= df[vac_col] * recovered_frac
        df["Remaining Population"] = df["Census2019"] - (
            df["Cumulative Recovered Infections Estimate"] + df[vac_col]
        )
    else:
        df["Remaining Population"] = df["Census2019"] - (
            df["Cumulative Recovered Infections Estimate"] + df[vac_col]
        )

    # T cell crossover immunity
    cross_immune = 0
    # cross_immune = st.slider('Cross Immunity (See below for more info)', 0, 50, 0, step=5)
    if cross_immune != 0:
        df["Cross Immunity"] = cross_immune / 100 * df["Census2019"] - df[vac_col] * (
            cross_immune / 100
        )
    else:
        df["Cross Immunity"] = np.zeros(len(df))
    df["Remaining Population"] -= df["Cross Immunity"]

    df["Estimated Population Immunity %"] = (
        (
            df["Cumulative Recovered Infections Estimate"]
            + df[vac_col]
            + df["Cross Immunity"]
        )
        / df["Census2019"]
        * 100
    )

    # matplotlib_charts(
    #     df.iloc[-30*12:],
    #     [
    #         vac_col,
    #         # "Cumulative Recovered Infections Estimate",
    #         # "Administered_Dose2",
    #         "Estimated Population Immunity %",
    #         "hospitalizedCurrently",
    #         "positiveIncrease",
    #     ],
    # )

    immune_progress_bar(df, vac_col)
    st.area_chart(
        df[
            [
                "Remaining Population",
                "Cumulative Recovered Infections Estimate",
                vac_col,
            ]
        ]
    )
    st.area_chart(df["Estimated Population Immunity %"], height=80)
    st.line_chart(df[["positiveIncrease", "hospitalizedCurrently"]], height=200)
    last_peak_progress_bar(df)

    # Forecast Immnuity
    # arima_ui(df, ["Estimated Population Immunity %"], 0, 90 * 3)

    st.markdown(open("Immunity.md", "r").read())


def immune_progress_bar(df, vac_col):
    # Progress bar - pop immune
    immune_pct = round(df["Estimated Population Immunity %"].iloc[-1], 2)
    st.subheader("Estimated Population Immunity: {}%".format(immune_pct))
    st.progress(immune_pct / 100)

    # recovered = df['Cumulative Recovered Infections Estimate'].iloc[-1] / df['Census2019'] * 100
    # recovered = round(recovered.iloc[-1], 2)
    # st.subheader("Estimated Population Immunity from Recovered Infections: {}%".format(recovered))
    # st.progress(recovered / 100)
    #
    # recovered = df[vac_col].iloc[-1] / df['Census2019'] * 100
    # recovered = round(recovered.iloc[-1], 2)
    # st.subheader("Estimated Population Immunity from Vaccinations (1st Dose): {}%".format(recovered))
    # st.progress(recovered / 100)


def last_peak_progress_bar(df):
    # Find immunity at last peak
    peaks, _ = find_peaks(df["hospitalizedCurrently"])  # todo winter 2020/2021 only
    peak_date = df.index[peaks[-1]]
    found_thresh = df["Estimated Population Immunity %"].iloc[peaks[-1]]

    # Progress bar - estimate
    st.subheader(
        "Hospitalizations began to decline after {} when the estimated population immunity was {}%.".format(
            peak_date._date_repr, round(found_thresh, 2)
        )
    )
    st.progress(found_thresh / 100)
    # todo x% of immunity was from recovered infections, vaccines, pre-existing immunity from other coronaviruses.


def matplotlib_charts(df, cols):
    plt.style.use("seaborn")
    # plt.style.use("seaborn-whitegrid")
    # plt.style.use("fivethirtyeight")
    # st.pyplot(df[cols].plot.area().get_figure())

    # st.pyplot(df[[
    #             "Remaining Population",
    #             "Cumulative Recovered Infections Estimate",
    #             "First Doses Administered",]
    #         ].plot.area().get_figure())

    plots = df[cols].plot.line(subplots=True)
    st.pyplot(plots[0].get_figure())

    # plots = df[cols].plot(
    #     subplots=True, layout=(2, 2)
    # )
    # st.pyplot(plots[0][0].get_figure())


# Unused functions below. May use in future. ---------------------------------------------------------------------------
@st.cache()
def ml_regression(X, y, lookahead=7):
    """
    Feed correlated and shifted variables into ML model for forecasting.
    Doesn't seem to do better than weighted average forecast.

    :param X: Correlation table. df[cols]
    :param y: Target series. df[b]
    :param lookahead: Forecast this many days ahead.
    :return: Forecasted series.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    y_shift = y.shift(lookahead)
    X.fillna(0, inplace=True)
    y_shift.fillna(0, inplace=True)
    # X.interpolate(inplace=True, limit_direction='both')
    # y.interpolate(inplace=True, limit_direction='both')
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_shift, random_state=0, shuffle=False
    )
    reg = GradientBoostingRegressor(random_state=0, verbose=True)
    # reg = RandomForestRegressor(random_state=0, verbose=True)
    reg.fit(X_train, y_train)

    pred = reg.predict(X_test)

    score = reg.score(X_test, y_test)
    reg.fit(X, y_shift)

    # sktime
    y.fillna(0, inplace=True)
    y_train, y_test = temporal_train_test_split(y, test_size=14)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    forecaster = ReducedRegressionForecaster(
        regressor=reg, window_length=12, strategy="recursive"
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    fig, ax = plot_series(
        y_train, y_test, y_pred, labels=["y_train", "y_test", "y_pred"]
    )
    st.write(fig)
    smape_loss(y_test, y_pred)

    return reg.predict(X)


def rename_columns(df):
    # todo
    col_map = {
        "inIcuCurrently": "Currently in ICU",
        "hospitalizedCurrently": "Currently Hospitalized",
        "deathIncrease": "Daily Deaths",
        "positiveIncrease": "Daily Positive Tests",
        "percentPositive": "Percent of Tests Positive",
        "totalTestResultsIncrease": "Daily Tests",
    }
    mobility_cols = {
        "retail_and_recreation_percent_change_from_baseline": "Retail/Recreation Mobility",
        "grocery_and_pharmacy_percent_change_from_baseline": "Grocery/Pharmacy Mobility",
        "parks_percent_change_from_baseline": "Parks Mobility",
        "transit_stations_percent_change_from_baseline": "Transit Stations Mobility",
        "workplaces_percent_change_from_baseline": "Workplaces Mobility",
        "residential_percent_change_from_baseline": "Residential Mobility",
    }
    df = df.rename(col_map)
    df = df.rename(mobility_cols)

    cols = list(df.columns)
    cols.extend(["Case Fatality Rate", "Infection Fatality Rate"])
    return df, cols


def get_correlations(df, cols):
    st.header("Correlations")
    df = df[cols]
    cor_table = df.corr(method="pearson", min_periods=30)
    st.write(cor_table)
    max_r = 0
    max_idx = None
    seen = []
    cors = pd.DataFrame(columns=["a", "b", "r"])
    for i in cor_table.index:
        for j in cor_table.index:
            if i == j or i == "index" or j == "index":
                continue
            if cor_table.loc[i, j] == 1:
                continue
            if cor_table.loc[i, j] > max_r:
                max_idx = (i, j)
                max_r = max(cor_table.loc[i, j], max_r)
            if (j, i) not in seen:
                cors = cors.append(
                    {"a": i, "b": j, "r": cor_table.loc[i, j]}, ignore_index=True
                )
                seen.append((i, j))
    st.write(max_idx, max_r)
    st.write(cors.sort_values("r", ascending=False).reset_index(drop=True))


if __name__ == "__main__":
    st.info("Note: My main data source, covidtracking.com, has shut down their operation. The data here was last updated March 7, 2021. I will migrate to a new data source in the future.")
    # todo global cols lists. One for cors and one for UI
    cols = [
        "Infection Fatality Rate",
        "positiveIncrease",
        "deathIncrease",
        "hospitalizedCurrently",
        "inIcuCurrently",
        "percentPositive",
        "totalTestResultsIncrease",
        "Case Fatality Rate",
    ]
    cols.extend(
        [
            "retail_and_recreation_percent_change_from_baseline",
            "grocery_and_pharmacy_percent_change_from_baseline",
            "parks_percent_change_from_baseline",
            "transit_stations_percent_change_from_baseline",
            "workplaces_percent_change_from_baseline",
            "residential_percent_change_from_baseline",
        ]
    )

    # Disabled data download due to the end of covidtracking.com
    # https://covidtracking.com/analysis-updates/covid-tracking-project-end-march-7
    # todo switch to Johns Hopkins Github repo
    # download_data(wait_hours=4)

    w, h, = (
        900,
        400,
    )
    states = pd.read_csv("states_daily.csv")["state"].unique()

    with st.sidebar:
        st.title("Covid-19 Data Explorer")
        st.subheader("Select a page below:")
        mode = st.radio(
            "Menu",
            [
                "Population Immunity and Vaccination",
                "Correlations Forecast",
                "Correlation Explorer",
                "ARIMA Forecast",
            ],
        )
        st.subheader("Select a state or all US states:")
        all_states = st.checkbox("All US States", True)
        locations = np.append(["USA Total"], states)
        state = st.selectbox("State", states, index=37)

    # https://docs.streamlit.io/en/stable/troubleshooting/caching_issues.html#how-to-fix-the-cached-object-mutated-warning
    df = copy.deepcopy(process_data(all_states, state))
    df_arima = copy.deepcopy(df)

    if mode == "Correlations Forecast":
        st.title("Correlations Forecast")
        # df,cols= rename_columns(df)
        b = st.selectbox("Choose a variable:", cols, index=2)
        # lookback = st.slider('How far back should we look for correlations?', min_value=0, max_value=len(df),
        #                      value=len(df) - 70,
        #                      step=10, format="%d days")
        lookback = len(df) - 70
        cors_df = get_cor_table(cols, lookback, df)

        days_back = forecast_ui(cors_df, lookback)
        lines, cors_table = compute_weighted_forecast(days_back, b, cors_df)

        if len(cors_table) < 3:
            st.warning(
                "Few correlations found. Forecast may not be accurate. Try another variable."
            )

        plot_forecast(lines, cors_table)

        st.markdown(
            """
        ## How is this forecast made?

        This forecast is a weighted average of variables from the table below. $shift$ is the number of days $a$ is shifted forward, and $r$ is the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between shifted $a$ and $b$.
        """
        )
        st.write(cors_table)

        st.markdown(
            """ 
        ## Further Explanation
        The model searches every combination of $a$, $b$, and $shift$ for the highest $r$ values. Only correlations $>0.5$ are used. $r$ is used to weight each component of the forecast, and each component is scaled and aligned to the forecasted variable $b$. The forecast length is the average $shift$ weighted by the average $r$.

        Ordinary Least Squares regression is also used to scale each series from the *a* column as well as the final forecast.
        """
        )
    elif mode == "Correlation Explorer":
        st.title("Interactive Correlation Explorer")
        st.write("Choose two variables and see if they are correlated.")
        cols, a, b, lookback = get_shifted_correlations(df, cols)

    elif mode == "ARIMA Forecast":
        arima_ui(df_arima, cols)

    elif mode == "Population Immunity and Vaccination":
        pop_immunity(df)

    st.markdown(
        """
        ### Sources 
        Data is pulled daily from https://covidtracking.com. 
        Mobility data is from [google.com/covid19/mobility](https://www.google.com/covid19/mobility/). 
        Vaccination data is from the CDC and collected at https://github.com/youyanggu/covid19-cdc-vaccination-data
        """
    )
    st.markdown(
        "Infection fatality rate and true infections are estimated using the formula described by https://covid19-projections.com/estimating-true-infections-revisited:"
    )
    st.latex("prevalenceRatio({day_{i}}) = (1500/(day_i+50)) * positivityRate^{0.5}+2")

    st.write(
        "See this app's source code at https://github.com/remingm/covid19-correlations-forecast"
    )
    st.markdown("Created by [Michael Remington](http://www.michael-remington.com).")
    st.write(
        "Disclaimer: This site was made by a data scientist, not an infectious disease expert."
    )

    st.markdown("![visitors](https://visitor-badge.glitch.me/badge?page_id=remingm.covid)")