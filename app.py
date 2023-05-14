import datetime

import dash
from dash import Dash, dcc, html, dcc, Input, Output, State
from datetime import datetime as dt
# import dash_core_components as dcc
# import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import main as m
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from dash_bootstrap_templates import load_figure_template

load_figure_template("slate")

pd.options.mode.chained_assignment = None  # default="warn"
import dash_bootstrap_components as dbc


app = Dash(external_stylesheets=[dbc.themes.SLATE],
           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

sidebar = html.Div(
    [dbc.Row(
        [
            html.Div([
                html.P("Select a time range for the Analysis",
                       style={"margin-top": "14px", "margin-left": "0px"}),

                dcc.DatePickerRange(id="Date-Picker",
                                    number_of_months_shown=2,
                                    min_date_allowed=dt(2017, 1, 1).date(),
                                    max_date_allowed=dt(2023, 1, 11).date(),
                                    calendar_orientation="horizontal",
                                    initial_visible_month=dt(2022, 12, 1).date(),
                                    start_date=dt(2021, 12, 31).date(),
                                    end_date=dt(2023, 1, 11).date())

            ]

            )
        ],

    ),
        dbc.Row(
            [
                html.Div([
                    html.P("Select a Ticker from the List",
                           style={"margin-top": "15px"}),

                    dcc.Dropdown(
                        id="dropdown-Ticker",
                        options=["MSFT", "META", "TTE", "AMZN", "XOM", "TSLA", "SPY", "AAPL", "GOOG", "MCD", "CVX"],
                        value="TSLA",
                        placeholder="Select a ticker from the list",
                        multi=False,
                        style={"color": "black"}
                    ),

                    html.P("Select an Indicator from the List",
                           style={"margin-top": "15px"}),
                    dcc.Dropdown(
                        id="dropdown-Indicator",
                        options=["RSI", "Stochastic Oscilator", "Sentiment Analysis", "WILLR", "MACD"],
                        placeholder="Select a Indicator from the list",
                        multi=False,
                        value="Stochastic Oscilator",
                        style={"color": "black"}
                    )

                ])],

        ),
        dbc.Row(
            [

                html.P("Input Indicator variables",
                       style={"margin-top": "25px"}),
                html.Hr(),
                html.Div([html.P("Input the long window",
                                 style={"margin-top": "8px", "align": "center", "color": "white"}),
                          dcc.Input(size="sm", id="Long Window",
                                    type="number",
                                    value=13,
                                    required=True,
                                    min=0,
                                    style={"display": "block", "margin-top": "4px"})], id="Long_Window_Div"),

                html.Div([
                    html.P("Input the short window",
                           style={"margin-top": "4px", "align": "center", "color": "white"}),
                    dcc.Input(id="Short Window",
                              type="number",
                              required=True,
                              value=3,
                              min=0,
                              style={"display": "block"})], id="Short_Window_Div"),
                html.Div([
                    html.P("Input your MACD Signal line",
                           style={"margin-top": "4px", "align": "center", "color": "white"}),
                    dcc.Input(id="MACD Sig",
                              type="number",
                              required=True,
                              value=9,
                              min=0,
                              style={"display": "block"})
                ], id="MACD_Sig_Div")],

            className="bg-primary text-white font-italic dropdown-item")
    ]
)

content = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row([

                            html.P(id="Ticker-Chart-Headline", style={"textAlign": "center"}),
                            dcc.Graph(id="line-graph"),

                        ]),
                        dbc.Row([
                            html.P(id="Indicator-Chart-Headline", style={"textAlign": "center"}),
                            dcc.Graph(id="Indicator-Graph")
                        ])

                    ], width={"size": 7},
                    className="bg-dark text-white"
                ),
                dbc.Col(
                    [
                        html.P(id="Backtesting-Graph-Headline", style={"textAlign": "center"}),
                        dcc.Graph(id="Backtesting-Graph", responsive=True)

                    ], width={"size": 5},
                    className="bg-dark text-white"
                )
            ],
            style={
                "margin-top": "0px", "margin-left": "8px",
                "margin-bottom": "8px"}),

    ]
)


def transform_to_list(input):
    if isinstance(input, list):
        # input is already a list, return it as is
        return input
    elif isinstance(input, str):
        # input is a string, convert it to a list and return it
        return input.split()
    else:
        # input is not a string or a list, raise an error
        raise ValueError("Input must be a string or a list")


app.layout = dbc.Container(
    [
        html.H1("Stock market trading Dashboard", id="Headline",
                style={"color": "white", "textAlign": "center", "font-weight": "bold"}),
        dbc.Row(
            [
                dbc.Col(sidebar, width=2, className="text-white bg-primary font-italic"),
                dbc.Col(content, width=10, className="text-white font-italic")
            ],

        ),
    ],
    fluid=True
)


@app.callback(
    Output("line-graph", "figure"),
    Output("Ticker-Chart-Headline", "children"),
    [Input("dropdown-Ticker", "value")],
    [Input("Date-Picker", "start_date")],
    [Input("Date-Picker", "end_date")],
    [Input("dropdown-Indicator", "value")])
# [Input("SMA_CheckBox", "value")])
def update_line_chart(ticker, start, end, indicator):
    dff = m.df
    dim = transform_to_list(ticker)

    chosenTicker = dff.loc[(dff["Ticker"].isin(dim)) & (dff["Date"] >= start) & (dff["Date"] <= end)]

    fig = go.Figure(data=[go.Candlestick(x=chosenTicker["Date"],
                                         open=chosenTicker["Open"],
                                         high=chosenTicker["High"],
                                         low=chosenTicker["Low"],
                                         close=chosenTicker["Close"], name=ticker)])
    fig.update_layout(height=300, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1
                      ))

    # remove line breaks at weekends
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig, f"Price Chart of {ticker} in ($)"


@app.callback(
    Output("Indicator-Chart-Headline", "children"),
    Output("Short Window", "min"),
    # Output("Long Window", "children"),
    Output("Long_Window_Div", "style"),
    Output("Short_Window_Div", "style"),
    Output("MACD_Sig_Div", "style"),
    [Input("dropdown-Indicator", "value")],
    [Input("dropdown-Ticker", "value")])
def update_indicator_input(indicator, ticker):
    if indicator in ["Stochastic Oscilator"]:
        return f"{indicator} Indicator for {ticker}", 0, {"display": "block"}, {"display": "block"}, {"display": "None"}
    if indicator in ["RSI"]:
        return f"{indicator} Indicator for {ticker}", 2, {"display": "block"}, {"display": "None"}, {"display": "None"}
    if indicator in ["WILLR"]:
        return f"{indicator} Indicator for {ticker}", 2, {"display": "block"}, {"display": "None"}, {"display": "None"}
    if indicator in ["MACD"]:
        return f"{indicator} Indicator for {ticker}", 0, {"display": "block"}, {"display": "block"}, {
            "display": "block"}
    if indicator in ["Sentiment Analysis"]:
        return f"{indicator} Indicator for {ticker}", 0, {"display": "None"}, {"display": "None"}, {"display": "None"}


@app.callback(
    Output("Indicator-Graph", "figure"),
    [Input("dropdown-Indicator", "value")],
    [Input("dropdown-Ticker", "value")],
    [Input("Date-Picker", "start_date")],
    [Input("Date-Picker", "end_date")],
    [Input("Short Window", "value")],
    [Input("Long Window", "value")],
    [Input("MACD Sig", "value")])
def update_Indicators(indicator, ticker, start, end, sW, lW, sig):
    # print("#################### Backtesting Chart ##########################")
    # print(indicator)
    # print(start)
    # print(end)
    fig = go.Figure()
    df = m.df
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.02,
                          xanchor="right",
                          x=1
                      ))

    # Stochastic Oscilator
    if indicator in ["Stochastic Oscilator"] and (sW != None) and (lW != None):
        df = df.loc[(df["Ticker"] == ticker) & (df["Date"] >= start) & (df["Date"] <= end)]
        df = m.add_indicators(df=df, period_d=sW, period_k=lW)

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df[f"STOCHk_{lW}_{sW}"],
            line=dict(color="#3D9970", width=1),
            name="Fast %K",
            showlegend=True
        ))

        # Slow signal (%d)
        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df[f"STOCHd_{lW}_{sW}"],
                line=dict(color="#000000", width=1),
                name="Slow %D",
                showlegend=True
            ))

        # Add overbought/oversold
        fig.add_hline(y=20, line_color="#FF0000", line_width=1)
        fig.add_hline(y=80, line_color="#00FF00", line_width=1)

        fig.update_yaxes(range=[0, 100])

    # Relative Strength index
    if indicator in ["RSI"] and (lW != None):
        df = df.loc[(df["Ticker"] == ticker) & (df["Date"] >= start) & (df["Date"] <= end)]
        df = m.add_indicators(df=df, rsiLength=lW)

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df[f"RSI{lW}"],
            line=dict(color="#ff9900", width=2),
            name=f"RSI {lW}",
            showlegend=True
        ))

        # Add overbought/oversold
        fig.add_hline(y=30, line_color="#FF0000", line_width=1)
        fig.add_hline(y=70, line_color="#00FF00", line_width=1)

        fig.update_yaxes(range=[0, 100])

    # Williams %R
    if indicator in ["WILLR"] and (lW != None):
        df = df.loc[(df["Ticker"] == ticker) & (df["Date"] >= start) & (df["Date"] <= end)]
        df = m.add_indicators(df=df, WillrLength=lW)

        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df[f"WILLR{lW}"],
            line=dict(color="#ff9900", width=2),
            name=f"WILLR {lW}",
            showlegend=True
        ))

        # Add overbought/oversold
        fig.add_hline(y=-20, line_color="#00FF00", line_width=1)
        fig.add_hline(y=-80, line_color="#FF0000", line_width=1)

        fig.update_yaxes(range=[-100, 0])

    # MACD
    if indicator in ["MACD"] and (sW != None) and (lW != None) and (sig != None):
        df = df.loc[(df["Ticker"] == ticker) & (df["Date"] >= start) & (df["Date"] <= end)]
        df = m.add_indicators(df=df, MACDFast=sW, MACDSlow=lW, MACDSig=sig)
        # print(df)
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df[f"MACD_{sW}_{lW}_{sig}"],
            line=dict(color="#ff9900", width=1),
            name="MACD",
            showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df[f"MACDs_{sW}_{lW}_{sig}"],
            line=dict(color="#000000", width=1),
            name="Signal",
            showlegend=True
        ))

        # add colors to the histogram
        colors = ["green" if macd >= 0 else "red" for macd in df[f"MACDh_{sW}_{lW}_{sig}"]]

        # Plot the histogram
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df[f"MACDh_{sW}_{lW}_{sig}"],
                name="histogram",
                marker_color=colors,
                showlegend=True
            )
        )

    if indicator in ["Sentiment Analysis"]:
        df = m.read_sentiment_values(ticker)

        # add colors to the histogram
        colors = ["green" if val > 0 else "red" for val in df["sentiment_value"]]
        # print(df)
        # Plot the histogram
        fig.add_trace(
            go.Bar(
                x=df["Date"],
                y=df["sentiment_value"],
                name="sentiment",
                marker_color=colors,
            )
        )
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    return fig


@app.callback(
    Output("Backtesting-Graph-Headline", "children"),
    Output("Backtesting-Graph", "figure"),
    [Input("dropdown-Ticker", "value")],
    [Input("Date-Picker", "start_date")],
    [Input("Date-Picker", "end_date")])
def update_Backtesting_Graph(ticker, start, end):

    df = m.add_Backtesting_Ratios(m.df, start, end)
    df = df.loc[(df["Ticker"].isin([ticker, "SPY"]))]
    df = pd.melt(df, id_vars=["Ticker"],
                 value_vars=["Sharpe Ratio", "Total Return", "Sortino Ratio", "Annulized Volatility",
                             "Treynor Ratio"]).round(2)

    fig = px.bar(df, x="value", y="variable", color="Ticker", text="value", barmode="group", orientation="h",
                 color_discrete_map={"SPY": "grey"})
    fig.update_yaxes(visible=True, showticklabels=True, showgrid=True, zeroline=True,
                     categoryorder='array',
                     categoryarray=["Sharpe Ratio", "Sortino Ratio", "Treynor Ratio", "Annulized Volatility",
                                    "Total Return"])
    fig.update_xaxes(title="", visible=True, showticklabels=False, showgrid=False, zeroline=True)
    fig.update_traces(textangle=0)
    fig.update_layout(

        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)"
    )

    return f"Backtesting Ratios for {ticker} compared to the SPY ETF", fig


@app.callback(
    Output("Date-Picker", "start_date"),
    Output("Date-Picker", "end_date"),
    [Input("dropdown-Indicator", "value")],
    [Input("Date-Picker", "start_date")],
    [Input("Date-Picker", "end_date")]
)
def update_DatePicker_Component(indicator, start, end):
    start = dt.strptime(start, "%Y-%m-%d").date()
    end = dt.strptime(end, "%Y-%m-%d").date()

    min_date = dt(2022, 12, 18).date()
    max_date = dt(2022, 1, 8).date()

    if indicator in ["Sentiment Analysis"]:
        if start < min_date or end > max_date:
            return dt(2022, 12, 18).date(), dt(2023, 1, 8).date()
        return start, end
    else:
        return start, end


if __name__ == "__main__":
    app.run_server(debug=False)
