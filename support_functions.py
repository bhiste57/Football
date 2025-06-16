import openai
import pandas as pd
import os

# Set your API key here or load it from environment
openai.api_key = os.getenv("OPENAI_API_KEY")


def communicate_AI(prompt: str) -> pd.DataFrame:
    """
    Send a prompt to ChatGPT and collect the answer. The prompt must be as precise as possible -
    ask for a particular type of format etc...

    :param prompt: [str] The prompt to send to ChatGPT
    :return: [pd.DataFrame] The answer
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return pd.DataFrame( response['choices'][0]['message']['content'].strip() )


def create_prompt(league, country, season, round) -> str:
    """
    Create an intelligible prompt to send to an AI to collect data we want.

    :param league: [str] The league we're interested in - could be Premier League, Ligue 1...
    :param country: [str] The country we're interested in'
    :param season: [list] The list of two following years for the season we're interested in
    :param round: [int] The rounds number we're interested in
    :return: [str] The clear prompt
    """

    assert season[0] == season[-1]-1, 'The two years of a season must be following'

    prompt = (f'Can you get me all the results of the games from {league} of {country} at round number {round} '
              f'during season {season[0]}-{season[-1]}. Return me them as a list of dictionaries with the following '
              f'keys: season, country, league, round, home_team, away_team, home_score, away_score.')

    return prompt


def merge_rounds(dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multiple rounds to create the whole season.
    :param dataframes: [pd.DataFrame] The dataframes with same season to merge
    :return: [pd.DataFrame] The merged dataframe
    """
    for df in dataframes:
        assert len(df.season.unique().tolist()) == 1, f'Dataframe of round {df.round.iloc[0]} got different seasons.'
    full_season = pd.concat(dataframes).sort_values('round').reset_index(drop=True)
    return full_season


def merge_seasons(dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    Merge multiple seasons to create the whole league.
    :param dataframes: [pd.DataFrame] The dataframes with same league to merge
    :return: [pd.DataFrame] The merged dataframe
    """
    for df in dataframes:
        assert (len(df.league.unique().tolist())==1 and len(df.country.unique().tolist())==1),\
            f'Dataframe of season {df.season.iloc[0]} got different leagues.'
    full_league = pd.concat(dataframes).sort_values('season').reset_index(drop=True)
    return full_league


def create_excel_database(dataframe: pd.DataFrame):
    """
    Export the full results dataframe to an Excel spreadsheet.
    :param dataframe: [pd.DataFrame] The full results dataframe
    :return:
    """
    dataframe.to_excel(r'Football/database.xlsx', index=False)