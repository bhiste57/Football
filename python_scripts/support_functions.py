import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import ast

# Set your API key here or load it from environment / can't push it, need to be personal
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def communicate_AI(prompt: str, picture_name: str) -> pd.DataFrame:
    """
    Send a prompt and a picture to AI to read information. The prompt must be as precise as possible -
    ask for a particular type of format etc...

    :param prompt: [str] The prompt to send to AI
    :param picture_name: [str] The name of the picture to send to AI
    :return: [pd.DataFrame] The answer
    """
    image = Image.open(os.path.join('..', 'screenshots_results', picture_name + '.png'))
    model = genai.GenerativeModel('gemini-1.5-flash')

    response = model.generate_content([prompt, image])
    text = response.text
    return text


def AI_answer_to_dict(answer: str) -> dict:
    """
    Convert the raw string answer from AI to a dictionary.
    :param answer: [str] The raw string answer
    :return: [dict] The answer as a dictionary
    """
    start = answer.find('[')
    end = answer.rfind(']')
    trimmed = answer[start:end + 1]
    data = ast.literal_eval(trimmed)
    return data


def AI_answer_to_df(answer: str) -> pd.DataFrame:
    """
    Convert the raw string answer from AI to a dataframe.
    :param answer: [str] The raw string answer
    :return: [pd.DataFrame] The answer as a dataframe
    """
    start = answer.find('[')
    end = answer.rfind(']')
    trimmed = answer[start:end + 1]
    data = ast.literal_eval(trimmed)
    return pd.DataFrame(data)


def create_prompt(league, country, season, day) -> str:
    """
    Create an intelligible prompt to send to an AI to collect data we want.

    :param league: [str] The league we're interested in - could be Premier League, Ligue 1...
    :param country: [str] The country we're interested in'
    :param season: [list] The list of two following years for the season we're interested in
    :param day: [int] The days number we're interested in
    :return: [str] The clear prompt
    """

    assert season[0] == season[-1]-1, 'The two years of a season must be following'

    prompt = (f'Get me the results of the games from {league} of {country} at day number {day} '
              f'during season {season[0]}-{season[-1]} from that picture.'
              f' Return me them as a list of dictionaries with the following '
              f'keys: season, country, league, day, home_team, away_team, home_score, away_score. '
              f"Don't write anything else in the answer")

    return prompt


def merge_days(dataframes: list) -> pd.DataFrame:
    """
    Merge multiple days to create the whole season.
    :param dataframes: [list] The list of dataframes with same season to merge
    :return: [pd.DataFrame] The merged dataframe
    """
    for df in dataframes:
        assert len(df.season.unique().tolist()) == 1, f'Dataframe of day {df.day.iloc[0]} got different seasons.'
    full_season = pd.concat(dataframes).sort_values('day').reset_index(drop=True)
    return full_season


def merge_seasons(dataframes: list) -> pd.DataFrame:
    """
    Merge multiple seasons to create the whole league.
    :param dataframes: [list] The list of dataframes with same league to merge
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
    file_path = os.path.join('..', 'database.xlsx')

    if os.path.exists(file_path):
        user_input = input(f"⚠️ The file '{file_path}' already exists. Overwrite? (y/n): ").strip().lower()
        if user_input != 'y':
            print("❌ Export cancelled by user.")
            return
    dataframe.sort_values(['country','league','season', 'day'], ignore_index=True).to_excel(file_path, index=False)
    print(f"✅ Data exported to '{file_path}' successfully.")


def fetch_excel_database() -> pd.DataFrame:
    """
    Fetch the database contains in the Excel spreadsheet and return as a DataFrame.
    :return: [pd.DataFrame] The full results dataframe
    """
    return pd.read_excel(os.path.join('..', 'database.xlsx'))
