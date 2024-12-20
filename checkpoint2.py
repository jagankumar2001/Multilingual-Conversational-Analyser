import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import random
import speech_recognition as sr
from googletrans import Translator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime

# Simulating dynamic engagement values (replace with real data collection logic)
def simulate_dynamic_engagement(speaker_1_text, speaker_2_text):
    speaker_1_words = len(speaker_1_text.split())
    speaker_2_words = len(speaker_2_text.split())

    sentiment_score_1 = random.uniform(0, 1)
    sentiment_score_2 = random.uniform(0, 1)

    active_1 = speaker_1_words * sentiment_score_1
    active_2 = speaker_2_words * sentiment_score_2

    total_active = active_1 + active_2
    if total_active == 0:
        active_1_normalized = 50
        active_2_normalized = 50
    else:
        active_1_normalized = (active_1 / total_active) * 100
        active_2_normalized = (active_2 / total_active) * 100

    passive_1 = 100 - active_1_normalized
    passive_2 = 100 - active_2_normalized

    return {"active_speaker_1": active_1_normalized, "active_speaker_2": active_2_normalized,
            "passive_speaker_1": passive_1, "passive_speaker_2": passive_2}

# Data Collection for Speech-to-Text
class DataCollector:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def collect_audio(self, speaker_id):
        print(f"Speaker {speaker_id}, please say something...")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            audio = self.recognizer.listen(source)
            print(f"Audio captured from Speaker {speaker_id}. Processing...")
            try:
                text = self.recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return f"Could not understand the audio from Speaker {speaker_id}."
            except sr.RequestError as e:
                return f"API Error: {e}"

# Preprocessing: Cleaning Text Data
class Preprocessor:
    def __init__(self):
        self.translator = Translator()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text, target_language):
        detected_lang = self.translator.detect(text).lang
        if detected_lang != target_language:
            text = self.translator.translate(text, src=detected_lang, dest=target_language).text

        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in self.stop_words]
        return ' '.join(tokens)

# Initialize Dash App
app = dash.Dash(__name__)

# Conversation log stored as a global variable (for simplicity)
conversation_log = []

# Supported languages
LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'hi': 'Hindi',
    'zh-cn': 'Chinese (Simplified)',
    'ar': 'Arabic'
}

# Layout with button for dynamic updates and language selection
app.layout = html.Div([
    html.H1("Dynamic Student Engagement Dashboard"),
    dcc.Dropdown(
        id="language-dropdown",
        options=[{"label": name, "value": code} for code, name in LANGUAGES.items()],
        value="en",  # Default language
        placeholder="Select a language for translation"
    ),
    html.Button("Start/Stop Engagement Session", id="start-button", n_clicks=0),
    html.Div(id="status-message", children="Press the button to start collecting engagement data."),
    dcc.Graph(id="engagement-pie-chart"),
    html.H2("Conversation Log"),
    html.Div(id="conversation-table")
])

# Function to update engagement dynamically
@app.callback(
    [Output("engagement-pie-chart", "figure"), Output("conversation-table", "children")],
    [Input("start-button", "n_clicks")],
    [State("language-dropdown", "value")]
)
def update_engagement_and_log(n_clicks, target_language):
    global conversation_log

    if n_clicks > 0:
        # Instantiate the necessary classes
        collector = DataCollector()
        preprocessor = Preprocessor()

        # Collect audio dynamically from two speakers
        speaker_1_text = collector.collect_audio(speaker_id=1)
        speaker_2_text = collector.collect_audio(speaker_id=2)

        # Process the text data and translate to the selected language
        processed_text_1 = preprocessor.preprocess_text(speaker_1_text, target_language)
        processed_text_2 = preprocessor.preprocess_text(speaker_2_text, target_language)

        # Append to conversation log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conversation_log.append({"Speaker": "Speaker 1", "Timestamp": timestamp, "Text": processed_text_1})
        conversation_log.append({"Speaker": "Speaker 2", "Timestamp": timestamp, "Text": processed_text_2})

        # Simulate dynamic engagement
        engagement = simulate_dynamic_engagement(processed_text_1, processed_text_2)

        # Prepare data for pie chart
        labels = list(engagement.keys())
        values = list(engagement.values())

        # Create pie chart
        fig = px.pie(values=values, names=labels, title="Student Engagement")

        # Generate conversation table
        conversation_df = pd.DataFrame(conversation_log)
        table = html.Table([
            html.Thead(html.Tr([html.Th(col) for col in conversation_df.columns])),
            html.Tbody([
                html.Tr([html.Td(conversation_df.iloc[i][col]) for col in conversation_df.columns])
                for i in range(len(conversation_df))
            ])
        ])

        return fig, table

    return {}, html.Div("No conversations recorded yet.")

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
