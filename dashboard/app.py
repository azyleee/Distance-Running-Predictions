import streamlit as st
import sys

sys.path.append('./')

from helpers.distance_running import read_strava_csv

st.title('Distance Running Predictions')

user_csv = st.file_uploader('Upload Strava CSV')

if user_csv:
    user_df = read_strava_csv(user_csv)

    st.table(user_df)


