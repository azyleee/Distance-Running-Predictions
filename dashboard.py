import streamlit as st
import pandas as pd
from helpers.distance_running import read_strava_csv, get_best_model, user_finetune, minutes2hms, prepare_user_data

st.markdown("""
    <style>
        /* font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');
        *, *::before, *::after {
            font-family: 'Inter', sans-serif;
        }
    </style>
    #
""", unsafe_allow_html=True)

if 'ft_handler' not in st.session_state:
    st.session_state.ft_handler = None

st.title('Distance Running Predictions')

st.header('')
col1, col_, col2 = st.columns([0.45,0.1,0.45])

with col1:
    st.header('1. Upload Strava Activities')

    user_file = st.file_uploader('')

    if user_file:

        st.header('2. Enter Race Details')

        race_info = dict()

        race_info['distance (m)'] = st.number_input("Race Distance (km)")
        race_info['distance (m)'] *= 1.02*1000.0

        race_info['elevation gain (m)'] = st.number_input("Course Elevation Gain (m)")
        race_info['average heart rate (bpm)'] = st.number_input("Target Heart Rate")
        race_info['timestamp'] = st.date_input("Race Date")
        race_info['timestamp'] = race_info['timestamp'].strftime("%Y-%m-%d")

        if user_file and race_info['distance (m)'] != 0.0 and race_info['average heart rate (bpm)'] != 0.0:

            st.header('3. Fit ML Model')

            # convert dict to pd.DataFrame
            for key, value in race_info.items():
                race_info[key] = [value]
            race_info = pd.DataFrame(race_info)

            user_df = read_strava_csv(user_file)

            fit_button = st.button('CREATE')

            if fit_button:

                base_handler, x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor, x_race_tensor, y_race_tensor, result  = get_best_model(user_df, race_info, model_folder = 'base_models_complete')

                ft_handler = user_finetune(
                    base_handler, 
                    x_train_tensor, 
                    x_test_tensor, 
                    y_train_tensor, 
                    y_test_tensor,
                    epochs = 3,
                    batch_size = 1,
                    patience = 25
                )

                st.session_state.ft_handler = ft_handler

with col2:

    if st.session_state.ft_handler is not None:
        
        st.header('Prediction')

        base_dataobject = st.session_state.ft_handler.dataobject
        x_race_tensor = prepare_user_data(base_dataobject, user_df, race_info)[4]

        prediction = st.session_state.ft_handler.predict(x_race_tensor, scaled = True)
        time = prediction[0]
        hours, minutes, seconds = minutes2hms(time)

        st.write('Predicted Race Time: ')
        if hours > 0:
            st.subheader(f'{hours}h {minutes}m {seconds}s')
        else:
            st.subheader(f'{minutes}m {seconds}s')

        confidence = st.session_state.ft_handler.testing_losses_mape[-1]
        st.write(f'Confidence: {(1-confidence)*100:.2f}%')