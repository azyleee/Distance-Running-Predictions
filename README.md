# Distance-Running-Predictions

<p align="center">
  <img src="https://github.com/azyleee/Distance-Running-Predictions/blob/main/images/alphafly-crop.png" alt="llama playing chess" width=500/>
</p>

This project created an application to predict distance running race times based on a user's Strava history.

The app is built upon a series of base models which were created in this project, and stored in ```base_models_complete```. 

The models employed are deep, but simple, PyTorch neural networks.

The app finds the best performing base model for the new user's data, then applies data transform and scaling, then fine-tuning to fit the model to the new data.

Key project steps include:

- Conducting Exploratory Data Analysis
- Performing feature engineering using a [Kaggle dataset](httpswww.kaggle.comdatasetsolegoaerrunning-races-strava)
- Applying data cleaning, mining, transformation, and scaling techniques

The entire process was conducted in ```main.ipynb```, with modules contained in the ```helpers``` folder.

The application achieved a notable 97% accuracy in race time prediction for the creator's 5K time.

<!-- To try out the application yourself, visit the deployed [dashboard on Streamlit](https://distance-running-predictions.streamlit.app/)" -->

# Prerequisite Libraries
* ```numpy```
* ```pandas```
* ```matplotlib```
* ```plotly```
* ```datetime```
* ```scikit-learn```
* ```scipy```
* ```torch```