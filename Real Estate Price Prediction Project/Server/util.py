import json
import pickle
import numpy as np
import warnings

# Suppress the warning
warnings.filterwarnings("ignore")

locations = None
data_columns = None
model = None

def get_estimated_price(location,sqft,bhk,bath):
    try:
        loc_index = data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index>=0:
        x[loc_index] = 1

    return round(model.predict([x])[0],2)

def get_location_names():
    return locations

def load_saved_artifacts():
    print("Start...loading saved artifacts...") 
    global data_columns
    global locations
    with open("./Artifacts/columns.json", 'r') as file:
        data_columns = json.load(file)['data_columns']
        locations = data_columns[3:]

    global model
    with open("./Artifacts/banglore_home_prices_model.pickle", 'rb') as file:
        model = pickle.load(file)
    print("Stop...loading saved artifacts is done...")

if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000, 3, 3))
    print(get_estimated_price('1st Phase JP Nagar', 1000, 2, 2))
    print(get_estimated_price('Kalhalli', 1000, 2, 2)) 
    print(get_estimated_price('Ejipura', 1000, 2, 2))