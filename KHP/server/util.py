import pickle
import json
import numpy as np

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,bhk,total_sq_ft,emi):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1

    x = np.zeros(len(__data_columns))
    x[0] = bhk
    x[1] = total_sq_ft
    x[2] = emi
    if loc_index>=0:
        x[loc_index] = 1

    return round(__model.predict([x])[0],2)*1000

def get_location_names():
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __data_columns
    global __locations
    with open('./artifacts/columns.json', "r") as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:] 

    global __model
    if __model is None:
        with open('./artifacts/Kolkata_house_price_predicator_1.pickle', 'rb') as f:
            __model = pickle.load(f)
    print("loading saved artifacts...done")


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('tollygunge',5, 2600, 54))
    print(get_estimated_price('madhaymgram', 4, 900, 26))