import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import numpy as np

with open('mejor_modelo.pkl', 'rb') as file:
    model = pickle.load(file)


def try_sex(input_df):
    # Crear un DataFrame vacío con las mismas columnas que input_df
    empty_df = pd.DataFrame(
        columns=[
            'age',
            'sex',
            'education',
            'marital-status',
            'occupation',
            'race_income_ratio',
            'hours-per-week',
            'country_income_ratio'
        ]
    )

    ### Education
    education_order = {'Pre-HS': 15,'High-School': 25,'College': 35,'Bachelors': 50, 'Postgraduate': 60}
    input_df['education'] = input_df['education'].map(education_order)

    #### Marital status 
    input_df['marital-status'] = input_df['marital-status'].replace({
        'Married-AF-spouse': 30,'Married-civ-spouse': 30,'Married-spouse-absent': 30,'Never-married': 45,
        'Divorced': 35,'Separated': 45,'Widowed': 45 })

    ### Occupation
    occupation_mapping = {
        'Prof-specialty': 'High-Income',
        'Exec-managerial': 'High-Income',
        'Tech-support': 'Mid-Income',
        'Sales': 'Mid-Income',
        'Craft-repair': 'Mid-Income',
        'Protective-serv': 'Mid-Income',
        'Adm-clerical': 'Low-Income',
        'Farming-fishing': 'Low-Income',
        'Machine-op-inspct': 'Low-Income',
        'Transport-moving': 'Low-Income',
        'Handlers-cleaners': 'Low-Income',
        'Other-service': 'Low-Income',
        'Priv-house-serv': 'Low-Income',
        'Armed-Forces': 'Mid-Income'
    }
    occupation_order = {
        'Low-Income': 30,
        'Mid-Income': 40,
        'High-Income': 50,
        np.nan: np.nan
    }
    input_df['occupation'] = input_df['occupation'].map(occupation_mapping)
    input_df['occupation'] = input_df['occupation'].map(occupation_order)

    ### Race
    input_df['race_income_ratio'] = input_df['race_income_ratio'].replace({
        'White': 'White',
        'Black': 'Black',
        'Asian-Pac-Islander': 'Other',
        'Amer-Indian-Eskimo': 'Other',
        'Other': 'Other'
    })
    race_vals = {
        'Black': 0.120811,
        'White': 0.253987,
        'Other': 0.214614
    }
    input_df['race_income_ratio'] = input_df['race_income_ratio'].map(race_vals)
    input_df['race_income_ratio'] = input_df['race_income_ratio'] * 200

    ### Country
    less_frequent_countries = [
        'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador',
        'India', 'Cuba', 'England', 'China', 'South', 'Jamaica', 'Italy',
        'Dominican-Republic', 'Japan', 'Guatemala', 'Poland', 'Vietnam',
        'Columbia', 'Haiti', 'Portugal', 'Taiwan', 'Iran', 'Greece',
        'Nicaragua', 'Peru', 'Ecuador', 'France', 'Ireland', 'Hong', 'Thailand',
        'Cambodia', 'Trinadad&Tobago', 'Laos', 'Yugoslavia',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Honduras', 'Hungary',
        'Holand-Netherlands'
    ]
    input_df['country_income_ratio'] = input_df['country_income_ratio'].replace(less_frequent_countries, 'Other')
    country_vals = {'Mexico': 0.049422, 'Other': 0.226733, 'United-States': 0.243977}
    input_df['country_income_ratio'] = input_df['country_income_ratio'].map(country_vals)
    input_df['country_income_ratio'] = input_df['country_income_ratio'] * 200

    # Obtener la única fila de input_df y modificar la columna 'sex'
    row = input_df.iloc[0].copy()
    row['sex'] = 1 if row['sex'] == 'Male' else (0 if row['sex'] == 'Female' else np.nan)
    
    # Convertir la fila en un DataFrame de una sola fila
    row_df = pd.DataFrame([row], columns=empty_df.columns)
    
    # Concatenar el DataFrame vacío con la fila modificada
    empty_df = pd.concat([empty_df, row_df], ignore_index=True)

    for col in empty_df.columns:
        empty_df[col] = pd.to_numeric(empty_df[col], errors='coerce')

    
    return empty_df.head()


##################################################################################################

# Interfaz de Streamlit
st.title('TFM: Predicción de Ingresos',)
st.info('Machine learning app para predecir si una persona ganará más o menos de 50K dolares al año.')
st.header("Ingresa tus datos", divider=True)

# Entradas del usuario
age = st.slider('Edad', min_value=18, max_value=90)

sex = st.selectbox('Sexo', ['Female', 'Male'])

education = st.selectbox('Educación', ['Pre-HS','High-School','College','Bachelors','Postgraduate'])

marital_status = st.selectbox('Estado Civil', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 
                                               'Married-spouse-absent', 'Married-AF-spouse'])

occupation = st.selectbox('Ocupación', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 
                                        'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                                          'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])

race = st.selectbox('Raza', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])

hours_per_week = st.slider('Horas trabajadas por Semana', min_value=0, max_value=70)

native_country = st.selectbox('País de Residencia', ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 
                                                 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 
                                                 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 
                                                 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 
                                                 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 
                                                 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 
                                                 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])


# Crear un diccionario con los datos de entrada
cust_data = {
    'age': age,
    'sex': sex,
    'education': education,
    'marital-status': marital_status,
    'occupation': occupation,
    'race_income_ratio': race,
    'hours-per-week': hours_per_week,
    'country_income_ratio': native_country,
}



# Botón para predecir
if st.button('Check data'):
    input_df = pd.DataFrame.from_dict(cust_data, orient='index').T
    preprocessed_data = try_sex(input_df)

    # Asegúrate de que las columnas están en el orden correcto
    trained_features = model.get_booster().feature_names
    st.write(preprocessed_data)
    preprocessed_data = preprocessed_data[trained_features]
    
    # Realizar la predicción
    prediction = model.predict(preprocessed_data)
    st.write("Prediction:", prediction)


