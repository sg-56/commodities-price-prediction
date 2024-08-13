import streamlit as st
import pandas as pd
import os
import sys
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV
from src.logger import logging



def hide_sidebar():
    return st.markdown(
    """
    <style>
        [data-testid="collapsedControl"] {
        display: none
        }
    </style>
    """,
    unsafe_allow_html=True,
    )

def GetCommodities():
    return pd.read_csv('notebooks/Dataset.csv',usecols=['commodity']).commodity.unique()

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        result = pd.DataFrame(columns=['name',
                        'Train_MAE',
                        'Test_MAE',
                        'Train_MSE',
                        'Test_MSE',
                        'Train_RMSE',
                        'Test_RMSE',
                        'Train_R2',
                        'Test_R2'])
        for current_model in models:
            model = models[current_model].fit(X_train,y_train)
            logging.info(f"Fitting of Model : {current_model} completed")
            train_preds = model.predict(X_train)
            test_preds = model.predict(X_test)
            #R2 Score
            train_r2_score = r2_score(y_train, train_preds)
            test_r2_score = r2_score(y_test,test_preds)
            #mean Absolute Error
            train_mae = mean_absolute_error(y_train,train_preds)
            test_mae = mean_absolute_error(y_test,test_preds)
            #MSE
            train_mse = mean_squared_error(y_train,train_preds)
            test_mse = mean_squared_error(y_test,test_preds)
            #RMSE
            train_rmse = root_mean_squared_error(y_train,train_preds)
            test_rmse = root_mean_squared_error(y_test,test_preds)
            metrics = { 'name':[current_model],
                        'Train_MAE':[train_mae],
                        'Test_MAE':[test_mae],
                        'Train_MSE':[train_mse],
                        'Test_MSE':[test_mse],
                        'Train_RMSE':[train_rmse],
                        'Test_RMSE':[test_rmse],
                        'Train_R2':[train_r2_score],
                        'Test_R2':[test_r2_score]
                        }
            model_result = pd.DataFrame(metrics)
            result =  pd.concat([result,model_result])
        print(result)
        return result

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def getinfo(commodity_name):
    coms = {
  "Rice": "A versatile and staple grain consumed globally, rice is a key ingredient in many cuisines, used as a base for dishes like biryani, fried rice, risotto, and sushi. It comes in various varieties such as basmati, jasmine, and short-grain rice, each with unique texture and flavor profiles.",
  "Vanaspati Packed": "A type of hydrogenated vegetable oil, vanaspati is commonly used in Indian cooking as a cheaper substitute for ghee (clarified butter). It is solid at room temperature and is often used in baking, frying, and making sweets and savory snacks.",
  "Gram Dal": "Also known as chana dal, these are split chickpeas that are an important source of protein in Indian cuisine. They are used in a variety of dishes, including soups, stews, curries, and snacks like dal vada.",
  "Tur Arhar Dal": "Split pigeon peas, also known as toor dal, are widely used in Indian cuisine. They are a staple in many households, used to make dals (lentil soups), sambars, and other dishes. They are known for their earthy flavor and high protein content.",
  "Urad Dal": "Also known as black gram, urad dal is used in various Indian dishes such as dals, soups, and South Indian staples like idlis, dosas, and vadas. It has a rich, creamy texture and is highly nutritious, providing protein and dietary fiber.",
  "Moong Dal": "Split mung beans, known for their quick cooking time and nutritional benefits. Moong dal is used in soups, stews, dals, and even desserts. It is a great source of protein, fiber, vitamins, and minerals, making it a staple in vegetarian diets.",
  "Masoor Dal": "Red lentils, which cook quickly and are commonly used in soups, stews, and Indian dals. They have a mild, earthy flavor and a soft texture when cooked. Masoor dal is rich in protein, fiber, and essential nutrients.",
  "Sugar": "A sweetener derived from sugarcane or sugar beet, used in a wide range of culinary applications, including baking, cooking, and beverages. Sugar comes in various forms, such as granulated, powdered, and brown sugar, each with specific uses.",
  "Milk": "A nutritious liquid produced by mammals, commonly consumed as a beverage and used in cooking and baking. Milk is a source of calcium, protein, and vitamins, and it comes in various forms like whole milk, skim milk, and plant-based alternatives like almond and soy milk.",
  "Groundnut Oil Packed": "Oil extracted from peanuts, known for its high smoking point and mild flavor. Groundnut oil is commonly used for frying, sautéing, and as a base for dressings and sauces. It is a good source of monounsaturated fats and vitamin E.",
  "Soya Oil Packed": "Oil extracted from soybeans, widely used for frying, baking, and in processed foods. Soya oil is low in saturated fat and high in polyunsaturated fats, including omega-3 fatty acids, making it a heart-healthy cooking oil.",
  "Wheat": "A cereal grain used to make flour for bread, pasta, pastry, and more. Wheat is a staple food in many cultures, providing essential nutrients like carbohydrates, fiber, and protein. It is available in various forms, such as whole wheat, refined flour, and semolina.",
  "Sunflower Oil Packed": "Oil extracted from sunflower seeds, used for cooking, frying, and in salad dressings. Sunflower oil is rich in vitamin E and low in saturated fat, making it a popular choice for heart-healthy diets.",
  "Palm Oil Packed": "Edible vegetable oil derived from the fruit of oil palm trees. Palm oil is widely used in cooking and in the food industry for products like margarine, baked goods, and snacks. It is known for its high saturated fat content and long shelf life.",
  "Gur": "Also known as jaggery, gur is an unrefined sugar made from sugarcane or palm sap. It has a rich, caramel-like flavor and is used in Indian sweets, desserts, and traditional medicines. Gur is considered healthier than refined sugar due to its higher mineral content.",
  "Tea Loose": "Dried tea leaves sold in loose form, used to brew tea. Loose tea offers a richer and more complex flavor compared to tea bags. It comes in various types, including black, green, oolong, and herbal teas, each with unique health benefits and flavors.",
  "Salt Pack Iodised": "Table salt with added iodine, essential for thyroid health and preventing iodine deficiency. Iodised salt is commonly used in cooking and as a seasoning, providing essential electrolytes and enhancing the flavor of foods.",
  "Potato": "A starchy root vegetable, commonly used in cooking and baking. Potatoes are versatile and can be prepared in many ways, such as boiling, frying, roasting, and mashing. They are a good source of carbohydrates, vitamin C, and potassium.",
  "Onion": "A bulb vegetable used as a base in many dishes for its flavor. Onions come in various types, including yellow, red, and white onions, each with its own flavor profile. They are rich in vitamins, minerals, and antioxidants.",
  "Tomato": "A juicy fruit used in cooking, salads, and sauces. Tomatoes are a key ingredient in many cuisines, known for their rich flavor and high content of vitamins A and C, potassium, and antioxidants like lycopene.",
  "Atta Wheat": "Whole wheat flour used to make Indian flatbreads like chapati, roti, and paratha. Atta is rich in fiber, protein, and essential nutrients, making it a healthier alternative to refined flours.",
  "Mustard Oil Packed": "Oil extracted from mustard seeds, commonly used in Indian cooking for its pungent flavor and high smoking point. Mustard oil is rich in monounsaturated fats and has antimicrobial properties. It is used in frying, sautéing, and as a preservative for pickles."
}
    return coms.get(commodity_name)



def getImages(commodity_name:str):
    coms = {
  "Rice": "https://images.pexels.com/photos/674574/pexels-photo-674574.jpeg",
  "Vanaspati Packed": "https://images.pexels.com/photos/410979/pexels-photo-410979.jpeg",
  "Gram Dal": "https://images.pexels.com/photos/1027307/pexels-photo-1027307.jpeg",
  "Tur Arhar Dal": "https://images.pexels.com/photos/958545/pexels-photo-958545.jpeg",
  "Urad Dal": "https://images.pexels.com/photos/207966/pexels-photo-207966.jpeg",
  "Moong Dal": "https://images.pexels.com/photos/105508/pexels-photo-105508.jpeg",
  "Masoor Dal": "https://images.pexels.com/photos/132259/pexels-photo-132259.jpeg",
  "Sugar": "https://images.pexels.com/photos/411025/pexels-photo-411025.jpeg",
  "Milk": "https://images.pexels.com/photos/236012/pexels-photo-236012.jpeg",
  "Groundnut Oil Packed": "https://images.pexels.com/photos/1028729/pexels-photo-1028729.jpeg",
  "Soya Oil Packed": "https://images.pexels.com/photos/1583837/pexels-photo-1583837.jpeg",
  "Wheat": "https://images.pexels.com/photos/145948/pexels-photo-145948.jpeg",
  "Sunflower Oil Packed": "https://images.pexels.com/photos/111131/pexels-photo-111131.jpeg",
  "Palm Oil Packed": "https://images.pexels.com/photos/208052/pexels-photo-208052.jpeg",
  "Gur": "https://images.pexels.com/photos/6665/food-cookies-peanut-butter-cake.jpg",
  "Tea Loose": "https://images.pexels.com/photos/405146/pexels-photo-405146.jpeg",
  "Salt Pack Iodised": "https://images.pexels.com/photos/1071097/pexels-photo-1071097.jpeg",
  "Potato": "https://images.pexels.com/photos/1398/vegetables-wood-food-potatoes.jpg",
  "Onion": "https://images.pexels.com/photos/1437267/pexels-photo-1437267.jpeg",
  "Tomato": "https://images.pexels.com/photos/1032558/pexels-photo-1032558.jpeg",
  "Atta Wheat": "https://images.pexels.com/photos/461428/pexels-photo-461428.jpeg",
  "Mustard Oil Packed": "https://images.pexels.com/photos/533353/pexels-photo-533353.jpeg"
}

    return coms.get(commodity_name)