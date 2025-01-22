
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import  RandomForestRegressor
# from sklearn.ensemble import  BaggingRegressor
# from sklearn.ensemble import  AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import  GradientBoostingRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error


import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)
from sklearn.preprocessing import LabelEncoder as LE


class ModelSelect:
    def load_data(self):
        try:
            df = pd.read_csv('../data/melb_data_s.csv')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            return df
        except Exception as err:
            print(f"unexpected {err=}, {type(err)=}")

    def explore_data(self, df):

        try:
            print(df.head())
            print(df.sample(20))  # random check 20 observations
            y = df['Price']
            print(y.head())
            print(df.info())
            numeric_dtyp_features = df.select_dtypes(['int', 'float']).columns
            print(numeric_dtyp_features, len(numeric_dtyp_features))
            non_mum_dtyp_features = df.select_dtypes(['object']).columns
            print(non_mum_dtyp_features, len(non_mum_dtyp_features))
            print(df.isna().sum().sort_values(ascending=False))  # missing values in features of CouncilArea, BuildingArea, Car, YearBuilt
            print((df.isna().sum() * 100 / df.isna().count()).sort_values(ascending=False))  # % of missing value in each feature
            for feature in df.select_dtypes('object').columns: # use this to make decisions of removal of those with high unique values
                print("Unique value of ", feature, ' = ', len(df[feature].unique()))

        except Exception as err:
            print(f"unexpected {err=}, {type(err)=}")

    def preprocess_data(self, df):

        try:
            y_var = df['Price']

            data_df = df.drop(['Address', 'Suburb', 'SellerG','Price', 'Date'], axis = 1)

            # This helps to decide values for processing missing values in these columns
            print(data_df[['BuildingArea', 'YearBuilt', 'CouncilArea', 'Car']].describe(include='all'))

            # replace the missing value that has object type with most occurrent value
            data_df['CouncilArea'] = data_df['CouncilArea'].fillna('Moreland')
            data_df['Car'] = data_df['Car'].fillna(data_df['Car'].median())

            data_df['BuildingArea'] = data_df['BuildingArea'].fillna(data_df['BuildingArea'].mean())
            data_df['YearBuilt'] = data_df['YearBuilt'].fillna(data_df['YearBuilt'].mode()[0])
            print(data_df.isna().sum().sort_values(ascending=False)) #recheck if any na values in features

            # Get list of categorical features
            s = (data_df.dtypes == 'object')
            object_cols = list(s[s].index)
            print("Categorical variables in the dataset:", object_cols)

            # Label Encoding the object dtypes and values to numeric dtypes and values.
            LE = LabelEncoder()
            for i in object_cols:
                data_df[i] = data_df[[i]].apply(LE.fit_transform)

            # Scaling
            scaler = MinMaxScaler().fit(data_df)
            x_var = scaler.transform(data_df)
            print("  ***** ", data_df)
            return x_var, y_var, data_df

        except Exception as err:
            print(f"unexpected {err=}, {type(err)=}")

    def build_models(self, x, y, data_df, models_metrics):

        try:
            train_x, test_x, train_y, test_y = train_test_split(x, y, random_state=42)
            print ('Data size: ', data_df.shape[0] )
            print (' train_x size: ', train_x.shape)
            print(' test_x size: ', test_x.shape)
            print(' train_y size: ', train_y.shape)
            print(' tesy_y size: ', test_y.shape)

            lr = LinearRegression()
            models_metrics= self.models(lr, train_x, test_x, train_y, test_y, models_metrics, '')

            gbr = GradientBoostingRegressor(n_estimators=1500, random_state=42)
            models_metrics=self.models(gbr, train_x, test_x, train_y, test_y, models_metrics,'')

            xgbr = XGBRegressor()
            models_metrics = self.models(xgbr, train_x, test_x, train_y, test_y, models_metrics,'')

            xgbr_nest = XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.01)
            models_metrics = self.models(xgbr_nest, train_x, test_x, train_y, test_y, models_metrics,'learning_rate=0.5')

            # using cross validation to tune xgbregressor model
            params = {
                'n_estimators': [500, 1000, 1500, 2000],
                'learning_rate': [0.05, 0.075, 0.1],
                'max_depth': [7, 9],
                'reg_lambda': [0.3, 0.5]
            }

            xgbr_cv = GridSearchCV(estimator=xgbr, param_grid=params, cv=5, n_jobs=-1)
            models_metrics = self.models(xgbr_cv, train_x, test_x, train_y, test_y, models_metrics, 'cv')
            print("Best parameters set:", xgbr_cv.best_params_)
            xgbr_nest.save_model(r'../model/' + xgbr_nest.__class__.__name__ + '.json')

            print(models_metrics)

            return (models_metrics)

        except Exception as err:
            print(f"unexpected {err=}, {type(err)=}")


    def models(self, model,train_x, test_x, train_y, test_y, metrics, param):
        try:
            model.fit(train_x, train_y)
            prediction = model.predict(test_x)
            if param == 'cv':
                rsquare = model.best_score_
            else:
                rsquare = model.score(test_x, test_y)

            mae = mean_absolute_error(test_y, prediction)
            models_metrics = self.models_metrics_log(model.__class__.__name__ + ' ' + param, rsquare, mae, metrics)

            return models_metrics
        except Exception as err:
            print(f"unexpected {err=}, {type(err)=}")

    def models_metrics_log(self, model_name, rsquare, mae, models_metrics):
        try:
            # mae mean absolute error
            new_row = pd.DataFrame({'model name': [model_name],'r-square': [rsquare], 'mean absolute error': [mae]})
            models_metrics = pd.concat([models_metrics, new_row], ignore_index = True)

            return models_metrics
        except Exception as err:
            print(f"unexpected {err=}, {type(err)=}")


    def load_save_model(self):
        # Loading the xgboost model from the JSON file
        xgbr_hyper = XGBRegressor()
        xgbr_hyper.load_model(r'../model/XGBRegressor.json')
        # prediction = xgbr_hyper.predict('new_data') call this when new data available



if __name__ == "__main__":
    ms = ModelSelect()
    models_summary = pd.DataFrame([],
                                  columns=['model name',
                                           'r-square',
                                           'mean absolute error'
                                           ])
    prop_data = ms.load_data()
    ms.explore_data(prop_data)
    x, y, data = ms.preprocess_data(prop_data)
    ms.build_models(x, y, data, models_summary)
    ms.load_save_model()

