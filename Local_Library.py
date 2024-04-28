import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA

def kalman_filter(data):
    # Source Code is stored locally as the support for python inbuilt function is no longer supported after one of the recent python updates.
    # Welch, G. and Bishop, G., 1995. An introduction to the Kalman filter., pages 11-15
    # https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf    
    # data: input numpy.ndarray data
    
    data_shape = data.shape # Shape of the data.
    process_variance = 2e-4 # Estimated process variance, Q.    
    # Instantiating default vectors for filter parameters
    x_hat = np.zeros(data_shape)      # a posteriori state estimate of x
    P = np.zeros(data_shape)         # a posteriori estimate error covariance
    x_hat_bar = np.zeros(data_shape) # a priori state estimate of x
    P_bar = np.zeros(data_shape)    # a priori estimate error covariance
    K = p.zeros(data_shape)         # gain or blending factor that minimizes the a posteriori error covariance
    R = 0.1**2 # estimate of measurement error covariance    
    x_hat[0] = 0.0 # Initial state value for posteriori state estimate of x
    P[0] = 1.0 # Initial state value for posteriori estimate error covariance
    for data_shape_iter in range(1,data_shape[0]):
        # time update filter parameter equations
        x_hat_bar[data_shape_iter] = x_hat[data_shape_iter-1]
        P_bar[data_shape_iter] = P[data_shape_iter-1]+process_variance
        # measurement update filter parameter equations
        K[data_shape_iter] = P_bar[data_shape_iter]/( P_bar[data_shape_iter]+R )
        x_hat[data_shape_iter] = x_hat_bar[data_shape_iter]+K[data_shape_iter]*(data[data_shape_iter]-x_hat_bar[data_shape_iter])
        P[data_shape_iter] = (1-K[data_shape_iter])*P_bar[data_shape_iter]        
    return x_hat
   
   
def portfolio_analyser(weights, returns, risk_free_rate, cov_matrix):
    # Analyses a portfolio for a given set of weights and returns annual return, annual risk and sharpe ratio.
    # weights: numpy array of weights for each stock option in the portfolio .
    # returns: portfolio returns data.
    # risk_free_rate: benchmark for retruns.
    # cov_matrix: covariance martix for the ETFs in the portfolio.
     
    mean_returns = returns.mean() # Calculating mean returns.
    weighted_annual_return = np.sum(mean_returns * weights)*252 # Calculating weighted annual retrun for the portfolio with ETFs weighted according to weights vector.
    weighted_annual_risk = np.sqrt(np.dot(np.transpose(weights), np.dot(cov_matrix, weights)))*np.sqrt(252) # annualised standard deviation for the portfolio with ETFs weighted according to weights vector.
    sharpe_ratio = (weighted_annual_return - risk_free_rate)/weighted_annual_risk # Calculation sharpe ratio for the portfolio.
    return [weighted_annual_return, weighted_annual_risk, sharpe_ratio]


def sharpe_ratio(return_series, N_pts, risk_free_rate):
    # Calculates sharpe ratio.
    # return_series: ETF returns in series format.
    # N_pts: number of sample to aggeregate over.
    # risk_free_rate: benchmark for retruns.
    
    mean = return_series.mean() * N_pts - risk_free_rate # Calculating mean adjusted by risk_free_rate over N_pts samples.
    sigma = return_series.std() * np.sqrt(N_pts) # Calurlating standard deviation over N_pts samples.
    return mean / sigma

def arch_volatility_predictor(data):
    # Predicts volatility of ETFs using ARCH model.
    # data: input returns data (% change).
    hyper_models = {'AR': {'best_params': {'p': 2, 'q': 1}}} # Hyper-parameters to optimise predictions.
    arch_tuned = arch_model(data, **hyper_models['AR']['best_params'], vol='ARCH').fit(update_freq=3, disp='off') # Fitting arch model on ETF returns data.
    arch_forecast_var = arch_tuned.forecast(horizon=1).variance.iloc[0,0] # Prediction volatility for the next instance.
    return arch_forecast_var

def garch_volatility_predictor(data):
    # Predicts volatility of ETFs using GARCH model.
    # data: input returns data (% change).
    hyper_models = {'AR': {'best_params': {'p': 2, 'q': 1}}} # Hyper-parameters to optimise predictions.
    arch_tuned = arch_model(data, **hyper_models['AR']['best_params'], vol='GARCH').fit(update_freq=3, disp='off') # Fitting garch model on ETF returns data.
    arch_forecast_var = arch_tuned.forecast(horizon=1).variance.iloc[0,0] # Prediction volatility for the next instance.     
    return arch_forecast_var 

def LSTM_predictor(lstm_model, X_train, y_train, X_test):
    # Predicts volatility of ETFs using LSTM model.
    # lstm_model: instantiated LSTM model.
    # X_train: training input % returns data.
    # y_train: test input % returns data.
    # X_Test: test % returns data based on which predictions are made.
    
    # Preparing training and test data for LSTM model
    X_train = np.array([X_train]) # Converting training X data to an array.
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # Reshaping training X data.
    X_test = np.array([X_test]) # Converting test X data to an array.
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # Reshaping test X data.
    y_train = np.array([y_train]) # Converting training y data to an array.  
    # Training the LSTM model
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0) # Using train X and y data to fit LSTM model.
    prediction = lstm_model.predict(X_test, verbose=0) # Making LSTM prediction.    
    return prediction

def svr_predictor(svr_model, X_train, y_train, X_test):
    # Predicts volatility of ETFs using SVR model.
    # svr_model: instantiated SVR model.
    # X_train: training input % returns data.
    # y_train: test input % returns data.
    # X_Test: test % returns data based on which predictions are made.
    
    # Converting the data to a format that is suitable for SVR model.
    X_train = np.array([X_train]).reshape(-1, 1) # Converting training X data to an array and reshaping it.
    y_train = np.array(y_train) # Converting training y data to an array. 
    X_test = np.array([X_test]).reshape(-1, 1) # Converting test X data to an array and reshaping it.       
    svr_model.fit(X_train, y_train)  # Training SVR model using the trainign data.    
    prediction = svr_model.predict(X_test) # Making predictions using the test data.   
    return prediction

def arima_predictor(data):
    # Predicts volatility of ETFs using ARIMA model.
    # data: input returns data (% change).
    arima_model = ARIMA(data, order=(5,1,0)).fit() # Fitting arima model on ETF returns data.
    arima_forecast_var = arima_model.forecast(steps=1) # Prediction volatility for the next instance.
    return arima_forecast_var  
   
def data_prep(data, feature, window, dropna=True, scale=True):
    # Prepares input data for subsequent machine learning operations (ARCH and GARCH).
    # data = Actual return data (pd.Series format).
    # feature: primary input feature (Actual_Return).
    # window: period window to estimate variance.
    # dropna: If True, drops NA.
    # scale: If True, scales data.
    
    if scale: # Performing scaling operaion on the input data.
        data_reshaped = data.values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)    
        data = pd.DataFrame(data_scaled, columns=[feature]) # Scaled data frame used to prepare data.
    else: # Scaling is ignored.
        data = data.to_frame(feature) # Unscaled data frame used to prepare data.
        
    
    # Creating a sequence of training X data subsets over the configured window.
    result_train_X = [data[feature].iloc[i:i + window].tolist() for i in range(len(data) - window + 1)]    
    for i in range(len(data) - window + 1, len(data)):
        result_train_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))        
    data['X_train'] = result_train_X    
    
    # Creating a sequence of training y data subsets over the configured window.
    result_train_y = [data[feature][i+window] for i in range(len(data) - window)]  
    for i in range(len(data) - window + 1, len(data) + 1):
        result_train_y.append(np.nan)
    data['y_train'] = result_train_y 

    # Creating a sequence of test X data subsets over the configured window.
    result_test_X = [data[feature].iloc[i+1:i + window+1].tolist() for i in range(len(data) - window)]
    for i in range(len(data) - window + 1, len(data)+1):
        result_test_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))
    data['X_test'] = result_test_X 

    # Creating a sequence of test y data subsets over the configured window.
    result_test_y = [data[feature][i+window+1] for i in range(len(data) - window - 1)]    
    for i in range(len(data) - window + 1, len(data)+2):
        result_test_y.append(np.nan)
    data['y_test'] = result_test_y
    
    if dropna: # If configure, dropping missing values.
        data.dropna(inplace=True)
        
    data['X_train_var'] = data['X_train'].apply(lambda x: np.var(x)) # Calculating training X subset variances.
    
    return data   
    
def data_prep_arima(data, input_feature, feature, window, dropna=True, scale=True):
    # Prepares input data for subsequent machine learning operations (LSTM and SVR).
    # data = Actual return data (pd.Series format).
    # input_feature: primary input feature (Actual_Return).
    # feature: name of the estimated feature with variance (Actual_Variance).
    # window: period window to estimate variance.
    # dropna: If True, drops NA.
    # scale: If True, scales data.
    
    # Calculating input data variance using a rolling window with size = window (input).
    data_df_raw = data.to_frame(input_feature)
    data_df_raw[feature] = data_df_raw[input_feature].rolling(window=window).var()
    data_df_raw.dropna(inplace=True)
    data = data_df_raw[feature]
    
    if scale: # Performing scaling operaion on the input data.
        data_reshaped = data.values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)    
        data = pd.DataFrame(data_scaled, columns=[feature]) # Scaled data frame used to prepare data.
    else:
        data = data.to_frame(feature) # Unscaled data frame used to prepare data.
        
    # Creating a sequence of training X data subsets over the configured window.
    result_train_X = [data[feature].iloc[i:i + window].tolist() for i in range(len(data) - window + 1)]    
    for i in range(len(data) - window + 1, len(data)):
        result_train_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))        
    data['X_train'] = result_train_X    
    
    # Creating a sequence of training y data subsets over the configured window.
    #result_train_y = [data[feature].iloc[i+1:i+window+1].tolist() for i in range(len(data) - window)]  
    result_train_y = [data[feature].iloc[i+window:i+window+1].tolist() for i in range(len(data) - window)]  
    for i in range(len(data) - window + 1, len(data) + 1):
        result_train_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i)) 
    data['y_train'] = result_train_y 
    
    # Creating a sequence of test X data subsets over the configured window.
    result_test_X = [data[feature].iloc[i+1:i + window+1].tolist() for i in range(len(data) - window)]
    for i in range(len(data) - window + 1, len(data)+1):
        result_test_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))
    data['X_test'] = result_test_X 

    # Creating a sequence of test y data subsets over the configured window.
    #result_test_y = [data[feature].iloc[i+2:i+window+2].tolist() for i in range(len(data) - window - 1)] 
    result_test_y = [data[feature].iloc[i+window+1:i+window+2].tolist()[0] for i in range(len(data) - window - 1)] 
    for i in range(len(data) - window + 1, len(data)+2):
        if i <= len(data):
            result_test_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i)) 
        else:
            result_test_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i - 1)) 
    data['y_test'] = result_test_y
    
    # Creating a temporary feature which is going to be used as a placeholder to remove NAs inserted during the trainig and test set generation process.
    result_test_y_temp = [data[feature][i+window+1] for i in range(len(data) - window - 1)]    
    for i in range(len(data) - window + 1, len(data)+2):
        result_test_y_temp.append(np.nan)
    data['y_test_temp'] = result_test_y_temp
    
    if dropna: # If configure, dropping missing values.
        data.dropna(inplace=True)
        
    #data['X_train_var'] = data['X_train'].apply(lambda x: np.var(x)) # Calculating training X subset variances.
    data['X_train_var'] = data['X_train'].apply(lambda x: np.mean(x)) # Calculating training X subset variances.
    
    data = data.drop("y_test_temp", axis=1) # Dropping the redundant placeholder feature.
    
    return data
    
def data_prep_svr(data, input_feature, feature, window, dropna=True, scale=True):
    # Prepares input data for subsequent machine learning operations (LSTM and SVR).
    # data = Actual return data (pd.Series format).
    # input_feature: primary input feature (Actual_Return).
    # feature: name of the estimated feature with variance (Actual_Variance).
    # window: period window to estimate variance.
    # dropna: If True, drops NA.
    # scale: If True, scales data.
    
    # Calculating input data variance using a rolling window with size = window (input).
    data_df_raw = data.to_frame(input_feature)
    data_df_raw[feature] = data_df_raw[input_feature].rolling(window=window).var()
    data_df_raw.dropna(inplace=True)
    data = data_df_raw[feature]
    
    if scale: # Performing scaling operaion on the input data.
        data_reshaped = data.values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)    
        data = pd.DataFrame(data_scaled, columns=[feature]) # Scaled data frame used to prepare data.
    else:
        data = data.to_frame(feature) # Unscaled data frame used to prepare data.
        
    # Creating a sequence of training X data subsets over the configured window.
    result_train_X = [data[feature].iloc[i:i + window].tolist() for i in range(len(data) - window + 1)]    
    for i in range(len(data) - window + 1, len(data)):
        result_train_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))        
    data['X_train'] = result_train_X    
    
    # Creating a sequence of training y data subsets over the configured window.
    result_train_y = [data[feature].iloc[i+1:i+window+1].tolist() for i in range(len(data) - window)]  
    #result_train_y = [data[feature].iloc[i+window:i+window+1].tolist() for i in range(len(data) - window)]  
    for i in range(len(data) - window + 1, len(data) + 1):
        result_train_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i)) 
    data['y_train'] = result_train_y 
    
    # Creating a sequence of test X data subsets over the configured window.
    result_test_X = [data[feature].iloc[i+1:i + window+1].tolist() for i in range(len(data) - window)]
    for i in range(len(data) - window + 1, len(data)+1):
        result_test_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))
    data['X_test'] = result_test_X 

    # Creating a sequence of test y data subsets over the configured window.
    result_test_y = [data[feature].iloc[i+2:i+window+2].tolist() for i in range(len(data) - window - 1)] 
    #result_test_y = [data[feature].iloc[i+window+1:i+window+2].tolist()[0] for i in range(len(data) - window - 1)] 
    for i in range(len(data) - window + 1, len(data)+2):
        if i <= len(data):
            result_test_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i)) 
        else:
            result_test_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i - 1)) 
    data['y_test'] = result_test_y
    
    # Creating a temporary feature which is going to be used as a placeholder to remove NAs inserted during the trainig and test set generation process.
    result_test_y_temp = [data[feature][i+window+1] for i in range(len(data) - window - 1)]    
    for i in range(len(data) - window + 1, len(data)+2):
        result_test_y_temp.append(np.nan)
    data['y_test_temp'] = result_test_y_temp
    
    if dropna: # If configure, dropping missing values.
        data.dropna(inplace=True)
        
    #data['X_train_var'] = data['X_train'].apply(lambda x: np.var(x)) # Calculating training X subset variances.
    data['X_train_var'] = data['X_train'].apply(lambda x: np.mean(x)) # Calculating training X subset variances.
    
    data = data.drop("y_test_temp", axis=1) # Dropping the redundant placeholder feature.
    
    return data
    
   
