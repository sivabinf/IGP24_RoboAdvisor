import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def kalman_filter(observations, initial_state_mean=0, initial_state_covariance=1, 
                  process_noise_covariance=0.1, measurement_noise_covariance=10):
    # Define parameters
    dt = 1  # time step
    F = np.array([[1]])  # state transition matrix
    H = np.array([[1]])  # observation matrix
    Q = np.array([[process_noise_covariance]])  # process noise covariance
    R = np.array([[measurement_noise_covariance]])  # measurement noise covariance

    # Initialize state estimates
    x = np.array([[initial_state_mean]])  # initial state estimate
    P = np.array([[initial_state_covariance]])  # initial state covariance

    # Kalman filter loop
    filtered_state_means = []
    for z in observations:
        # Predict
        x_pred = F @ x # Matrix multiplication
        P_pred = F @ P @ F.T + Q

        # Update
        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(len(x)) - K @ H) @ P_pred

        # Save filtered state estimate
        filtered_state_means.append(x[0, 0])

    return filtered_state_means
   
   
def portfolio_analyser(weights, returns, risk_free_rate, cov_matrix):
    # weights: numpy array of weights for each stock option in the portfolio 
     
    mean_returns = returns.mean()
    weighted_annual_return = np.sum(mean_returns * weights)*252   
    weighted_annual_risk = np.sqrt(np.dot(np.transpose(weights), np.dot(cov_matrix, weights)))*np.sqrt(252) # annualised standard deviation 
    sharpe_ratio = (weighted_annual_return - risk_free_rate)/weighted_annual_risk 
    return [weighted_annual_return, weighted_annual_risk, sharpe_ratio]


def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma
    
    
# def data_preprocessor(data, Window):
    # X = []
    # y = []
    # for i in range(Window, len(data)):
        # X.append(data[i-Window:i])
        # y.append(data[i])
    # return np.array(X), np.array(y)
    

# def data_preprocess(data, feature, window):
    # # Sorts a data farme into chunks of training and test datasets
    # # data: input Dataframe
    # # feature to extrace data from
    
    # new_data = {'X_train':[], 'y_train':[], 'X_test':[], 'y_test':[]}    
    
    # for i in range(len(data['Nums'])-(window+2)):
        # new_data['X_train'].append(list(data[feature][i:i+window+1].values))
        # new_data['y_train'].append(data[feature][i+window+1])
        # new_data['X_test'].append(list(data[feature][i+1:i+window+2].values))
        # new_data['y_test'].append(data[feature][i+window+2])
    
    # return pd.DataFrame(new_data)
    

def data_prep(data, feature, window, dropna=True, scale=True):
    # data = Actual return data (pd.Series format)
    if scale:
        data_reshaped = data.values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)    
        data = pd.DataFrame(data_scaled, columns=[feature])
    else:
        data = data.to_frame(feature)
        
    
    result_train_X = [data[feature].iloc[i:i + window].tolist() for i in range(len(data) - window + 1)]    
    for i in range(len(data) - window + 1, len(data)):
        result_train_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))        
    data['X_train'] = result_train_X    
    
#     result_train_y = [data[feature][i+window].tolist() for i in range(len(data) - window)]  
    result_train_y = [data[feature][i+window] for i in range(len(data) - window)]  
    for i in range(len(data) - window + 1, len(data) + 1):
        result_train_y.append(np.nan)
    data['y_train'] = result_train_y 

    
    result_test_X = [data[feature].iloc[i+1:i + window+1].tolist() for i in range(len(data) - window)]
    for i in range(len(data) - window + 1, len(data)+1):
        result_test_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))
    data['X_test'] = result_test_X 

    result_test_y = [data[feature][i+window+1] for i in range(len(data) - window - 1)]    
    for i in range(len(data) - window + 1, len(data)+2):
        result_test_y.append(np.nan)
    data['y_test'] = result_test_y
    
    if dropna:
        data.dropna(inplace=True)
        #data['y_train'] = data['y_train'].astype(int)
        #data['y_test'] = data['y_test'].astype(int)
        
    data['X_train_var'] = data['X_train'].apply(lambda x: np.var(x))
    
    return data
    
  
    
# def data_prep_lstm(data, feature, window, dropna=True, scale=True):
    # # data = Actual return data
    # if scale:
        # data_reshaped = data.values.reshape(-1, 1)
        # scaler = StandardScaler()
        # data_scaled = scaler.fit_transform(data_reshaped)    
        # data = pd.DataFrame(data_scaled, columns=[feature])
    # else:
        # data = data.to_frame(feature)
        
    
    # result_train_X = [data[feature].iloc[i:i + window].tolist() for i in range(len(data) - window + 1)]    
    # for i in range(len(data) - window + 1, len(data)):
        # result_train_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))        
    # data['X_train'] = result_train_X    
    
    
    # result_train_y = [data[feature].iloc[i:i+window+1].tolist() for i in range(len(data) - window)]  
    # for i in range(len(data) - window + 1, len(data) + 1):
        # result_train_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i + 1)) 
    # data['y_train'] = result_train_y 
    
    
    # result_test_X = [data[feature].iloc[i+1:i + window+1].tolist() for i in range(len(data) - window)]
    # for i in range(len(data) - window + 1, len(data)+1):
        # result_test_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))
    # data['X_test'] = result_test_X 

    # result_test_y = [data[feature].iloc[i+1:i+window+2].tolist() for i in range(len(data) - window - 1)] 
# #     result_test_y = [data[feature][i+window+1] for i in range(len(data) - window - 1)]    
    # for i in range(len(data) - window + 1, len(data)+2):
        # result_test_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i + 1)) 
# #         result_test_y.append(np.nan)
    # data['y_test'] = result_test_y
    
    # result_test_y_temp = [data[feature][i+window+1] for i in range(len(data) - window - 1)]    
    # for i in range(len(data) - window + 1, len(data)+2):
        # result_test_y_temp.append(np.nan)
    # data['y_test_temp'] = result_test_y_temp
    
    # if dropna:
        # data.dropna(inplace=True)
        # #data['y_train'] = data['y_train'].astype(int)
        # #data['y_test'] = data['y_test'].astype(int)
        
# #     data['X_train_var'] = data['X_train'].apply(lambda x: np.var(x))
    
    # data = data.drop("y_test_temp", axis=1)
    
    # return data
    
    
def data_prep_lstm(data, feature, window, dropna=True, scale=True):
    # data = Actual return data
    if scale:
        data_reshaped = data.values.reshape(-1, 1)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)    
        data = pd.DataFrame(data_scaled, columns=[feature])
    else:
        data = data.to_frame(feature)
        
    
    result_train_X = [data[feature].iloc[i:i + window].tolist() for i in range(len(data) - window + 1)]    
    for i in range(len(data) - window + 1, len(data)):
        result_train_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))        
    data['X_train'] = result_train_X    
    
    
    result_train_y = [data[feature].iloc[i+1:i+window+1].tolist() for i in range(len(data) - window)]  
    for i in range(len(data) - window + 1, len(data) + 1):
        result_train_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i)) 
    data['y_train'] = result_train_y 
    
    
    result_test_X = [data[feature].iloc[i+1:i + window+1].tolist() for i in range(len(data) - window)]
    for i in range(len(data) - window + 1, len(data)+1):
        result_test_X.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i))
    data['X_test'] = result_test_X 

    result_test_y = [data[feature].iloc[i+2:i+window+2].tolist() for i in range(len(data) - window - 1)] 
#     result_test_y = [data[feature][i+window+1] for i in range(len(data) - window - 1)]    
    for i in range(len(data) - window + 1, len(data)+2):
        if i <= len(data):
            result_test_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i)) 
        else:
            result_test_y.append(data[feature].iloc[i:].tolist() + [np.nan] * (window - len(data) + i - 1)) 
#         result_test_y.append(np.nan)
    data['y_test'] = result_test_y
    
    result_test_y_temp = [data[feature][i+window+1] for i in range(len(data) - window - 1)]    
    for i in range(len(data) - window + 1, len(data)+2):
        result_test_y_temp.append(np.nan)
    data['y_test_temp'] = result_test_y_temp
    
    if dropna:
        data.dropna(inplace=True)
        #data['y_train'] = data['y_train'].astype(int)
        #data['y_test'] = data['y_test'].astype(int)
        
    data['X_train_var'] = data['X_train'].apply(lambda x: np.var(x))
    
    data = data.drop("y_test_temp", axis=1)
    
    return data
    
    
def mean_abs_err(sample, x, y):
    return (abs(sample[x] - sample[y]))