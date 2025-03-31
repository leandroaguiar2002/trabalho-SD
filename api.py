import yfinance as yf
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch import optim
from pydantic import BaseModel
from typing import List, Dict, Optional

class StockForecastResponse(BaseModel):
    stock_ticker: str
    forecast_prices: Dict[str, Dict[str, float]]

class StockForecastRequest(BaseModel):
    days: int = 10

app = FastAPI()

class NeuralNetwork(nn.Module):
    def __init__(self, num_feature):
        super(NeuralNetwork, self).__init__()
        self.lstm = nn.LSTM(num_feature, 64, batch_first=True) 
        self.dropout = nn.Dropout(p=0.5)  
        self.fc = nn.Linear(64, num_feature)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        x = self.dropout(hidden[-1]) 
        x = self.fc(x)
        return x

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

def split_data(sequence):
    train_data, test_data = train_test_split(sequence, test_size=0.2, shuffle=False)
    val_data, test_data = train_test_split(test_data, test_size=0.5, shuffle=False)
    return train_data, val_data, test_data

def split_Xy(dataset):
    X = dataset[:, :-1, :]
    Y = dataset[:, -1, -1]  
    return X, Y

def train(model, dataloader, optimizer, criterion):
    model.train()  
    epoch_loss = 0
    
    for inputs, target in dataloader:
        optimizer.zero_grad()  
        prediction = model(inputs)  
        target = target.unsqueeze(-1)  
        loss = criterion(prediction, target)          
        loss.backward()  
        optimizer.step() 

        epoch_loss += loss.item()

    return epoch_loss

def evaluate(model, dataloader, criterion):
    model.eval()  
    epoch_loss = 0
    
    with torch.no_grad():
        for inputs, target in dataloader:
            prediction = model(inputs) 
            target = target.unsqueeze(-1)  
            loss = criterion(prediction, target)              
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


@app.get("/")
async def root():
  return {'message': 'Stock Price Forecasting API. Use /forecast/{stock} to get predictions.'}


@app.post('/forecast/{stock}', response_model=StockForecastResponse)
async def forecast_stock(stock: str, requestParam: StockForecastRequest):
  try:
    tck = yf.Ticker(stock)
    start_date = "2010-03-20"
    df = tck.history(start=start_date, end=None)
    
    if df.empty:
        return {'error' : 'Stock not found'}
    
    df = df.reset_index()
    df['Volume'] = df['Volume'].astype('float64')
    df['SMA_10days'] = df['Close'].rolling(window=10).mean().fillna(df['Close'])
    
    ten_days = df[['Open','High','Low', 'Volume','SMA_10days','Close']].copy(deep=True)
    
    ten_days = ten_days.dropna()
    
    scaler = MinMaxScaler(feature_range=(0,1)).fit(ten_days.Low.values.reshape(-1,1))
    
    ten_days['Open'] = scaler.transform(ten_days.Open.values.reshape(-1,1))
    ten_days['High'] = scaler.transform(ten_days.High.values.reshape(-1,1))
    ten_days['Low'] = scaler.transform(ten_days.Low.values.reshape(-1,1))
    ten_days['Volume'] = scaler.transform(ten_days.Volume.values.reshape(-1,1))
    ten_days['Close'] = scaler.transform(ten_days.Close.values.reshape(-1,1))
    ten_days['SMA_10days'] = scaler.transform(ten_days['SMA_10days'].values.reshape(-1,1))
    
    data_10days = ten_days[['Open','High','Low', 'SMA_10days','Close']].values
    sequence_10days = []
    
    seq_len = requestParam.days + 1
    sequence_10days = create_sequences(data_10days, seq_len)
    #   for index in range(len(data_10days) - seq_len + 1): 
    #     sequence_10days.append(data_10days[index: index + seq_len])
    #   sequence_10days = np.array(sequence_10days)
    
    train_data_10days, val_data_10days, test_data_10days = split_data(sequence_10days)
    
    x_train_10d, y_train_10d = split_Xy(train_data_10days)
    x_val_10d, y_val_10d = split_Xy(val_data_10days)
    x_test_10d, y_test_10d = split_Xy(test_data_10days)
    
    batch_size = 64
    
    x_train_10d = torch.tensor(x_train_10d).float()
    y_train_10d = torch.tensor(y_train_10d).float()
    x_val_10d = torch.tensor(x_val_10d).float()
    y_val_10d = torch.tensor(y_val_10d).float()
    train_set_10d = TensorDataset(x_train_10d,y_train_10d)
    train_dataloader_10d = DataLoader(train_set_10d, batch_size=32, shuffle=False)
    val_set_10d = TensorDataset(x_val_10d ,y_val_10d)
    val_dataloader_10d = DataLoader(val_set_10d, batch_size=32, shuffle=False)
    
    mse = nn.MSELoss()
    epochs = 50
    num_feature = 4  
    model = NeuralNetwork(num_feature)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    model_10d = NeuralNetwork(5)
    optimizer = optim.Adam(model_10d.parameters())
    
    train_losses = []
    valid_losses = []
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model_10d, train_dataloader_10d, optimizer, mse)
        train_losses.append(train_loss)

        valid_loss = evaluate(model_10d, val_dataloader_10d, mse)
        valid_losses.append(valid_loss)

        print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
    
    x_test_10d = torch.tensor(x_test_10d).float()
    y_test_10d = torch.tensor(y_test_10d).float()

    with torch.no_grad():
        y_pred_10d = model_10d(x_test_10d)

    y_pred_10d = y_pred_10d.numpy()
    y_test_10d = y_test_10d.numpy()

    y_pred_10d = y_pred_10d.reshape(-1, y_pred_10d.shape[-1])[:, -1]
    
    last_sequence = sequence_10days[-1:, :, :]
    last_sequence = torch.from_numpy(last_sequence).float()

    days_num = requestParam.days
    with torch.no_grad():
        for i in range(days_num):
            pred = model_10d(last_sequence)  
            
            pred = pred.unsqueeze(1) 
            
            last_sequence = torch.cat((last_sequence, pred), dim=1) 
            last_sequence = last_sequence[:, 1:, :]  

    predicting_days = last_sequence.squeeze().numpy()

    predicting_days = scaler.inverse_transform(predicting_days)

    predicting_days = predicting_days[:days_num, :4]

    df_pred = pd.DataFrame(
        data=predicting_days,
        columns=['Open', 'High', 'Low', 'Close']
    )

    last_date_in_df = df['Date'].iloc[-1]

    next_dates = pd.date_range(start=last_date_in_df + pd.Timedelta(days=1), periods=days_num)

    df_dates = pd.DataFrame({'Date': next_dates})

    df_combined = pd.concat([df_dates, df_pred], axis=1)
    
    forecast_dict = {}
    forecast_dates=df_combined['Date'].dt.strftime('%Y-%m-%d').tolist()
    forecast_prices=df_combined[['Open', 'High', 'Low', 'Close']].to_dict(orient='records')
    
    for i, date in enumerate(forecast_dates):
        forecast_dict[date] = forecast_prices[i]    
    
    response =  StockForecastResponse(
        stock_ticker=stock,
        forecast_prices = forecast_dict,
    )
    
    return response

  except Exception as e:
    return {'error': str(e)}
      
