import torch
import torch.nn as nn

class DeepOilLSTM(nn.Module):
    def __init__(self):
        super(DeepOilLSTM, self).__init__()
        
        # INPUT LAYER: 
        # 5 Price Features from OANDA (Open, High, Low, Close, Volume)
        # + 3 Sentiment Features from Gemini (Bullish, Bearish, Shock Risk)
        # Total = 8 Inputs
        self.input_size = 8
        self.hidden_size = 32 # The "memory" capacity of the brain
        self.num_layers = 2   # 2 stacked LSTM layers for deep pattern recognition
        
        # The LSTM acts as the "Short Term Memory" to remember past price action
        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=True,
            dropout=0.2
        )
        
        # The Fully Connected layer makes the final decision
        self.fc = nn.Linear(self.hidden_size, 1)
        
        # Sigmoid squashes the output to a Probability between 0.0 (Sell) and 1.0 (Buy)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass data through LSTM
        out, _ = self.lstm(x)
        
        # Grab the output of the very last time step
        out = out[:, -1, :]
        
        # Pass through the decision layer
        out = self.fc(out)
        
        # Return probability
        return self.sigmoid(out)

if __name__ == "__main__":
    print("Initializing Deep Learning Brain...")
    model = DeepOilLSTM()
    print("Neural Network Architecture successfully built!")
    print(model)