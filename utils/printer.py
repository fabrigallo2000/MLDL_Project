import pandas as pd
import matplotlib.pyplot as plt

# Apertura del primo file CSV
df1 = pd.read_csv('loss_1epoch_5clientPerRound_25r_IID.csv')

# Creazione del grafico
plt.plot(df1['x_round'], df1['y'], label='1 epoch')

df2 = pd.read_csv('loss_5epoch_5clientPerRound_25r_IID.csv')
plt.plot(df2['x_round'], df2['y'], label='5 epochs')

df3 = pd.read_csv('loss_10epoch_5clientPerRound_25r_IID.csv')
plt.plot(df3['x_round'], df3['y'], label='10 epochs')

# Personalizzazione del grafico
plt.xlabel('Round')
plt.ylabel('Loss')
plt.grid(True)
plt.title('Loss: 25 round, 5 clients per round, IID')
plt.legend()


# Mostra il grafico
plt.show()
