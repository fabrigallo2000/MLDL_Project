import pandas as pd
import matplotlib.pyplot as plt

# Apertura del primo file CSV
df1 = pd.read_csv('accuracy_5epoch_20clientPerRound_25r_NIID.csv')

# Creazione del grafico
plt.plot(df1['x_round'], df1['y'], label='1 epoch')

df2 = pd.read_csv('accuracy_5epoch_5clientPerRound_25r_NIID.csv')
plt.plot(df2['x_round'], df2['y'], label='5 epochs')

'''df3 = pd.read_csv('accuracy_10epoch_5clientPerRound_25r_NIID.csv')
plt.plot(df3['x_round'], df3['y'], label='10 epochs')
'''
# Personalizzazione del grafico
plt.xlabel('Round')
plt.ylabel('Acauracy')
plt.grid(True)
plt.title('Accuracy: 25 round, 5 clients per round, NIID')
plt.legend()


# Mostra il grafico
plt.show()
