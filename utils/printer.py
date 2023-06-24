import pandas as pd
import matplotlib.pyplot as plt

# Apertura del primo file CSV
df1 = pd.read_csv('accuracy_10_clients_10_epochs_SPEC_50r_LS_0.csv')
plt.plot(df1['x_round'], df1['y'], label='ls=0')


df2 = pd.read_csv('accuracy_10_clients_10_epochs_SPEC_50r_LS_10.csv')
plt.plot(df2['x_round'], df2['y'], label='ls=10')


df3 = pd.read_csv('accuracy_10epoch_10clientPerRound_50r_base.csv')
plt.plot(df3['x_round'], df3['y'], label='standard')


'''df4 = pd.read_csv('accuracy_10_clients_10_epochs_SPEC_50r_LS_6.csv')
plt.plot(df4['x_round'], df4['y'], label='ls=6')

df5 = pd.read_csv('accuracy_10_clients_10_epochs_SPEC_50r_LS_8.csv')
plt.plot(df5['x_round'], df5['y'], label='ls=8')
'''
'''df6 = pd.read_csv('accuracy_10_clients_10_epochs_SPEC_50r_LS_20.csv')
plt.plot(df6['x_round'], df6['y'], label='ls=20')'''


# Personalizzazione del grafico
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('Accuracy: Spectral vs sandard (50 round, 10 clients per round, 10 epochs)')
plt.legend()


# Mostra il grafico
plt.show()
