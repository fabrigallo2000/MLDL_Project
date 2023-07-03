import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Apertura del primo file CSV
df1 = pd.read_csv('accuracy_stndConfig_BigD_TEST_SPEC_ls_0.csv')
#plt.plot(df1['x_round'], df1['y'], label='only spectral (ls=0)')


df2 = pd.read_csv('accuracy_stndConfig_BigD_TEST_SPEC_ls_5.csv')
plt.plot(df2['x_round'], df2['y'], label='FedAvg + Spectral analysis')


df3 = pd.read_csv('accuracy_stndConfig_BigD_TEST_base.csv')
plt.plot(df3['x_round'], df3['y'], label='FedAvg ')


'''df4 = pd.read_csv('accuracy_stndConfig_BigD_TEST_SPEC_ls_10.csv')
plt.plot(df4['x_round'], df4['y'], label='ls=10')
'''
'''df5 = pd.read_csv('accuracy_10_clients_10_epochs_SPEC_50r_LS_8.csv')
plt.plot(df5['x_round'], df5['y'], label='ls=8')

df6 = pd.read_csv('accuracy_10_clients_10_epochs_SPEC_50r_LS_20.csv')
plt.plot(df6['x_round'], df6['y'], label='ls=20')'''


# Personalizzazione del grafico
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.grid(True)
plt.title('FedAvg vs FedAvg + Spectral analysis')
plt.legend()


# Mostra il grafico
plt.show()

