import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


excel_file1 = "./data_322.xlsx"

df = pd.read_excel(excel_file1, sheet_name="Лист2", header=None)

n1= 27
n2 = 29
start1 = 132
start2 = 96
freq1 = np.array(df[9][start1:start1 + n1]) / 31.1# / 20.97#/ 19.37 #3 /19.2
freq2 = np.array(df[9][start2:start2 + n2]) / 20.97 #3 /19.2

#U = np.array(df[11][1:n]) #4
angle1 = np.array(df[10][start1:start1 + n1]) / 1.271597026# / 1.418783779# / 1.385996759
angle2 = np.array(df[10][start2:start2 + n2]) / 1.418783779


#print(freq2, angle2)
# y = np.polyfit()

from scipy import optimize

def approximation(x, alpha, beta, gamma):
    return alpha * np.arctan((beta * x).astype(float)) + gamma
print(np.arctan(1))
params = np.zeros((100, 3))
for i in range(100):
    params[i], x = optimize.curve_fit(approximation, freq1.astype(float), angle1.astype(float), p0=[(np.random.rand() - 0.5) / 0.5 * 1000, (np.random.rand() - 0.5) / 0.5 * 10, (np.random.rand() - 0.5) / 0.5 * 100])
#print(alpha, beta, gamma)


fig, ax = plt.subplots()
fig.suptitle("ФЧХ", fontsize=14, fontweight='bold')

ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

plt.xlabel('freq, kHz', loc='right')
plt.ylabel('phi', loc='top')
ax.plot(freq1, angle1, lw = 1, color='red', marker='o', label="25нФ", markersize=4, ls='')
ax.plot(freq2, angle2, lw = 1, color='green', marker='o', label="57.2нФ", markersize=4, ls='')

#ax.plot(freq1, approximation(freq1, alpha, beta, gamma), color="red", label="Аппроксимация", ls='-')
plt.grid(linewidth = "0.2")
plt.savefig("ФЧХ.svg")
plt.show()
plt.clf()
plt.plot(np.linspace(0.9, 1.1, 100), approximation(np.linspace(0.9, 1.1, 100), alpha, beta, gamma))

plt.show()
