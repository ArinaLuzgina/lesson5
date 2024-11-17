import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from matplotlib.ticker import FormatStrFormatter
from functions import plot_graph, mnk, nice_print

np.set_printoptions(suppress=True)


def approximation(x, Q, omega_0, delta, tau):
    return Q *(np.sin(delta) - (tau * omega_0) * (x - 1) * np.cos(delta)) / (x * (1 + (tau * omega_0 * (x - 1)) ** 2))
def approximation_PFC(x, f_0, delta, tau):
    return np.arctan(tau * f_0 * (x - 1))/np.pi + 1 / 2 - delta / np.pi

# constants
R = 3.45

excel_file1 = "./data_322.xlsx"


# start
df1 = pd.read_excel(excel_file1, sheet_name="Лист1", header=None)

basic_properties = np.zeros((22, 7))
rows = 1
lenghtes = 7
for i in range(4):
    basic_properties[i * 2] = df1[i + 1][rows : rows + lenghtes]

basic_properties[1] = 0
basic_properties[3] = 10 ** (-4)
basic_properties[5] = basic_properties[4] * 0.05 / 100
basic_properties[7] = basic_properties[6] * 0.05 / 100
basic_properties[8] = 1 / (((basic_properties[2] * 10 ** (3)) * 2 * np.pi) ** 2 * (basic_properties[0] * 10 ** (-9)))
basic_properties[9] = (1 / (2 * np.pi**2 * (basic_properties[0] * 10 ** (-9)) * (basic_properties[2] * 10 ** (3)) ** 3) * basic_properties[3])
basic_properties[10] = np.round(basic_properties[4] / basic_properties[6], decimals=4)
basic_properties[11] = np.round(np.sqrt((basic_properties[5] / basic_properties[6]) ** 2 + (basic_properties[4] * basic_properties[7] / basic_properties[6]) ** 2), decimals=4)
basic_properties[12] = np.round(np.sqrt(basic_properties[8] / (basic_properties[0] * 10 ** (-9))), decimals=2)
basic_properties[13] = np.sqrt(basic_properties[9] ** 2 / (4 * basic_properties[8] * (basic_properties[0] * 10 ** (-9))))
basic_properties[14] = np.round(np.sqrt(basic_properties[8] / (basic_properties[0] * 10 ** (-9))) / basic_properties[10], decimals=2)
basic_properties[15] = np.sqrt((basic_properties[11] ** 2 * basic_properties[8] / ((basic_properties[0] * 10 ** (-9)) * basic_properties[10] ** 4)) + (basic_properties[9] ** 2 / (4 * basic_properties[10] ** 2 * basic_properties[8] * (basic_properties[0] * 10 ** (-9)))))
basic_properties[16] = basic_properties[12] * 10**-3
basic_properties[17] = basic_properties[13] * 10**-3
basic_properties[18] = basic_properties[14] - R - basic_properties[16]
basic_properties[19] = np.sqrt(basic_properties[15] ** 2 + basic_properties[17] ** 2)
basic_properties[20] = basic_properties[6] / basic_properties[16]
basic_properties[21] = np.sqrt((basic_properties[7] / basic_properties[14]) ** 2 + (basic_properties[6] * basic_properties[15] / basic_properties[14] ** 2) ** 2)
names = ["C нФ","dC нФ","f_0kHz","df_0kHz","Uc V","dUc V","E B","dE B","L Гн","dL Гн","Q","dQ","p Ом","dp Ом","Rsu Ом","dRsu Ом","RSm Ом","dRSm Ом","R_L Ом","dR_L Ом","I А","dI А"]

#nice_print(data=basic_properties, names=names)

averages = np.zeros((3, 2))
averages[0][0] = np.mean(basic_properties[8])
averages[1][0] = np.sqrt(np.sum((basic_properties[8] - averages[0][0]) ** 2)) / (basic_properties[8].shape[0] * (basic_properties[8].shape[0] - 1))
averages[:, 0] *= 10**6
averages[:, 0] = np.round(averages[:, 0], decimals=0)
averages[0][1] = np.mean(basic_properties[18])
averages[1][1] = np.sqrt(np.sum((basic_properties[18] - averages[0][1]) ** 2)) / (basic_properties[18].shape[0] * (basic_properties[18].shape[0] - 1))
averages[:, 1] = np.round(averages[:, 1], decimals=2)
averages[2, 0] = averages[1, 0] / averages[0,0] * 100
averages[2, 1] = averages[1, 1] / averages[0,1] * 100

#nice_print(averages.T, names=["L, мкГн", "R_L, Ом"])

del df1

# second part
df2 = pd.read_excel(excel_file1, sheet_name="Лист2", header=None)
rows = [60, 96, 132]
cols = [9, 11, 10]
lenghtes = [31, 29, 27]
resonances_params = np.zeros((3, 3))  # 1 - 67.4
resonances_params[0] = np.array([19.37, 0.2313, 3.6267]) #67.4
resonances_params[1] = np.array([20.97, 0.2313, 4.08]) #57.2
resonances_params[2] = np.array([31.3, 0.233, 5.652]) #25

fit_par = np.zeros((3, 4))

u_03_674 = np.zeros((4, lenghtes[0]))
u_03_674[0], u_03_674[2] = (df2[cols[0]][rows[0] : rows[0] + lenghtes[0]], df2[cols[1]][rows[0] : rows[0] + lenghtes[0]])
u_03_674[0] /= resonances_params[0][0]
u_03_674[2] /= resonances_params[0][1]
u_03_674[1] = np.sqrt((10 ** (-4) * u_03_674[0]) ** 2 + (u_03_674[0] * 10 ** (-4)) ** 2)
u_03_674[3] = np.sqrt((0.05 * u_03_674[2]) ** 2 + (u_03_674[2] * 0.05) ** 2)

u_03_572 = np.zeros((4, lenghtes[1]))
u_03_572[0], u_03_572[2] = (df2[cols[0]][rows[1] : rows[1] + lenghtes[1]], df2[cols[1]][rows[1] : rows[1] + lenghtes[1]])
u_03_572[0] /= resonances_params[1][0]
u_03_572[2] /= resonances_params[1][1]
u_03_572[1] = np.sqrt((10 ** (-4) * u_03_572[0]) ** 2 + (u_03_572[0] * 10 ** (-4)) ** 2)
u_03_572[3] = np.sqrt((0.05 * u_03_572[2]) ** 2 + (u_03_572[2] * 0.05) ** 2)

u_03_25 = np.zeros((4, lenghtes[2]))
u_03_25[0], u_03_25[2] = (df2[cols[0]][rows[2] : rows[2] + lenghtes[2]],df2[cols[1]][rows[2] : rows[2] + lenghtes[2]])
u_03_25[0] /= resonances_params[2][0]
u_03_25[2] /= resonances_params[2][1]
u_03_25[1] = np.sqrt((10 ** (-4) * u_03_25[0]) ** 2 + (u_03_25[0] * 10 ** (-4)) ** 2)
u_03_25[3] = np.sqrt((0.05 * u_03_25[2]) ** 2 + (u_03_25[2] * 0.05 ) ** 2)

(fit_par[0][0], fit_par[0][1], fit_par[0][2], fit_par[0][3]), _ = optimize.curve_fit(approximation, u_03_674[0], u_03_674[2], p0=[basic_properties[10][4],resonances_params[0][0],np.pi / 4,0.5],bounds=[[basic_properties[10][4] * 0.5,resonances_params[0][0] * 0.5,0,0,],[basic_properties[10][4] * 1.5,resonances_params[0][0] * 1.5,np.pi * 2,2,],],method="trf",)  # dogbox
y_fit_03_674 = (approximation(np.sort(u_03_674[0]),fit_par[0][0],fit_par[0][1],fit_par[0][2],fit_par[0][3]))
(fit_par[1][0], fit_par[1][1], fit_par[1][2], fit_par[1][3]), _ = optimize.curve_fit(approximation, u_03_572[0], u_03_572[2], p0=[basic_properties[10][3],resonances_params[1][0],np.pi / 4,0.5],bounds=[[basic_properties[10][3] * 0.5,resonances_params[1][0] * 0.5,0,0,],[basic_properties[10][3] * 1.5,resonances_params[1][0] * 1.5,np.pi * 2,2,],],method="trf",)  # dogbox
y_fit_03_572 = (approximation(np.sort(u_03_572[0]),fit_par[1][0],fit_par[1][1],fit_par[1][2],fit_par[1][3]))
(fit_par[2][0], fit_par[2][1], fit_par[2][2], fit_par[2][3]), _ = optimize.curve_fit(approximation, u_03_25[0], u_03_25[2], p0=[basic_properties[10][0],resonances_params[2][0],np.pi / 4,0.5],bounds=[[basic_properties[10][0] * 0.5,resonances_params[2][0] * 0.5,0,0,],[basic_properties[10][0] * 1.5,resonances_params[2][0] * 1.5,np.pi * 2,2,],],method="trf",)  # dogbox
y_fit_03_25 = (approximation(np.sort(u_03_25[0]),fit_par[2][0],fit_par[2][1],fit_par[2][2],fit_par[2][3]))

colors_palete = ["#05e31b", "#f00a19", "#0879c9", "#000"]

fig1, ax1 = plot_graph([u_03_674, u_03_572, u_03_25, np.array([np.sort(u_03_674[0]), y_fit_03_674]), 
                        np.array([np.sort(u_03_572[0]), y_fit_03_572]), np.array([np.sort(u_03_25[0]), y_fit_03_25])],
                          6, point_start_to_end=[[0.88, 1.12, 5], [3, 27, 5]],
                            titles=[r"f/f${_0}$", r"U${_c}$/$\varepsilon$", r"АЧХ для $0.3U_{c0}$"],
                              colors=colors_palete[:3] * 2, markersizes=[3, 3, 3, 0, 0, 0],
                                labels=[r"$C_5=67.4 нФ$", r"$C_4=57.2 нФ$", r"$C_1=25 нФ$", r"$C_5$, аппроксимация", r"$C_4$, аппроксимация", r"$C_1$, аппроксимация"],
                                  name_fig="AFC.svg", lses=['', '', '', '-'],
                                  points_draw_lines_to=[[1, resonances_params[0][2] / resonances_params[0][1], colors_palete[3], None, None], [1, resonances_params[1][2] / resonances_params[1][1], colors_palete[3], None, None], [1, resonances_params[2][2] / resonances_params[2][1], colors_palete[3], None, None]],
                                    horizontal_vertical_lines=[['h', resonances_params[0][2] / (np.sqrt(2) * resonances_params[0][1]), colors_palete[0], ('', 0), 14],
                                                               ['h', resonances_params[1][2] / (np.sqrt(2) * resonances_params[1][1]), colors_palete[1], ('', 0), 14],
                                                               ['h', resonances_params[2][2] / (np.sqrt(2) * resonances_params[2][1]), colors_palete[2], ('', 0), 14]],
                                                               ticks_and_font_size=[10, 10, 16])


print(fit_par)
(fit_par[0][0], fit_par[0][1], fit_par[0][2], fit_par[0][3]), _ = optimize.curve_fit(approximation, u_03_674[0], u_03_674[2] + u_03_674[3], p0=[basic_properties[10][4],resonances_params[0][0],np.pi / 4,0.5],bounds=[[basic_properties[10][4] * 0.5,resonances_params[0][0] * 0.5,0,0,],[basic_properties[10][4] * 1.5,resonances_params[0][0] * 1.5,np.pi * 2,2,],],method="trf",)  # dogbox
(fit_par[1][0], fit_par[1][1], fit_par[1][2], fit_par[1][3]), _ = optimize.curve_fit(approximation, u_03_572[0], u_03_572[2] + u_03_572[3], p0=[basic_properties[10][3],resonances_params[1][0],np.pi / 4,0.5],bounds=[[basic_properties[10][3] * 0.5,resonances_params[1][0] * 0.5,0,0,],[basic_properties[10][3] * 1.5,resonances_params[1][0] * 1.5,np.pi * 2,2,],],method="trf",)  # dogbox
(fit_par[2][0], fit_par[2][1], fit_par[2][2], fit_par[2][3]), _ = optimize.curve_fit(approximation, u_03_25[0], u_03_25[2] + u_03_25[3], p0=[basic_properties[10][0],resonances_params[2][0],np.pi / 4,0.5],bounds=[[basic_properties[10][0] * 0.5,resonances_params[2][0] * 0.5,0,0,],[basic_properties[10][0] * 1.5,resonances_params[2][0] * 1.5,np.pi * 2,2],],method="trf",)  # dogbox
print(fit_par)
(fit_par[0][0], fit_par[0][1], fit_par[0][2], fit_par[0][3]), _ = optimize.curve_fit(approximation, u_03_674[0], u_03_674[2] - u_03_674[3], p0=[basic_properties[10][4],resonances_params[0][0],np.pi / 4,0.5],bounds=[[basic_properties[10][4] * 0.5,resonances_params[0][0] * 0.5,0,0,],[basic_properties[10][4] * 1.5,resonances_params[0][0] * 1.5,np.pi * 2,2,],],method="trf",)  # dogbox
(fit_par[1][0], fit_par[1][1], fit_par[1][2], fit_par[1][3]), _ = optimize.curve_fit(approximation, u_03_572[0], u_03_572[2] - u_03_572[3], p0=[basic_properties[10][3],resonances_params[1][0],np.pi / 4,0.5],bounds=[[basic_properties[10][3] * 0.5,resonances_params[1][0] * 0.5,0,0,],[basic_properties[10][3] * 1.5,resonances_params[1][0] * 1.5,np.pi * 2,2,],],method="trf",)  # dogbox
(fit_par[2][0], fit_par[2][1], fit_par[2][2], fit_par[2][3]), _ = optimize.curve_fit(approximation, u_03_25[0], u_03_25[2] - u_03_25[3], p0=[basic_properties[10][0],resonances_params[2][0],np.pi / 4,0.5],bounds=[[basic_properties[10][0] * 0.5,resonances_params[2][0] * 0.5,0,0,],[basic_properties[10][0] * 1.5,resonances_params[2][0] * 1.5,np.pi * 2,2,],],method="trf",)  # dogbox
print(fit_par)
print("--------")

#[1, resonances_params[0][2] / resonances_params[0][1], colors_palete[3]], [u_03_572[0][0], u_03_572[2][0], colors_palete[1]], [u_03_25[0][0], u_03_25[2][0], colors_palete[2]]
#r"$\frac{U_{c5}}{\varepsilon \cdot \sqrt{2}}$"

#the last part ФЧХ
sigma_X = 0.25
sigma_X_0 = 0.25
eps_f = 1 * 10**-4

fit_par_freq = np.zeros((3, 3))

phi_03_674 = np.zeros((4, lenghtes[0]))
phi_03_674[0], phi_03_674[2] = (df2[cols[0]][rows[0] : rows[0] + lenghtes[0]], df2[cols[2]][rows[0] : rows[0] + lenghtes[0]])
phi_03_674[0] /= resonances_params[0][0]
phi_03_674[2] /= np.pi
phi_03_674[1] = phi_03_674[0] * eps_f * np.sqrt(2)
phi_03_674[3] = np.sqrt((1/np.array(df2[15][rows[0] : rows[0] + lenghtes[0]], dtype=np.float64) * sigma_X) ** 2 + (np.array(df2[14][rows[0] : rows[0] + lenghtes[0]], dtype=np.float64) * sigma_X_0/ np.array(df2[15][rows[0] : rows[0] + lenghtes[0]], dtype=np.float64) ** 2) ** 2)

phi_03_572 = np.zeros((4, lenghtes[1]))
phi_03_572[0], phi_03_572[2] = (df2[cols[0]][rows[1] : rows[1] + lenghtes[1]], df2[cols[2]][rows[1] : rows[1] + lenghtes[1]])
phi_03_572[0] /= resonances_params[1][0]
phi_03_572[2] /= np.pi
phi_03_572[1] = phi_03_572[0] * eps_f * np.sqrt(2)
phi_03_572[3] = np.sqrt((1/ np.array(df2[15][rows[1] : rows[1] + lenghtes[1]], dtype=np.float64) * sigma_X) ** 2 + (np.array(df2[14][rows[1] : rows[1] + lenghtes[1]], dtype=np.float64) * sigma_X_0/ np.array(df2[15][rows[1] : rows[1] + lenghtes[1]], dtype=np.float64) ** 2) ** 2)

phi_03_25 = np.zeros((4, lenghtes[2]))
phi_03_25[0], phi_03_25[2] = (df2[cols[0]][rows[2] : rows[2] + lenghtes[2]], df2[cols[2]][rows[2] : rows[2] + lenghtes[2]])
phi_03_25[0] /= resonances_params[2][0]
phi_03_25[2] /= np.pi
phi_03_25[1] = phi_03_25[0] * eps_f * np.sqrt(2)
phi_03_25[3] = np.sqrt((1/ np.array(df2[15][rows[2] : rows[2] + lenghtes[2]], dtype=np.float64) * sigma_X) ** 2 + (np.array(df2[14][rows[2] : rows[2] + lenghtes[2]], dtype=np.float64) * sigma_X_0/ np.array(df2[15][rows[2] : rows[2] + lenghtes[2]], dtype=np.float64) ** 2) ** 2)


(fit_par_freq[0][0], fit_par_freq[0][1], fit_par_freq[0][2]), _ = optimize.curve_fit(approximation_PFC, phi_03_674[0], phi_03_674[2], p0=[resonances_params[0][0], np.pi / 4,0.5], bounds=[[resonances_params[0][0] * 0.5, 0, 0],[resonances_params[0][0] * 1.5, np.pi, 2]], method="trf") # dogbox
y_fit_phi_03_674 = (approximation_PFC(np.sort(phi_03_674[0]),fit_par_freq[0][0],fit_par_freq[0][1],fit_par_freq[0][2]))
(fit_par_freq[1][0], fit_par_freq[1][1], fit_par_freq[1][2]), _ = optimize.curve_fit(approximation_PFC, phi_03_572[0], phi_03_572[2], p0=[resonances_params[1][0], np.pi / 4,0.5], bounds=[[resonances_params[1][0] * 0.5, 0, 0],[resonances_params[1][0] * 1.5, np.pi, 2]], method="trf") # dogbox
y_fit_phi_03_572 = (approximation_PFC(np.sort(phi_03_572[0]),fit_par_freq[1][0],fit_par_freq[1][1],fit_par_freq[1][2]))
(fit_par_freq[2][0], fit_par_freq[2][1], fit_par_freq[2][2]), _ = optimize.curve_fit(approximation_PFC, phi_03_25[0], phi_03_25[2], p0=[resonances_params[2][0], np.pi / 4,0.5], bounds=[[resonances_params[2][0] * 0.5, 0, 0],[resonances_params[2][0] * 1.5, np.pi, 2]], method="trf") # dogbox
y_fit_phi_03_25 = (approximation_PFC(np.sort(phi_03_25[0]),fit_par_freq[2][0],fit_par_freq[2][1],fit_par_freq[2][2]))

fig1, ax1 = plot_graph([phi_03_674, phi_03_572, phi_03_25, np.array([np.sort(phi_03_674[0]), y_fit_phi_03_674]), np.array([np.sort(phi_03_572[0]), y_fit_phi_03_572]), np.array([np.sort(phi_03_25[0]), y_fit_phi_03_25])], quant=6, 
           point_start_to_end=[[0.88, 1.12, 5], [0, 1, 6]],
           colors=colors_palete[:3] * 2, titles=[r"f/f$_0$", r"$\phi /\pi$", r"ФЧХ для $0.3U_{c0}$"],
           labels=[r"$C_5=67.4 нФ$", r"$C_4=57.2 нФ$", r"$C_1=25 нФ$", r"$C_5$, аппроксимация", r"$C_4$, аппроксимация", r"$C_1$, аппроксимация"],
           legend_position="upper left", lses=['', '', '', '-', '-', '-'], markersizes=[3, 3, 3, 0, 0, 0],
           horizontal_vertical_lines=[['h', 1/4, colors_palete[3], ('1/4', 0.94), 10],
                                      ['h', 3/4, colors_palete[3], ('3/4', 0.99), 10]],
                                      save_flag=False, ticks_and_font_size=[10, 10, 16])
ax1.axvline(x=np.sort(phi_03_25[0])[7], ymax=1/4, color=colors_palete[2], linestyle='dashed')
ax1.axvline(x=np.sort(phi_03_25[0])[lenghtes[2] - 7], ymax=3/4, color=colors_palete[2], linestyle='dashed')
ax1.axvline(x=np.sort(phi_03_572[0])[10], ymax=1/4, color=colors_palete[1], linestyle='dashed')
ax1.axvline(x=np.sort(phi_03_572[0])[lenghtes[1] - 7] + 0.001, ymax=3/4, color=colors_palete[1], linestyle='dashed')
ax1.axvline(x=np.sort(phi_03_674[0])[9], ymax=1/4, color=colors_palete[0], linestyle='dashed')
ax1.axvline(x=np.sort(phi_03_674[0])[lenghtes[0]- 9], ymax=3/4, color=colors_palete[0], linestyle='dashed')

plt.savefig("PFC.svg")

print(fit_par_freq)
(fit_par_freq[0][0], fit_par_freq[0][1], fit_par_freq[0][2]), _ = optimize.curve_fit(approximation_PFC, phi_03_674[0], phi_03_674[2] + phi_03_674[3], p0=[resonances_params[0][0], np.pi / 4,0.5], bounds=[[resonances_params[0][0] * 0.5, 0, 0],[resonances_params[0][0] * 1.5, np.pi, 2]], method="trf") # dogbox
(fit_par_freq[1][0], fit_par_freq[1][1], fit_par_freq[1][2]), _ = optimize.curve_fit(approximation_PFC, phi_03_572[0], phi_03_572[2] +phi_03_572[3], p0=[resonances_params[1][0], np.pi / 4,0.5], bounds=[[resonances_params[1][0] * 0.5, 0, 0],[resonances_params[1][0] * 1.5, np.pi, 2]], method="trf") # dogbox
(fit_par_freq[2][0], fit_par_freq[2][1], fit_par_freq[2][2]), _ = optimize.curve_fit(approximation_PFC, phi_03_25[0], phi_03_25[2] + phi_03_25[3], p0=[resonances_params[2][0], np.pi / 4,0.5], bounds=[[resonances_params[2][0] * 0.5, 0, 0],[resonances_params[2][0] * 1.5, np.pi, 2]], method="trf") # dogbox
print(fit_par_freq)
(fit_par_freq[0][0], fit_par_freq[0][1], fit_par_freq[0][2]), _ = optimize.curve_fit(approximation_PFC, phi_03_674[0], phi_03_674[2] - phi_03_674[3], p0=[resonances_params[0][0], np.pi / 4,0.5], bounds=[[resonances_params[0][0] * 0.5, 0, 0],[resonances_params[0][0] * 1.5, np.pi, 2]], method="trf") # dogbox
(fit_par_freq[1][0], fit_par_freq[1][1], fit_par_freq[1][2]), _ = optimize.curve_fit(approximation_PFC, phi_03_572[0], phi_03_572[2] - phi_03_572[3], p0=[resonances_params[1][0], np.pi / 4,0.5], bounds=[[resonances_params[1][0] * 0.5, 0, 0],[resonances_params[1][0] * 1.5, np.pi, 2]], method="trf") # dogbox
(fit_par_freq[2][0], fit_par_freq[2][1], fit_par_freq[2][2]), _ = optimize.curve_fit(approximation_PFC, phi_03_25[0], phi_03_25[2] - phi_03_25[3], p0=[resonances_params[2][0], np.pi / 4,0.5], bounds=[[resonances_params[2][0] * 0.5, 0, 0],[resonances_params[2][0] * 1.5, np.pi, 2]], method="trf") # dogbox
print(fit_par_freq)