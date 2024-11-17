import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize
from matplotlib.ticker import FormatStrFormatter
from functions import plot_graph, mnk

np.set_printoptions(suppress=True)


def nice_print(data, names):
    print(end="\t")
    for i in range(len(names)):
        print(i, end="\t")
    print(end="\n \t")

    for i in range(len(names)):
        print(names[i], end="\t")
    print(end="\n")
    for i in range(data.shape[1]):
        print(i, end="\t")
        for j in range(data.shape[0]):
            print(np.round(data[j][i], decimals=5), end="\t")
        print(end="\n")


def approximation(x, f_0, U_c0, phi_c, tau):
    return U_c0 * f_0 / x * np.cos(-phi_c) / np.sqrt(1 + (tau * (x - f_0)) ** 2)
def approximation_PFC(x, f_0, tau, delta):
    return np.arctan(tau * (x - f_0)) + np.pi / 2 - delta


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
basic_properties[8] = 1 / (
    ((basic_properties[2] * 10 ** (3)) * 2 * np.pi) ** 2
    * (basic_properties[0] * 10 ** (-9))
)
basic_properties[9] = (
    1
    / (
        2
        * np.pi**2
        * (basic_properties[0] * 10 ** (-9))
        * (basic_properties[2] * 10 ** (3)) ** 3
    )
    * basic_properties[3]
)
basic_properties[10] = np.round(basic_properties[4] / basic_properties[6], decimals=4)
basic_properties[11] = np.round(
    np.sqrt(
        (basic_properties[5] / basic_properties[6]) ** 2
        + (basic_properties[4] * basic_properties[7] / basic_properties[6]) ** 2
    ),
    decimals=4,
)
basic_properties[12] = np.round(
    np.sqrt(basic_properties[8] / (basic_properties[0] * 10 ** (-9))), decimals=2
)
basic_properties[13] = np.sqrt(
    basic_properties[9] ** 2
    / (4 * basic_properties[8] * (basic_properties[0] * 10 ** (-9)))
)
basic_properties[14] = np.round(
    np.sqrt(basic_properties[8] / (basic_properties[0] * 10 ** (-9)))
    / basic_properties[10],
    decimals=2,
)
basic_properties[15] = np.sqrt(
    (
        basic_properties[11] ** 2
        * basic_properties[8]
        / ((basic_properties[0] * 10 ** (-9)) * basic_properties[10] ** 4)
    )
    + (
        basic_properties[9] ** 2
        / (
            4
            * basic_properties[10] ** 2
            * basic_properties[8]
            * (basic_properties[0] * 10 ** (-9))
        )
    )
)
basic_properties[16] = basic_properties[12] * 10**-3
basic_properties[17] = basic_properties[13] * 10**-3
basic_properties[18] = basic_properties[14] - R - basic_properties[16]
basic_properties[19] = np.sqrt(basic_properties[15] ** 2 + basic_properties[17] ** 2)
basic_properties[20] = basic_properties[6] / basic_properties[16]
basic_properties[21] = np.sqrt(
    (basic_properties[7] / basic_properties[14]) ** 2
    + (basic_properties[6] * basic_properties[15] / basic_properties[14] ** 2) ** 2
)
names = [
    "C нФ",
    "dC нФ",
    "f_0kHz",
    "df_0kHz",
    "Uc V",
    "dUc V",
    "E B",
    "dE B",
    "L Гн",
    "dL Гн",
    "Q",
    "dQ",
    "p Ом",
    "dp Ом",
    "Rsu Ом",
    "dRsu Ом",
    "RSm Ом",
    "dRSm Ом",
    "R_L Ом",
    "dR_L Ом",
    "I А",
    "dI А",
]
nice_print(data=basic_properties, names=names)
averages = np.zeros((2, 2))
averages[0][0] = np.mean(basic_properties[8])
averages[1][0] = np.sqrt(np.sum((basic_properties[8] - averages[0][0]) ** 2)) / (
    basic_properties[8].shape[0] * (basic_properties[8].shape[0] - 1)
)
averages[0][1] = np.mean(basic_properties[18])
averages[1][1] = np.sqrt(np.sum((basic_properties[18] - averages[0][1]) ** 2)) / (
    basic_properties[18].shape[0] * (basic_properties[18].shape[0] - 1)
)
print(averages[0], averages[1], sep="\n")

del df1

# second part
df2 = pd.read_excel(excel_file1, sheet_name="Лист2", header=None)
rows = [1, 31, 132]
cols = [3, 4, 10, 11]
lenghtes = [27, 28, 26, 19, 46]
resonances_params = np.zeros((3, 4))  # 1 - 67.4
resonances_params[0] = np.array(
    [19.37, 3.6267, basic_properties[10][4], basic_properties[6][4]]
)
resonances_params[1] = np.array(
    [20.97, 3.9791, basic_properties[10][3], basic_properties[6][3]]
)
resonances_params[2] = np.array(
    [31.3, 5.65, basic_properties[10][0], basic_properties[6][0]]
)
fit_par = np.zeros((5, 4))
u_06_674 = np.zeros((4, lenghtes[0]))
u_06_674[0], u_06_674[2] = (
    df2[cols[0]][rows[0] : rows[0] + lenghtes[0]] / resonances_params[0][0],
    df2[cols[1]][rows[0] : rows[0] + lenghtes[0]] / resonances_params[0][1],
)
u_06_674[1] = np.sqrt((10 ** (-4) * u_06_674[0]) ** 2 + (u_06_674[0] * 10 ** (-4)) ** 2)
u_06_674[3] = np.sqrt((0.05 / 100 * u_06_674[2]) ** 2 + (u_06_674[2] * 0.05 / 100) ** 2)

u_03_674 = np.zeros((4, lenghtes[1]))
u_03_674[0], u_03_674[2] = (
    df2[cols[2]][rows[0] : rows[0] + lenghtes[1]] / resonances_params[0][0],
    df2[cols[3]][rows[0] : rows[0] + lenghtes[1]] / resonances_params[0][1],
)
u_03_674[1] = np.sqrt((10 ** (-4) * u_03_674[0]) ** 2 + (u_03_674[0] * 10 ** (-4)) ** 2)
u_03_674[3] = np.sqrt((0.05 / 100 * u_03_674[2]) ** 2 + (u_03_674[2] * 0.05 / 100) ** 2)

u_06_572 = np.zeros((4, lenghtes[2]))
u_06_572[0], u_06_572[2] = (
    df2[cols[0]][rows[1] : rows[1] + lenghtes[2]] / resonances_params[1][0],
    df2[cols[1]][rows[1] : rows[1] + lenghtes[2]] / resonances_params[1][1],
)
u_06_572[1] = np.sqrt((10 ** (-4) * u_06_572[0]) ** 2 + (u_06_572[0] * 10 ** (-4)) ** 2)
u_06_572[3] = np.sqrt((0.05 / 100 * u_06_572[2]) ** 2 + (u_06_572[2] * 0.05 / 100) ** 2)

u_03_572 = np.zeros((4, lenghtes[3]))
u_03_572[0], u_03_572[2] = (
    df2[cols[2]][rows[1] : rows[1] + lenghtes[3]] / resonances_params[1][0],
    df2[cols[3]][rows[1] : rows[1] + lenghtes[3]] / resonances_params[1][1],
)
u_03_572[1] = np.sqrt((10 ** (-4) * u_03_572[0]) ** 2 + (u_03_572[0] * 10 ** (-4)) ** 2)
u_03_572[3] = np.sqrt((0.05 / 100 * u_03_572[2]) ** 2 + (u_03_572[2] * 0.05 / 100) ** 2)

u_05_25 = np.zeros((4, lenghtes[4]))
u_05_25[0], u_05_25[2] = (
    df2[cols[0]][rows[2] : rows[2] + lenghtes[4]] / resonances_params[2][0],
    df2[cols[1]][rows[2] : rows[2] + lenghtes[4]] / resonances_params[2][1],
)
u_05_25[1] = np.sqrt((10 ** (-4) * u_05_25[0]) ** 2 + (u_05_25[0] * 10 ** (-4)) ** 2)
u_05_25[3] = np.sqrt((0.05 / 100 * u_05_25[2]) ** 2 + (u_05_25[2] * 0.05 / 100) ** 2)

(fit_par[0][0], fit_par[0][1], fit_par[0][2], fit_par[0][3]), _ = optimize.curve_fit(
    approximation,
    u_03_674[0] * resonances_params[0][0],
    u_03_674[2] * resonances_params[0][1],
    p0=[
        resonances_params[0][0],
        resonances_params[0][1],
        np.arctan(1) + np.pi / 2,
        0.5,
    ],
    bounds=[
        [
            resonances_params[0][0] * 0.9,
            resonances_params[0][1] * 0.9,
            0,
            0,
        ],
        [
            resonances_params[0][0] * 1.0000000001,
            resonances_params[0][1] * 1.0000000001,
            np.pi * 2,
            2,
        ],
    ],
    method="trf",
)  # dogbox
(fit_par[1][0], fit_par[1][1], fit_par[1][2], fit_par[1][3]), _ = optimize.curve_fit(
    approximation,
    u_03_572[0] * resonances_params[1][0],
    u_03_572[2] * resonances_params[1][1],
    p0=[
        resonances_params[1][0],
        resonances_params[1][1],
        np.arctan(1) + np.pi / 2,
        0.5,
    ],
    bounds=[
        [resonances_params[1][0] * 0.9, resonances_params[1][1] * 0.9, 0, 0],
        [
            resonances_params[1][0] * 1.0000000001,
            resonances_params[1][1] * 1.0000000001,
            np.pi * 2,
            2,
        ],
    ],
    method="trf",
)  # dogbox

# 21.031395195551767 3.983079099999999 1.759019328195435e-19 1.6151147479898957
print(fit_par[0][0], fit_par[0][1], fit_par[0][2], fit_par[0][3]) #674
print(fit_par[1][0], fit_par[1][1], fit_par[1][2], fit_par[1][3]) #572

y_fit_674_03 = (
    approximation(
        np.sort(u_03_674[0]) * resonances_params[0][0],
        fit_par[0][0],
        fit_par[0][1],
        fit_par[0][2],
        fit_par[0][3],
    )
    / fit_par[0][1]
)
y_fit_572_03 = (
    approximation(
        np.sort(u_03_572[0]) * resonances_params[1][0],
        fit_par[1][0],
        fit_par[1][1],
        fit_par[1][2],
        fit_par[1][3],
    )
    / fit_par[1][1]
)

fig, ax = plt.subplots()
fig.suptitle(r"АЧХ для $0.3U_{c0}$", fontsize=14, fontweight="bold")

ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.spines["bottom"].set_position(
    (
        "data",
        min(
            np.min(u_03_674[2]),
            np.min(u_03_572[2]),
            #np.min(u_05_25[2]),
            np.min(y_fit_674_03),
            np.min(y_fit_572_03),
        )
        * 0.98,
    )
)
ax.spines["left"].set_position(
    ("data", min(np.min(u_03_674[0]), 
                 np.min(u_03_572[0]), 
                 #np.min(u_05_25[0])
                 ) * 0.98)
)
ax.set(
    xlim=(
        min(np.min(u_03_674[0]),
            np.min(u_03_572[0]), 
            #np.min(u_05_25[0])
            ) * 0.98,
        max(np.max(u_03_674[0]), 
            np.max(u_03_572[0]), 
            #np.max(u_05_25[0])
            ) * 1.02,
    ),
    ylim=(
        min(
            np.min(u_03_674[2]),
            np.min(u_03_572[2]),
            # np.min(u_05_25[2]),
            np.min(y_fit_674_03),
            np.min(y_fit_572_03),
        )
        * 0.98,
        max(
            np.max(u_03_674[2]),
            np.max(u_03_572[2]),
            # np.max(u_05_25[2]),
            np.max(y_fit_674_03),
            np.max(y_fit_572_03),
        )
        * 1.05,
    ),
)
plt.xticks(
    np.linspace(
        min(np.min(u_03_674[0]), 
            np.min(u_03_572[0]), 
            #np.min(u_05_25[0])
            ) * 0.98,
        max(np.max(u_03_674[0]), 
            np.max(u_03_572[0]), 
            #np.max(u_05_25[0])
            ) * 1.02,
        8,
    ),
    rotation=0,
    size=7,
)
plt.yticks(
    np.linspace(
        min(
            np.min(u_03_674[2]),
            np.min(u_03_572[2]),
            #np.min(u_05_25[2]),
            np.min(y_fit_674_03),
            np.min(y_fit_572_03),
        )
        * 0.98,
        max(
            np.max(u_03_674[2]),
            np.max(u_03_572[2]),
            #np.max(u_05_25[2]),
            np.max(y_fit_674_03),
            np.max(y_fit_572_03),
        )
        * 1.05,
        8,
    ),
    size=7,
)

plt.xlabel(r"$\frac{f}{f_0}$", loc="right")
plt.ylabel(r"$\frac{U_c}{U_{c0}}$", loc="top")

ax.errorbar(
    u_03_674[0],
    u_03_674[2],
    xerr=u_03_674[1],
    yerr=u_03_674[3],
    lw=0.5,
    color="g",
    marker="o",
    label=r"$C_5$=67.4 нФ",
    markersize=3,
    ls="",
)
ax.errorbar(
    u_03_572[0],
    u_03_572[2],
    xerr=u_03_572[1],
    yerr=u_03_572[3],
    lw=0.5,
    color="b",
    marker="o",
    label=r"$C_4$=57.2 нФ",
    markersize=3,
    ls="",
)

# ax.errorbar(
#     u_05_25[0],
#     u_05_25[2],
#     xerr=u_05_25[1],
#     yerr=u_05_25[3],
#     lw=0.5,
#     color="r",
#     marker="o",
#     label=r"$C_1$=25 нФ",
#     markersize=3,
#     ls="",
# )
ax.plot(
    np.sort(u_03_674[0]),
    y_fit_674_03,
    lw=0.5,
    color="g",
    marker="",
    label=r"$C_5$, approximation",
    markersize=0,
    ls="-",
)
ax.plot(
    np.sort(u_03_572[0]),
    y_fit_572_03,
    lw=0.5,
    color="b",
    marker="",
    label=r"$C_4$, approximation",
    markersize=0,
    ls="-",
)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.legend(loc="upper right")
plt.grid(linewidth="0.2")
plt.savefig("АЧХ_03.svg")
plt.show()
plt.clf()
plt.cla()
plt.close()
del fig
del ax
fig, ax = plt.subplots()

(fit_par[2][0], fit_par[2][1], fit_par[2][2], fit_par[2][3]), _ = optimize.curve_fit(
    approximation,
    u_06_674[0] * resonances_params[0][0],
    u_06_674[2] * resonances_params[0][1],
    p0=[
        resonances_params[0][0],
        resonances_params[0][1],
        np.arctan(1) + np.pi / 2,
        0.5,
    ],
    bounds=[
        [
            resonances_params[0][0] * 0.9,
            resonances_params[0][1] * 0.9,
            0,
            0,
        ],
        [
            resonances_params[0][0] * 1.0000001,
            resonances_params[0][1] * 1.0000001,
            np.pi * 2,
            2,
        ],
    ],
    method="trf",
)  # dogbox
(fit_par[3][0], fit_par[3][1], fit_par[3][2], fit_par[3][3]), _ = optimize.curve_fit(
    approximation,
    u_06_572[0] * resonances_params[1][0],
    u_06_572[2] * resonances_params[1][1],
    p0=[
        resonances_params[1][0],
        resonances_params[1][1],
        np.arctan(1) + np.pi / 2,
        0.5,
    ],
    bounds=[
        [resonances_params[1][0] * 0.9, resonances_params[1][1] * 0.9, 0, 0],
        [
            resonances_params[1][0] * 1.001,
            resonances_params[1][1] * 1.001,
            np.pi * 2,
            2,
        ],
    ],
    method="trf",
)  # dogbox
(fit_par[4][0], fit_par[4][1], fit_par[4][2], fit_par[4][3]), _ = optimize.curve_fit(
    approximation,
    u_05_25[0] * resonances_params[2][0],
    u_05_25[2] * resonances_params[2][1],
    p0=[
        resonances_params[2][0],
        resonances_params[2][1],
        np.arctan(1) + np.pi / 2,
        0.5,
    ],
    bounds=[
        [resonances_params[2][0] * 0.9, resonances_params[2][1] * 0.9, 0, 0],
        [
            resonances_params[2][0] * 1.0000000001,
            resonances_params[2][1] * 1.0000000001,
            np.pi * 2,
            2,
        ],
    ],
    method="trf",
)  # dogbox

print(fit_par[2][0], fit_par[2][1], fit_par[2][2], fit_par[2][3], " 674 -06") #674
print(fit_par[3][0], fit_par[3][1], fit_par[3][2], fit_par[3][3], " 572 -06") #572
print(fit_par[4][0], fit_par[4][1], fit_par[4][2], fit_par[4][3], " 25 -05") #25

y_fit_674_06 = (
    approximation(
        np.sort(u_06_674[0]) * resonances_params[0][0],
        fit_par[2][0],
        fit_par[2][1],
        fit_par[2][2],
        fit_par[2][3],
    )
    / fit_par[2][1]
)
y_fit_572_06 = (
    approximation(
        np.sort(u_06_572[0]) * resonances_params[1][0],
        fit_par[3][0],
        fit_par[3][1],
        fit_par[3][2],
        fit_par[3][3],
    )
    / fit_par[3][1]
)
y_fit_25_05 = (
    approximation(
        np.sort(u_05_25[0]) * resonances_params[2][0],
        fit_par[4][0],
        fit_par[4][1],
        fit_par[4][2],
        fit_par[4][3],
    )
    / fit_par[4][1]
)

fig.suptitle(r"АЧХ для $0.6U_{c0}$", fontsize=14, fontweight="bold")

ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.spines["bottom"].set_position(
    (
        "data",
        min(
            np.min(u_06_674[2]),
            np.min(u_06_572[2]),
            np.min(u_05_25[2]),
            np.min(y_fit_674_06),
            np.min(y_fit_572_06),
            np.min(y_fit_25_05),
        )
        * 0.98,
    )
)
ax.spines["left"].set_position(
    ("data", min(np.min(u_06_674[0]), 
                 np.min(u_06_572[0]), 
                 np.min(u_05_25[0])
                 ) * 0.998)
)
ax.set(
    xlim=(
        min(np.min(u_06_674[0]),
            np.min(u_06_572[0]), 
            np.min(u_05_25[0])
            ) * 0.998,
        max(np.max(u_06_674[0]), 
            np.max(u_06_572[0]), 
            np.max(u_05_25[0])
            ) * 1.002,
    ),
    ylim=(
        min(
            np.min(u_06_674[2]),
            np.min(u_06_572[2]),
            np.min(u_05_25[2]),
            np.min(y_fit_674_06),
            np.min(y_fit_572_06),
            np.min(y_fit_25_05),
        )
        * 0.98,
        max(
            np.max(u_06_674[2]),
            np.max(u_06_572[2]),
            np.max(u_05_25[2]),
            np.max(y_fit_674_06),
            np.max(y_fit_572_06),
            np.max(y_fit_25_05),
        )
        * 1.05,
    ),
)
plt.xticks(
    np.linspace(
        min(np.min(u_06_674[0]), 
            np.min(u_06_572[0]), 
            np.min(u_05_25[0])
            ) * 0.998,
        max(np.max(u_06_674[0]), 
            np.max(u_06_572[0]), 
            np.max(u_05_25[0])
            ) * 1.002,
        8,
    ),
    rotation=0,
    size=7,
)
plt.yticks(
    np.linspace(
        min(
            np.min(u_06_674[2]),
            np.min(u_06_572[2]),
            np.min(u_05_25[2]),
            np.min(y_fit_674_06),
            np.min(y_fit_572_06),
            np.min(y_fit_25_05),
        )
        * 0.98,
        max(
            np.max(u_06_674[2]),
            np.max(u_06_572[2]),
            np.max(u_05_25[2]),
            np.max(y_fit_674_06),
            np.max(y_fit_572_06),
            np.max(y_fit_25_05),
        )
        * 1.05,
        8,
    ),
    size=7,
)

plt.xlabel(r"$\frac{f}{f_0}$", loc="right")
plt.ylabel(r"$\frac{U_c}{U_{c0}}$", loc="top")
ax.errorbar(
    u_06_674[0],
    u_06_674[2],
    xerr=u_06_674[1],
    yerr=u_06_674[3],
    lw=0.5,
    color="g",
    marker="o",
    label=r"$C_5$=67.4 нФ",
    markersize=3,
    ls="",
)
ax.errorbar(
    u_06_572[0],
    u_06_572[2],
    xerr=u_06_572[1],
    yerr=u_06_572[3],
    lw=0.5,
    color="b",
    marker="o",
    label=r"$C_4$=57.2 нФ",
    markersize=3,
    ls="",
)

ax.errorbar(
    u_05_25[0],
    u_05_25[2],
    xerr=u_05_25[1],
    yerr=u_05_25[3],
    lw=0.5,
    color="r",
    marker="o",
    label=r"$C_1$=25 нФ",
    markersize=3,
    ls="",
)
ax.plot(
    np.sort(u_06_674[0]),
    y_fit_674_06,
    lw=0.5,
    color="g",
    marker="",
    label=r"$C_5$, approximation",
    markersize=0,
    ls="-",
)
ax.plot(
    np.sort(u_06_572[0]),
    y_fit_572_06,
    lw=0.5,
    color="b",
    marker="",
    label=r"$C_4$, approximation",
    markersize=0,
    ls="-",
)
ax.plot(
    np.sort(u_05_25[0]),
    y_fit_25_05,
    lw=0.5,
    color="r",
    marker="",
    label=r"$C_1$, approximation",
    markersize=0,
    ls="-",
)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#plt.legend(loc='upper right')
plt.legend(bbox_to_anchor=(0.75, 0.75))
plt.grid(linewidth="0.2")
plt.savefig("АЧХ_06.svg")
plt.show()
plt.clf()
plt.cla()
plt.close()
del fig
del ax

#the last part ФЧХ
rows = [60, 96, 132]
cols = [9, 10, 14, 15]
lenghtes = [31, 29, 27]
resonances_params = np.zeros((3, 2))  # 1 - 67.4
resonances_params[0] = np.array(
    [19.37,  df2[cols[1]][rows[0]]]
)
resonances_params[1] = np.array(
    [20.97,  df2[cols[1]][rows[1]]]
)
resonances_params[2] = np.array(
    [31.3, df2[cols[1]][rows[2]]]
)
fit_par_freq = np.zeros((3, 4))

phi_03_674 = np.zeros((4, lenghtes[0]))
phi_03_674[0], phi_03_674[2] = (
    df2[cols[0]][rows[0] : rows[0] + lenghtes[0]] / resonances_params[0][0],
    df2[cols[1]][rows[0] : rows[0] + lenghtes[0]] / resonances_params[0][1],
)
phi_03_674[1] = np.sqrt((10 ** (-4) * phi_03_674[0]) ** 2 + (phi_03_674[0] * 10 ** (-4)) ** 2)
phi_03_674[3] = np.sqrt((np.pi / np.array(df2[cols[3]][rows[0] : rows[0] + lenghtes[0]], dtype=np.float64) * 0.25) ** 2 + (np.array(df2[cols[2]][rows[0] : rows[0] + lenghtes[0]], dtype=np.float64) * np.pi * 0.25/ np.array(df2[cols[3]][rows[0] : rows[0] + lenghtes[0]], dtype=np.float64) ** 2) ** 2)

phi_03_572 = np.zeros((4, lenghtes[1]))
phi_03_572[0], phi_03_572[2] = (
    df2[cols[0]][rows[1] : rows[1] + lenghtes[1]] / resonances_params[1][0],
    df2[cols[1]][rows[1] : rows[1] + lenghtes[1]] / resonances_params[1][1],
)
phi_03_572[1] = np.sqrt((10 ** (-4) * phi_03_572[0]) ** 2 + (phi_03_572[0] * 10 ** (-4)) ** 2)
phi_03_572[3] = np.sqrt((np.pi / np.array(df2[cols[3]][rows[1] : rows[1] + lenghtes[1]], dtype=np.float64) * 0.25) ** 2 + (np.array(df2[cols[2]][rows[1] : rows[1] + lenghtes[1]], dtype=np.float64) * np.pi * 0.25/ np.array(df2[cols[3]][rows[1] : rows[1] + lenghtes[1]], dtype=np.float64) ** 2) ** 2)

phi_03_25 = np.zeros((4, lenghtes[2]))
phi_03_25[0], phi_03_25[2] = (
    df2[cols[0]][rows[2] : rows[2] + lenghtes[2]] / resonances_params[2][0],
    df2[cols[1]][rows[2] : rows[2] + lenghtes[2]] / resonances_params[2][1],
)
phi_03_25[1] = np.sqrt((10 ** (-4) * phi_03_25[0]) ** 2 + (phi_03_25[0] * 10 ** (-4)) ** 2)
phi_03_25[3] = np.sqrt((np.pi / np.array(df2[cols[3]][rows[2] : rows[2] + lenghtes[2]], dtype=np.float64) * 0.25) ** 2 + (np.array(df2[cols[2]][rows[2] : rows[2] + lenghtes[2]], dtype=np.float64) * np.pi * 0.25/ np.array(df2[cols[3]][rows[2] : rows[2] + lenghtes[2]], dtype=np.float64) ** 2) ** 2)

(fit_par_freq[0][0], fit_par_freq[0][1], fit_par_freq[0][2]), _ = optimize.curve_fit(
    approximation_PFC,
    phi_03_674[0] * resonances_params[0][0],
    phi_03_674[2] * resonances_params[0][1],
    p0=[
        resonances_params[0][0],
        fit_par[0][3],
        0.1,
    ],
    bounds=[
        [
            resonances_params[0][0] * 0.99,
            fit_par[0][3] * 0.99,
            0,
        ],
        [
            resonances_params[0][0] * 1.001,
            fit_par[0][3] * 1.001,
            np.pi ,
        ],
    ],
    method="trf",
 ) # dogbox

(fit_par_freq[1][0], fit_par_freq[1][1], fit_par_freq[1][2]), _ = optimize.curve_fit(
    approximation_PFC,
    phi_03_572[0] * resonances_params[1][0],
    phi_03_572[2] * resonances_params[1][1],
    p0=[
        resonances_params[1][0],
        fit_par[1][3],
        0.1,
    ],
    bounds=[
        [
            resonances_params[1][0] * 0.99,
            fit_par[1][3] * 0.99,
            0,
        ],
        [
            resonances_params[1][0] * 1.001,
            fit_par[1][3] * 1.001,
            np.pi ,
        ],
    ],
    method="trf",
 ) # dogbox
(fit_par_freq[2][0], fit_par_freq[2][1], fit_par_freq[2][2]), _ = optimize.curve_fit(
    approximation_PFC,
    phi_03_25[0] * resonances_params[2][0],
    phi_03_25[2] * resonances_params[2][1],
    p0=[
        resonances_params[2][0],
        fit_par[4][3],
        0.1,
    ],
    bounds=[
        [
            resonances_params[2][0] * 0.99,
            fit_par[4][3] * 0.99,
            0,
        ],
        [
            resonances_params[2][0] * 1.001,
            fit_par[4][3] * 1.001,
            np.pi ,
        ],
    ],
    method="trf",
 ) # dogbox

print(fit_par_freq[0][0], fit_par_freq[0][1], fit_par_freq[0][2], " 674")
print(fit_par_freq[1][0], fit_par_freq[1][1], fit_par_freq[1][2], " 572")
print(fit_par_freq[2][0], fit_par_freq[2][1], fit_par_freq[2][2], " 25")

y_fit_674_03 = (
    approximation_PFC(
        np.sort(phi_03_674[0]) * resonances_params[0][0],
        fit_par_freq[0][0],
        fit_par_freq[0][1],
        fit_par_freq[0][2],
    )
    / resonances_params[0][1]
)

y_fit_572_03 = (
    approximation_PFC(
        np.sort(phi_03_572[0]) * resonances_params[1][0],
        fit_par_freq[1][0],
        fit_par_freq[1][1],
        fit_par_freq[1][2],
    )
    / resonances_params[1][1]
)

y_fit_25_03 = (
    approximation_PFC(
        np.sort(phi_03_25[0]) * resonances_params[2][0],
        fit_par_freq[2][0],
        fit_par_freq[2][1],
        fit_par_freq[2][2],
    )
    / resonances_params[2][1]
)

fig, ax = plt.subplots()
fig.suptitle(r"ФЧХ для $0.3U_{c0}$", fontsize=14, fontweight="bold") #$C_4=57.2 нФ$

ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")
ax.spines["bottom"].set_position(
    (
        "data",
        min(
            np.min(phi_03_674[2]-phi_03_674[3]),
            np.min(phi_03_572[2]-phi_03_572[3]),
            np.min(phi_03_25[2] - phi_03_25[3]),
            np.min(y_fit_674_03),
            np.min(y_fit_572_03),
            np.min(y_fit_25_05),
        )
        * 0.95,
    )
)
ax.spines["left"].set_position(
    ("data", min(np.min(phi_03_674[0]), 
                 np.min(phi_03_572[0]), 
                 np.min(phi_03_25[0])
                 ) * 0.998)
)
ax.set(
    xlim=(
        min(np.min(phi_03_674[0]),
            np.min(phi_03_572[0]), 
            np.min(phi_03_25[0])
            ) * 0.998,
        max(np.max(phi_03_674[0]), 
            np.max(phi_03_572[0]), 
            np.max(phi_03_25[0])
            ) * 1.002,
    ),
    ylim=(
        min(
            np.min(phi_03_674[2]-phi_03_674[3]),
            np.min(phi_03_572[2]-phi_03_572[3]),
            np.min(phi_03_25[2] - phi_03_25[3]),
            np.min(y_fit_674_03),
            np.min(y_fit_572_03),
            np.min(y_fit_25_05),
        )
        * 0.95,
        max(
            np.max(phi_03_674[2] + phi_03_674[3]),
            np.max(phi_03_572[2] + phi_03_572[3]),
            np.max(phi_03_25[2] + phi_03_25[3]),
            np.max(y_fit_674_03),
            np.max(y_fit_572_03),
            np.max(y_fit_25_05),
        )
        * 1.05,
    ),
)
plt.xticks(
    np.linspace(
        min(np.min(phi_03_674[0]), 
            np.min(phi_03_572[0]), 
            np.min(phi_03_25[0])
            ) * 0.998,
        max(np.max(phi_03_674[0]), 
            np.max(phi_03_572[0]), 
            np.max(phi_03_25[0])
            ) * 1.002,
        8,
    ),
    rotation=0,
    size=7,
)
plt.yticks(
    np.linspace(
        min(
            np.min(phi_03_674[2]-phi_03_674[3]),
            np.min(phi_03_572[2]-phi_03_572[3]),
            np.min(phi_03_25[2] - phi_03_25[3]),
            np.min(y_fit_674_03),
            np.min(y_fit_572_03),
            np.min(y_fit_25_05),
        )
        * 0.95,
        max(
            np.max(phi_03_674[2] + phi_03_674[3]),
            np.max(phi_03_572[2] + phi_03_572[3]),
            np.max(phi_03_25[2] + phi_03_25[3]),
            np.max(y_fit_674_03),
            np.max(y_fit_572_03),
            np.max(y_fit_25_05),
        )
        * 1.05,
        8,
    ),
    size=7,
)

plt.xlabel(r"$\frac{f}{f_0}$", loc="right")
plt.ylabel(r"$\frac{\phi}{\phi_{0}}$", loc="top")
ax.errorbar(
    phi_03_674[0],
    phi_03_674[2],
    xerr=phi_03_674[1],
    yerr=phi_03_674[3],
    lw=0.5,
    color="g",
    marker="o",
    label=r"$C_5$=67.4 нФ",
    markersize=3,
    ls="",
)
ax.errorbar(
    phi_03_572[0],
    phi_03_572[2],
    xerr=phi_03_572[1],
    yerr=phi_03_572[3],
    lw=0.5,
    color="b",
    marker="o",
    label=r"$C_4$=57.2 нФ",
    markersize=3,
    ls="",
)

ax.errorbar(
    phi_03_25[0],
    phi_03_25[2],
    xerr=phi_03_25[1],
    yerr=phi_03_25[3],
    lw=0.5,
    color="r",
    marker="o",
    label=r"$C_1$=25 нФ",
    markersize=3,
    ls="",
)
ax.plot(
    np.sort(phi_03_674[0]),
    y_fit_674_03,
    lw=0.5,
    color="g",
    marker="",
    label=r"$C_5$, approximation",
    markersize=0,
    ls="-",
)
ax.plot(
    np.sort(phi_03_572[0]),
    y_fit_572_03,
    lw=0.5,
    color="b",
    marker="",
    label=r"$C_4$, approximation",
    markersize=0,
    ls="-",
)
ax.plot(
    np.sort(phi_03_25[0]),
    y_fit_25_03,
    lw=0.5,
    color="r",
    marker="",
    label=r"$C_1$, approximation",
    markersize=0,
    ls="-",
)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.legend(loc='lower right')
#plt.legend(bbox_to_anchor=(0.75, 0.75))
plt.grid(linewidth="0.2")
plt.savefig("ФЧХ_03.svg")
plt.show()
plt.clf()
plt.cla()
plt.close()
