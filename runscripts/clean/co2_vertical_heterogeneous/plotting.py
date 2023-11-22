
############
# Plotting #
############
# Comparison vs. Peaceman for the first, third and last layer. Only first ensemble
# member.
fig_1 = plt.figure(1)
fig_2 = plt.figure(2)
for i, color in zip([0, 2, 4], plt.cm.rainbow(np.linspace(0, 1, 3))):
    pressures_member = pressures[0, ..., i]
    bhp_member = bhps[0, ..., i]
    injection_rate_per_second_per_cell_member = injection_rate_per_second_per_cell[
        0, ..., i
    ]
    WI_data_member = WI_data[0, ..., i]
    WI_analytical_member = WI_analytical[0, ..., i]

    # Plot analytical vs. data WI in the upper layer.
    plt.figure(1)
    plt.scatter(
        timesteps, WI_data_member, label=f"Layer {i} data", color=color, linestyle="-"
    )
    plt.plot(
        timesteps,
        WI_analytical_member,
        label=f"Layer {i} Peaceman",
        color=color,
        linestyle="--",
    )

    # Plot bhp predicted by Peaceman and data vs actual bhp in the upper layer.
    # NOTE: bhp predicted by data and actual bhp should be identical.
    bhp_data: np.ndarray = (
        injection_rate_per_second_per_cell_member / WI_data_member + pressures_member
    )
    bhp_analytical: np.ndarray = (
        injection_rate_per_second_per_cell_member / WI_analytical_member
        + pressures_member
    )
    plt.figure(2)
    plt.scatter(
        timesteps,
        bhp_data,
        label=rf"Layer {i}: calculated from data $WI$",
        color=color,
    )
    plt.plot(
        timesteps,
        bhp_analytical,
        label=rf"Layer {i}: calculated from Peaceman $WI$",
        color=color,
        linestyle="--",
    )
    plt.plot(
        timesteps,
        bhp_member,
        label=rf"Layer {i}: data",
        color=color,
        linestyle="-",
    )

plt.figure(1)
plt.legend()
plt.xlabel(r"$t\,[d]$")
plt.ylabel(r"$WI\,[m^4\cdot s/kg]$")
plt.title(r"$WI$")
plotting.save_fig_and_data(
    fig_1,
    pathlib.Path(ensemble_dirname) / "WI_data_vs_Peaceman.png",
)


plt.figure(2)
plt.legend()
plt.xlabel(r"$t\,[d]$")
plt.ylabel(r"$p\,[Pa]$")
plt.title(r"$p_{bhp}$ for various layers")
plotting.save_fig_and_data(
    fig_2, pathlib.Path(ensemble_dirname) / "pbh_data_vs_Peaceman.png"
)
plt.show()
