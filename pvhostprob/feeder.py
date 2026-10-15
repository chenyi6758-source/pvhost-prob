"""Synthetic radial LV feeder builder.

A self-contained test feeder so examples and tests never depend on external
network data: a radial 0.4 kV feeder with an MV/LV substation (slack) at bus 0
and identical line segments plus one load per downstream bus. Voltage drop
under load and voltage rise under PV injection both grow with electrical
distance from the substation — the qualitative behaviour real LV feeders show
and the reason locational hosting capacity varies along the feeder.
"""

import pandapower as pp


def build_lv_feeder(
    n_buses=12,
    load_kw=3.0,
    r_ohm_per_km=0.40,
    x_ohm_per_km=0.10,
    seg_km=0.05,
    vn_kv=0.4,
    max_i_ka=0.20,
):
    """Build a radial LV feeder pandapower net.

    Parameters
    ----------
    n_buses : int
        Total buses including the substation bus (bus 0).
    load_kw : float
        Identical active load (kW, unity power factor) at every bus except bus 0.
    r_ohm_per_km, x_ohm_per_km : float
        Series impedance of the feeder cable.
    seg_km : float
        Length of each bus-to-bus segment.
    vn_kv : float
        Nominal voltage.
    max_i_ka : float
        Thermal rating used for line loading_percent.

    Returns
    -------
    pandapowerNet with ``bus`` names LV0..LV{n_buses-1}.
    """
    if n_buses < 2:
        raise ValueError("n_buses must be >= 2")
    net = pp.create_empty_network(name="synthetic LV feeder")
    buses = [
        pp.create_bus(net, vn_kv=vn_kv, name=f"LV{i}") for i in range(n_buses)
    ]
    pp.create_ext_grid(net, bus=buses[0], vm_pu=1.0, name="MV/LV substation")
    for i in range(n_buses - 1):
        pp.create_line_from_parameters(
            net,
            from_bus=buses[i],
            to_bus=buses[i + 1],
            length_km=seg_km,
            r_ohm_per_km=r_ohm_per_km,
            x_ohm_per_km=x_ohm_per_km,
            c_nf_per_km=0.0,
            max_i_ka=max_i_ka,
            name=f"seg{i}",
        )
        pp.create_load(
            net,
            bus=buses[i + 1],
            p_mw=load_kw / 1000.0,
            q_mvar=0.0,
            name=f"load{i + 1}",
        )
    return net
