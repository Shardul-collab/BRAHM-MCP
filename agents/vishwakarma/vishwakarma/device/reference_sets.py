# vishwakarma/device/reference_sets.py
#
# Literature parameter sets used to validate the device layer. Each value
# carries the page it was read from. These are for checking the pipeline
# against known physics, not a materials database -- for production runs the
# values should come from SHANI with primary citations.

SI_TABLE_URL = "https://www.ele.uri.edu/Courses/ele533/05/midterm/Physicaldata.html"
SI_TABLE_NOTE = ("secondary course table (Si, 300 K); the page cites no primary "
                 "source. Values match the classic Sze-era table; other tables "
                 "differ, e.g. Nv 1.04e19 vs ~1.8-3.1e19 cm^-3 elsewhere")

# Overrides for pipeline.material_from_dft: only what DFT does not provide.
SI_LITERATURE_OVERRIDES = {
    "band_gap_eV":          {"value": 1.12, "source": "literature", "ref": SI_TABLE_URL, "note": SI_TABLE_NOTE},
    "electron_affinity_eV": {"value": 4.05, "source": "literature", "ref": SI_TABLE_URL, "note": SI_TABLE_NOTE},
    "eps_r":                {"value": 11.8, "source": "literature", "ref": SI_TABLE_URL, "note": SI_TABLE_NOTE},
    "mu_n_cm2Vs":           {"value": 1500, "source": "literature", "ref": SI_TABLE_URL, "note": "intrinsic (lattice-limited) mobility; " + SI_TABLE_NOTE},
    "mu_p_cm2Vs":           {"value": 450,  "source": "literature", "ref": SI_TABLE_URL, "note": "intrinsic (lattice-limited) mobility; " + SI_TABLE_NOTE},
    # The table gives no lifetimes; these are process-dependent anyway.
    "srh_lifetime_n_s":     {"value": 1e-6, "source": "assumed", "note": "placeholder; set from measurement or SHANI"},
    "srh_lifetime_p_s":     {"value": 1e-6, "source": "assumed", "note": "placeholder; set from measurement or SHANI"},
}

# Same table, including the band-edge densities, for an all-literature control run.
SI_LITERATURE_FULL = dict(SI_LITERATURE_OVERRIDES)
SI_LITERATURE_FULL.update({
    "Nc_cm3": {"value": 2.8e19,  "source": "literature", "ref": SI_TABLE_URL, "note": SI_TABLE_NOTE},
    "Nv_cm3": {"value": 1.04e19, "source": "literature", "ref": SI_TABLE_URL, "note": SI_TABLE_NOTE},
})
