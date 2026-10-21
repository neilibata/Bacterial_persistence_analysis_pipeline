# `figures/basemap/` — country outlines for the world maps

`naturalearth_lowres/` is Natural Earth 1:110m Admin-0 countries: 177 countries
with the columns `pop_est`, `continent`, `name`, `iso_a3` and `gdp_md_est`, in
EPSG:4326. It is the dataset geopandas shipped as `naturalearth_lowres` before
geopandas 1.0 removed it. Natural Earth is public domain
(https://www.naturalearthdata.com).

`figures/make_metadata_summary.py` reads this copy first, through
`_find_ne_path()`, when drawing Figure 1 and the metadata maps. The copy is
bundled so that the maps do not depend on a basemap being discoverable inside
an installed package — where it may or may not be present, depending on how
that package was installed.
