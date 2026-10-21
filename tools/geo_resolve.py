"""Resolve a free-text geography string to one of the pipeline's 44 countries.

Shared by `verify_country_against_ena.py` and `verify_country_via_biosample.py`
so the two sources are judged by identical rules; when they disagreed earlier it
was worth knowing that the disagreement was in the data, not in two slightly
different parsers.

Both ENA's `country` and BioSample's `geo_loc_name` are free text. Observed forms,
restricted to runs this cohort actually uses:

    United States of America        bare country
    USA: Ann Arbor                  INSDC Country:Region
    Canada:Toronto                  ... without the space
    Ireland:Unspecified;Ireland     doubled, semicolon-separated
    ES:MD, JP:Tokyo                 ISO 3166-1 alpha-2 prefix
    usa:ca:san diego                lowercase, multi-level
    Xi'an, Weinan, Baoji            a bare city, no country at all
    missing, not applicable         explicit absence

Anything that does not resolve is None, meaning "unverifiable", never "conflicts".
A submitter writing a city they did not qualify is not evidence that our country
is wrong -- treating it as one produced 171 spurious disagreements against a
correct `China` in the first pass.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from persistence import config  # noqa: E402
from persistence.common import normalise_country  # noqa: E402

KNOWN = set(config.COUNTRY_TO_CONTINENT)

ABSENT = {
    "", "na", "n/a", "nan", "none", "null", "missing", "unknown",
    "not applicable", "not collected", "not provided", "not determined",
    "restricted access", "unspecified", "labcontrol test",
}

# ISO 3166-1 alpha-2 for the countries in this cohort. Only these are needed:
# an alpha-2 code for a country we have no samples from is still unverifiable.
ALPHA2 = {
    "AR": "Argentina", "AT": "Austria", "BD": "Bangladesh", "BE": "Belgium",
    "BF": "Burkina Faso", "BR": "Brazil", "CA": "Canada", "CD": "Democratic Republic of the Congo",
    "CH": "Switzerland", "CM": "Cameroon", "CN": "China", "CO": "Colombia",
    "DE": "Germany", "DK": "Denmark", "ES": "Spain", "ET": "Ethiopia",
    "FI": "Finland", "FJ": "Fiji", "FR": "France", "GB": "United Kingdom",
    "GH": "Ghana", "GW": "Guinea-Bissau", "HN": "Honduras", "ID": "Indonesia",
    "IL": "Israel", "IS": "Iceland", "IT": "Italy", "JP": "Japan",
    "KR": "South Korea", "KZ": "Kazakhstan", "LR": "Liberia", "MG": "Madagascar",
    "MN": "Mongolia", "NL": "Netherlands", "NP": "Nepal", "NZ": "New Zealand",
    "PE": "Peru", "PH": "Philippines", "RU": "Russia", "SE": "Sweden",
    "SV": "El Salvador", "TH": "Thailand", "TZ": "Tanzania", "US": "United States of America",
    "ZA": "South Africa",
}

# Cities that appear unqualified in ENA. All six are in Shaanxi (PRJEB29127,
# ZhuF), where the submitter put the city in the country field. Deliberately
# narrow: a general gazetteer would start guessing, and a wrong guess here
# manufactures a disagreement against metadata that is correct.
CITIES = {
    "xi'an": "China", "xian": "China", "lintong": "China", "weinan": "China",
    "hanzhong": "China", "xianyang": "China", "baoji": "China",
}


def resolve(value) -> str | None:
    """Free-text geography -> a country name the pipeline knows, or None."""
    if value is None or not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw or raw.lower() in ABSENT:
        return None

    # `Ireland:Unspecified;Ireland` -> take the first field of the first clause
    head = re.split(r"[;,]", raw)[0]
    parts = [p.strip() for p in head.split(":") if p.strip()]
    if not parts:
        return None
    first = parts[0]
    if first.lower() in ABSENT:
        return None

    # `normalise_country` is case-sensitive: it resolves 'USA' but returns 'usa'
    # unchanged, and 'china' unchanged. Submitters write both, so try the
    # plausible casings rather than only what was typed.
    for candidate in (first, first.upper(), first.title(),
                      first.capitalize(), first.lower()):
        name = normalise_country(candidate)
        if name in KNOWN:
            return name

    if len(first) == 2 and first.upper() in ALPHA2:
        name = ALPHA2[first.upper()]
        return name if name in KNOWN else None

    city = CITIES.get(first.lower())
    if city and city in KNOWN:
        return city

    return None
