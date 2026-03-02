import os
import re
import pandas as pd

def extract_values_from_file(filepath: str) -> dict:
    values = {
        "file": filepath,
        "mag_obj": None,
        "mag_det": None,
        "SOD": None,
        "SDD": None,
    }

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    patterns = {
        "mag_obj": r"^\s*mag_obj\s*:\s*([-\d\.]+)",
        "mag_det": r"^\s*mag_det\s*:\s*([-\d\.]+)",
        "SOD": r"^\s*SOD\s*[:=]\s*([-\d\.]+)",
        "SDD": r"^\s*SDD\s*[:=]\s*([-\d\.]+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text, flags=re.MULTILINE)
        if m:
            values[key] = float(m.group(1))

    return values


def scrape_scan_settings(root_folder: str = ".") -> pd.DataFrame:
    rows = []
    for root, _, files in os.walk(root_folder):
        for name in files:
            if name.lower() == "scan settings.txt":
                path = os.path.join(root, name)
                rows.append(extract_values_from_file(path))

    df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    return df


df = scrape_scan_settings(".")
df['src_offset'] = df['SOD'] - df['mag_obj']
df['det_offset'] = df['SDD'] - df['src_offset'] - df['mag_det']
print(df)