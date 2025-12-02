"""
Checks whether all columns from the import set table (CSV headers) exist on the target table.

Env vars required (same as run_import.py):
  SN_INSTANCE=https://<pdi>.service-now.com
  SN_USER=<user>
  SN_PASS=<pass>
  IMPORT_TABLE=dummy_data           # import set table
  TARGET_TABLE=incident             # target table

Usage:
  python servicenow/check_headers.py
"""

import os
import sys
from typing import Dict, Set

import requests
from dotenv import load_dotenv

load_dotenv()


REQUIRED_ENV = ["SN_INSTANCE", "SN_USER", "SN_PASS", "IMPORT_TABLE", "TARGET_TABLE"]

SKIP_ELEMENTS = {
    "sys_id",
    "sys_created_on",
    "sys_updated_on",
    "sys_created_by",
    "sys_updated_by",
    "sys_mod_count",
    "sys_import_set",
    "",
}


def require_env():
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}")


def sn_url(path: str) -> str:
    base = os.environ["SN_INSTANCE"].rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def auth():
    return (os.environ["SN_USER"], os.environ["SN_PASS"])


def api_get(path: str, params=None):
    resp = requests.get(sn_url(path), params=params, auth=auth())
    if not resp.ok:
        print(f"GET {resp.url} failed: {resp.status_code} {resp.reason}\n{resp.text}")
        resp.raise_for_status()
    data = resp.json()
    return data["result"] if isinstance(data, dict) and "result" in data else data


def dictionary_elements(table: str) -> Dict[str, str]:
    params = {
        "sysparm_query": f"name={table}",
        "sysparm_fields": "element,column_label,internal_type",
        "sysparm_limit": "1000",
    }
    result = api_get("/api/now/table/sys_dictionary", params=params)
    elements = {}
    for row in result:
        el = row.get("element", "")
        if el in SKIP_ELEMENTS:
            continue
        elements[el] = row.get("column_label") or el
    return elements


def main():
    require_env()
    import_table = os.environ["IMPORT_TABLE"]
    target_table = os.environ["TARGET_TABLE"]

    import_cols = dictionary_elements(import_table)
    target_cols = dictionary_elements(target_table)

    import_set: Set[str] = set(import_cols.keys())
    target_set: Set[str] = set(target_cols.keys())

    missing_on_target = sorted(import_set - target_set)
    extras_on_target = sorted(target_set - import_set)

    print(f"Import table: {import_table} ({len(import_set)} columns after skips)")
    print(f"Target table: {target_table} ({len(target_set)} columns after skips)")

    if missing_on_target:
        print("\nMissing on target (present in import, not in target):")
        for el in missing_on_target:
            print(f"- {el} ({import_cols.get(el)})")
    else:
        print("\nAll import columns exist on target.")

    # Optional: list target columns not in import
    if extras_on_target:
        print("\nTarget-only columns (not in import):")
        for el in extras_on_target:
            print(f"- {el} ({target_cols.get(el)})")


if __name__ == "__main__":
    main()
