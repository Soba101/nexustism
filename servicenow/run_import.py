"""
Automates the import-set → transform workflow for data/dummy_data.csv.

Steps:
1) Upload CSV to the import set table.
2) Compare import-table columns to target table; create missing target fields.
3) Trigger the transform map for the import set run.

Set these environment variables before running:
  SN_INSTANCE=https://dev191207.service-now.com
  SN_USER=admin
  SN_PASS=purqAA9@J^0O
  IMPORT_TABLE=dummy data upload
  TARGET_TABLE=incident
  TRANSFORM_MAP_SYS_ID=dummy to PDI
  CSV_PATH=data/dummy_data.csv

Usage:
  python servicenow/run_import.py
"""

import json
import os
import sys
from pathlib import Path
import csv
import re
import requests
from dotenv import load_dotenv
load_dotenv()

REQUIRED_ENV = [
    "SN_INSTANCE",
    "SN_USER",
    "SN_PASS",
    "IMPORT_TABLE",
    "TARGET_TABLE",
    "TRANSFORM_MAP_SYS_ID",
    "CSV_PATH",
]

# Field names to skip when diffing dictionaries.
SKIP_ELEMENTS = {
    "sys_id",
    "sys_created_on",
    "sys_updated_on",
    "sys_created_by",
    "sys_updated_by",
    "sys_mod_count",
}

# Optional overrides for target field creation keyed by lowercased element.
FIELD_TYPE_OVERRIDES = {
    "manday_effort_hrs": {"internal_type": "decimal"},
    "created": {"internal_type": "glide_date_time"},
    "closed": {"internal_type": "glide_date_time"},
    "urgency": {"internal_type": "string"},
    "impact": {"internal_type": "string"},
    "priority": {"internal_type": "string"},
    "state": {"internal_type": "string"},
    "company": {"internal_type": "reference", "reference": "cmn_company"},
    "assignment_group": {"internal_type": "reference", "reference": "sys_user_group"},
    "assigned_to": {"internal_type": "reference", "reference": "sys_user"},
    "opened_by": {"internal_type": "reference", "reference": "sys_user"},
    "closed_by": {"internal_type": "reference", "reference": "sys_user"},
    "service": {"internal_type": "reference", "reference": "cmdb_ci_service"},
    "service_offering": {"internal_type": "reference", "reference": "service_offering"},
}


def require_env():
    missing = [v for v in REQUIRED_ENV if not os.environ.get(v)]
    if missing:
        sys.exit(f"Missing environment variables: {', '.join(missing)}")


def sn_url(path: str) -> str:
    base = os.environ["SN_INSTANCE"].rstrip("/")
    return f"{base}/{path.lstrip('/')}"


def api_get(path: str, params=None):
    resp = requests.get(sn_url(path), params=params, auth=auth())
    if not resp.ok:
        print(f"GET {resp.url} failed: {resp.status_code} {resp.reason}\n{resp.text}")
        resp.raise_for_status()
    data = resp.json()
    return data["result"] if isinstance(data, dict) and "result" in data else data


def api_post(path: str, json_body=None, files=None, headers=None, params=None):
    resp = requests.post(
        sn_url(path),
        json=json_body,
        files=files,
        headers=headers,
        params=params,
        auth=auth(),
    )
    if not resp.ok:
        print(f"POST {resp.url} failed: {resp.status_code} {resp.reason}\n{resp.text}")
        resp.raise_for_status()
    data = resp.json()
    return data["result"] if isinstance(data, dict) and "result" in data else data


def auth():
    return (os.environ["SN_USER"], os.environ["SN_PASS"])


def upload_csv(csv_path: Path, import_table: str) -> str:
    if not csv_path.exists():
        sys.exit(f"CSV not found: {csv_path}")
    if import_table.upper().startswith("ISET"):
        sys.exit(
            f"IMPORT_TABLE looks like an import set run number ({import_table}). "
            "Use the import set TABLE name (e.g., u_dummy_import), not an import set run ID."
        )
    files = {"file": (csv_path.name, csv_path.read_bytes())}
    print(f"Uploading {csv_path} to import table {import_table} ...")
    try:
        result = api_post(f"/api/now/import/{import_table}", files=files)
        import_set = result.get("import_set") or result.get("sys_import_set")
        if not import_set:
            sys.exit(f"Could not find import_set in response: {json.dumps(result, indent=2)}")
        print(f"Upload complete. import_set sys_id: {import_set}")
        return import_set
    except requests.HTTPError as e:
        # Fallback: some instances reject multipart; send JSON rows instead.
        if e.response is None or e.response.status_code != 415:
            raise
        print("Multipart upload not supported; falling back to JSON row import from CSV.")
        return upload_csv_as_json_rows(csv_path, import_table)


def extract_import_set(result):
    if isinstance(result, dict):
        return result.get("import_set") or result.get("sys_import_set")
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first.get("import_set") or first.get("sys_import_set")
    return None


def normalize_field(name: str) -> str:
    """Normalize CSV header to import-set field name (u_ prefix, snake_case)."""
    norm = re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_").lower()
    if not norm.startswith("u_"):
        norm = f"u_{norm}"
    return norm


def upload_csv_as_json_rows(csv_path: Path, import_table: str) -> str:
    """Read CSV locally and POST rows as JSON to the import set API."""
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        sys.exit("CSV contains no data rows.")

    import_set_id = None
    for idx, row in enumerate(rows, start=1):
        body = {normalize_field(k): v for k, v in row.items()}
        if import_set_id:
            body["sys_import_set"] = import_set_id
        result = api_post(
            f"/api/now/import/{import_table}",
            json_body=body,
            headers={"Content-Type": "application/json"},
        )
        import_set_id = import_set_id or extract_import_set(result)
        if idx % 25 == 0 or idx == len(rows):
            print(f"Uploaded {idx}/{len(rows)} rows (import_set={import_set_id})")

    if not import_set_id:
        print("Import set sys_id not returned; querying latest import set for table.")
        import_set_id = latest_import_set_for_table(import_table)
        if not import_set_id:
            sys.exit("Could not determine import set sys_id after upload.")
    print(f"Upload complete via JSON rows. import_set sys_id: {import_set_id}")
    return import_set_id


def latest_import_set_for_table(import_table: str) -> str:
    """Fetch the most recent import set run for the given import table."""
    params = {
        "sysparm_query": f"table_name={import_table}^ORDERBYDESCsys_created_on",
        "sysparm_fields": "sys_id,table_name,sys_created_on,state",
        "sysparm_limit": "1",
    }
    result = api_get("/api/now/table/sys_import_set", params=params)
    if isinstance(result, list) and result:
        return result[0].get("sys_id")
    if isinstance(result, dict):
        return result.get("sys_id")
    return None


def fetch_dictionary(table: str):
    params = {
        "sysparm_query": f"name={table}",
        "sysparm_fields": "element,column_label,internal_type,reference",
        "sysparm_limit": "1000",
    }
    items = api_get("/api/now/table/sys_dictionary", params=params)
    return {i["element"]: i for i in items}


def make_label(element: str) -> str:
    return element.replace("_", " ").title()


def ensure_fields(import_dict: dict, target_dict: dict, target_table: str):
    created = []
    for element, meta in import_dict.items():
        if element in SKIP_ELEMENTS or element in target_dict:
            continue
        element_for_target = element if element.startswith("u_") else f"u_{element}"
        override = FIELD_TYPE_OVERRIDES.get(element.lower(), {})
        payload = {
            "name": target_table,
            "element": element_for_target,
            "column_label": meta.get("column_label") or make_label(element_for_target),
            "internal_type": override.get("internal_type", "string"),
        }
        if "reference" in override:
            payload["reference"] = override["reference"]
        if payload["internal_type"] == "string":
            payload["max_length"] = "255"
        print(f"Creating missing field on {target_table}: {payload}")
        created.append(api_post("/api/now/table/sys_dictionary", json_body=payload))
    if not created:
        print("No new fields needed on target table.")
    else:
        print(f"Created {len(created)} fields on {target_table}.")
    return created


def run_transform(import_table: str, import_set_id: str, transform_map_sys_id: str):
    payload = {
        "import_set": import_set_id,
        "transform_map": transform_map_sys_id,
    }
    print("Triggering transform ...")
    result = api_post(f"/api/now/import/{import_table}/transform", json_body=payload)
    print("Transform triggered.")
    return result


def summarize_run(import_set_id: str):
    params = {
        "sysparm_query": f"sys_id={import_set_id}",
        "sysparm_fields": "sys_id,state,inserted,updated,ignored,errors",
    }
    runs = api_get("/api/now/table/sys_import_set_run", params=params)
    if runs:
        print("Run summary:", json.dumps(runs[0], indent=2))
    else:
        print("No run summary found for import_set:", import_set_id)


def main():
    require_env()
    import_table = os.environ["IMPORT_TABLE"]
    target_table = os.environ["TARGET_TABLE"]
    transform_map_sys_id = os.environ["TRANSFORM_MAP_SYS_ID"]
    csv_path = Path(os.environ["CSV_PATH"])
    import_set_id = os.environ.get("IMPORT_SET_ID")

    if import_set_id:
        print(f"Using existing import_set: {import_set_id}")
    else:
        try:
            import_set_id = upload_csv(csv_path, import_table)
        except SystemExit:
            # Upload not available; fall back to latest import set for the table.
            print("Upload skipped; falling back to latest import set for table.")
            import_set_id = latest_import_set_for_table(import_table)
            if not import_set_id:
                sys.exit("No import set id available. Set IMPORT_SET_ID or enable upload.")

    import_dict = fetch_dictionary(import_table)
    target_dict = fetch_dictionary(target_table)
    ensure_fields(import_dict, target_dict, target_table)

    run_transform(import_table, import_set_id, transform_map_sys_id)
    summarize_run(import_set_id)


if __name__ == "__main__":
    main()
