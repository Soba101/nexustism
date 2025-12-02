# Phase 1: Upload CSV and Ensure Target Has Matching Fields

This phase uploads `data/dummy_data.csv` to your import set table and auto-creates any missing columns on the target table so every header has a destination.

## 1) Upload CSV into import set
```bash
curl -u "$SN_USER:$SN_PASS" -F "file=@$CSV_PATH" \
  "$SN_INSTANCE/api/now/import/$IMPORT_TABLE"
# capture import_set (sys_import_set.sys_id) from the response → set IMPORT_SET_ID
```

## 2) Discover headers in the import set table
```bash
curl -u "$SN_USER:$SN_PASS" \
  "$SN_INSTANCE/api/now/table/sys_dictionary?sysparm_query=name=$IMPORT_TABLE&sysparm_fields=element,column_label,internal_type"
# ignore system fields (sys_id, sys_created_on, etc.)
```

## 3) Discover existing fields on the target table
```bash
curl -u "$SN_USER:$SN_PASS" \
  "$SN_INSTANCE/api/now/table/sys_dictionary?sysparm_query=name=$TARGET_TABLE&sysparm_fields=element,column_label,internal_type"
```

## 4) Create missing fields on target (string by default)
- For each import element not present on target, POST to `sys_dictionary`:
```bash
curl -u "$SN_USER:$SN_PASS" -H "Content-Type: application/json" -d "{
  \"name\": \"$TARGET_TABLE\",
  \"element\": \"u_my_field\",
  \"column_label\": \"My Field\",
  \"internal_type\": \"string\",
  \"max_length\": \"255\"
}" "$SN_INSTANCE/api/now/table/sys_dictionary"
```
- For references, set `"internal_type": "reference"` and add `"reference": "<table>"` (e.g., `cmn_company`, `sys_user`, `sys_user_group`, `service_offering`).
- For decimals/ints/datetimes, use `decimal`, `integer`, `glide_date_time` as `internal_type`.

## Optional: scripted diff/creation helper (Python + requests)
```python
import os, requests
base = os.environ["SN_INSTANCE"]
auth = (os.environ["SN_USER"], os.environ["SN_PASS"])
import_table = os.environ["IMPORT_TABLE"]
target_table = os.environ["TARGET_TABLE"]

def cols(table):
    r = requests.get(f"{base}/api/now/table/sys_dictionary",
                     params={"sysparm_query": f"name={table}",
                             "sysparm_fields": "element,column_label,internal_type",
                             "sysparm_limit": "1000"},
                     auth=auth)
    r.raise_for_status()
    return {c["element"]: c for c in r.json()["result"]}

src = cols(import_table)
dst = cols(target_table)
skip = {"sys_id","sys_created_on","sys_updated_on","sys_created_by","sys_updated_by","sys_mod_count"}

for el, meta in src.items():
    if el in skip or el in dst:
        continue
    payload = {
        "name": target_table,
        "element": f"u_{el}" if not el.startswith("u_") else el,
        "column_label": meta.get("column_label") or el.replace("_"," ").title(),
        "internal_type": "string",
        "max_length": "255",
    }
    resp = requests.post(f"{base}/api/now/table/sys_dictionary", json=payload, auth=auth)
    resp.raise_for_status()
    print("Created", payload["element"])
```
- Adjust `internal_type` per column if you know the right type ahead of time (e.g., map `Closed`/`Created` to `glide_date_time`, `Manday Effort (hrs)` to `decimal`, references to their tables).
