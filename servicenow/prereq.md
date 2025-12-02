# ServiceNow Import/Transform Prereqs

Goal: verify you can reach your PDI and have everything needed to run the import-set → transform flow for `data/dummy_data.csv`.

- Set environment variables (adjust names):
  - `export SN_INSTANCE="https://<your_pdi>.service-now.com"`
  - `export SN_USER="<admin_or_api_user>"`
  - `export SN_PASS="<password_or_token>"`
  - `export IMPORT_TABLE="u_dummy_import"` # your import set table
  - `export TARGET_TABLE="incident"` # or a custom table
  - `export TRANSFORM_MAP_SYS_ID="<transform_map_sys_id>"` # map from import table → target
  - `export CSV_PATH="data/dummy_data.csv"`
- Network/auth check (Table API reachable):
  - `curl -u "$SN_USER:$SN_PASS" "$SN_INSTANCE/api/now/table/sys_user?sysparm_limit=1"`
- Import table exists:
  - `curl -u "$SN_USER:$SN_PASS" "$SN_INSTANCE/api/now/table/sys_dictionary?sysparm_query=name=$IMPORT_TABLE&sysparm_limit=1"`
- Transform map exists and points to the right tables:
  - `curl -u "$SN_USER:$SN_PASS" "$SN_INSTANCE/api/now/table/sys_transform_map/$TRANSFORM_MAP_SYS_ID"`
- Confirm your account roles:
  - Needs `admin` (or `import_transformer` + `personalize_dictionary`) to create fields and run transforms.
- CSV format note:
  - Has headers on row 1; date format `dd-MM-yy HH:mm`; references (users/groups/companies/services) should match names or you will need lookup scripts in the transform.
