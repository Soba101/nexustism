# Phase 2: Run Transform Map and Validate Results

After the target table has columns for all headers and the CSV is uploaded, run the transform and confirm records.

## 1) Trigger the transform map
- If you have a specific import set run ID (`IMPORT_SET_ID`) from Phase 1:
```bash
curl -u "$SN_USER:$SN_PASS" -H "Content-Type: application/json" -d "{
  \"import_set\": \"$IMPORT_SET_ID\",
  \"transform_map\": \"$TRANSFORM_MAP_SYS_ID\"
}" "$SN_INSTANCE/api/now/import/$IMPORT_TABLE/transform"
```
- If your Data Source is configured to auto-transform, you can also use:
  - `.../api/now/import/$IMPORT_TABLE/transform?sysparm_transform_after_load=true`

## 2) Monitor run status
```bash
curl -u "$SN_USER:$SN_PASS" \
  "$SN_INSTANCE/api/now/table/sys_import_set_run?sysparm_query=sys_id=$IMPORT_SET_ID&sysparm_fields=sys_id,state,inserted,updated,ignored,errors"
```
- Inspect per-row results:
```bash
curl -u "$SN_USER:$SN_PASS" \
  "$SN_INSTANCE/api/now/table/sys_import_set_row?sysparm_query=import_set=$IMPORT_SET_ID&sysparm_fields=sys_id,state,error_message,source&sysparm_limit=20"
```

## 3) Spot-check target records
- Query a few transformed records (e.g., by `number` if coalesced):
```bash
curl -u "$SN_USER:$SN_PASS" \
  "$SN_INSTANCE/api/now/table/$TARGET_TABLE?sysparm_query=numberSTARTSWITHINC900000&sysparm_fields=sys_id,number,short_description,opened_by,assignment_group,state"
```

## 4) Adjust and re-run if needed
- Fix field maps or scripts (choice translations, reference lookups, date parsing).
- Re-run: re-upload CSV (or reuse the same import set) and trigger the transform again.

## Quick mapping reminders for `dummy_data.csv`
- Coalesce on `number` to avoid duplicates.
- Map dates (`Created`, `Closed`) to `glide_date_time` with parsing scripts (`dd-MM-yy HH:mm`).
- References: `Opened by/Closed by/Assigned to` → `sys_user`; `Assignment group` → `sys_user_group`; `Company` → `cmn_company`; `Service`/`Service offering` → service tables.
- Choices: `Urgency`, `Impact`, `Priority`, `State`, `Category`, `Subcategory`, `Resolution code` must match choice names or be translated in field-map scripts.
- Freeform extras: map AMS fields and comments/work notes to custom `u_` fields if you do not want to overwrite core fields.
