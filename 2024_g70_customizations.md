# 2024 Genesis G70 Customizations

This document summarizes how `GENESIS_G70_2024` differs from earlier Genesis G70 definitions in this branch.

## Model-to-model differences

| Area | G70 2018 (`GENESIS_G70`) | G70 2019-23 (`GENESIS_G70_2020`) | G70 2024 (`GENESIS_G70_2024`) |
| --- | --- | --- | --- |
| Platform flags | `LEGACY` | `MANDO_RADAR` | `CHECKSUM_CRC8 | CAMERA_SCC` |
| Lateral baseline specs | `mass=1640`, `wheelbase=2.84`, `steerRatio=13.56` | Reuses 2018 specs | `mass=1769`, `wheelbase=2.83`, `steerRatio=12.9` |
| Harness | `hyundai_f` | `hyundai_f` (2019-21), `hyundai_l` (2022-23) | `hyundai_l` |
| SCC architecture | Radar-centric older stack | Radar-centric Mando SCC | Camera-SCC flag path with CRC8 checksumming |
| Fingerprint examples (camera/radar FW) | `95740-G9000` / `96400-G9100` | `95740-G9000` or `99211-G9000` / `96400-G9100`, `99110-G9300`, `96400-G9000` | `99211-G9500` / `99110-G9600` |
| LKAS11 handling | Existing legacy handling | Included in LKAS11 special-case set | Explicitly added to LKAS11 special-case set |
| Torque override | No G70-specific override entry in this branch baseline | No G70-specific override entry in this branch baseline | Added: `"GENESIS_G70_2024" = [2.7, 2.7, 0.11]` |

## Tuning locations for 2024 G70

- Platform definition and base specs: `opendbc_repo/opendbc/car/hyundai/values.py`
- ECU fingerprints (camera/radar IDs): `opendbc_repo/opendbc/car/hyundai/fingerprints.py`
- LKAS message behavior list membership: `opendbc_repo/opendbc/car/hyundai/hyundaican.py`
- Lateral torque tuning tuple: `opendbc_repo/opendbc/car/torque_data/override.toml`

## Documentation status note

The generated car docs currently list G70 through 2022-23 (plus `G70 Non-SCC 2021` in `opendbc_repo/docs/CARS.md`).
`GENESIS_G70_2024` is present in code paths above but not yet reflected in the generated support tables.
