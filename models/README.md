# BFMC AI Models

Place `.pt` (YOLOv8 / ultralytics) files here.
**Lane detection and BEV are handled by pure CV — not AI models.**

## Files in this folder

| File | Source | Detects | Notes |
|---|---|---|---|
| `traffic_light.pt` | `best_traffic_med_yolo_v8.pt` | Red / Yellow / Green | Default — 50 MB |
| `traffic_light_small.pt` | `best_traffic_small_yolo.pt` | Red / Yellow / Green | Balanced — 22 MB |
| `traffic_light_nano.pt` | `best_traffic_nano_yolo.pt` | Red / Yellow / Green | Fastest on Pi — 6 MB |
| `road_sign.pt` | `best.pt` | Highway entry/exit, Stop, Zebra, Parking, One-Way | Default (v1) — 6 MB |
| `road_sign_v2.pt` | `last.pt` | Same as above | Alternative (v2) — 6 MB |

## Obstacle detector
No dedicated obstacle model yet. Drop an `obstacle.pt` into this folder and it will be enabled automatically.

## Lane divider (optional)
Drop a `lane_divider.pt` here to enable AI-assisted divider detection as a supplement to CV lane tracking.

## Sign class indices (`road_sign.pt` / `road_sign_v2.pt`)

```
0: HIGHWAY_ENTRY
1: ZEBRA_CROSSING
2: STOP_SIGN
3: HIGHWAY_EXIT
4: PARKING
5: ONE_WAY
```

## Traffic light class names

```
traffic_light   (whole fixture)
red
yellow
green
```

## Switching models

Edit `config.py` Section 10 and comment/uncomment the relevant line.
