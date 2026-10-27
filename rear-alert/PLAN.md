# Rear Alert: plan for Wed 28 Oct to Sat 7 Nov 2026 (demo-ready by Diwali)

A rear-facing camera on a Honda Activa 3G records traffic behind the scooter. Python
detects and tracks vehicles, estimates how fast each one is closing in, and a trained
model decides when to light a **left** or **right** warning light on the handlebar.
Everything is built and tested on the laptop with recorded rides first; the handlebar
lights come last.

## The time you actually have

Office is 10.5 h a day. Add 5 h of project, about 1 h of travel and 7.5 h of sleep, and
that is already 24 h, with no time left for eating, bathing or family. That doesn't last a
week, so this plan uses a budget you can keep up:

| Days | Count | Hours each | Total |
|---|---|---|---|
| Weekdays (Wed 28 - Fri 30 Oct, Mon 2 - Fri 6 Nov) | 8 | 2 | 16 h |
| Sat 31 Oct, Sun 1 Nov | 2 | 8 | 16 h |
| Sat 7 Nov (Diwali eve, packaging only) | 1 | 5 | 5 h |
| **Total** | | | **37 h** |

Weekday session: **21:00 - 23:00**, lights out by 23:30. Weekends: two 4-hour halves
with a long lunch break in between. This assumes Saturdays are off; if you work Saturdays,
see "If you fall behind".

That is enough for a **focused, working MVP** with two real AI components and numbers to
show. It is not enough for the earlier 60-hour version, so these are cut: chessboard
calibration, ground homography, a live test, a metrics web page and a virtual handlebar.

### Night is fine for coding
Only **recording** needs daylight. Record the **morning ride to office** every day
(phone on the rear grab rail, about 10 min to set up). All coding happens in the evening
on footage you already have.

---

## What gets built

```
rear camera (phone, morning commutes + one weekend ride)
        |
        v
[1] detect.py    YOLO11, fine-tuned on your own frames (autos, two-wheelers...)  <- AI #1
[2] track.py     ByteTrack: one ID per vehicle across frames
[3] geometry.py  distance from box height (pinhole: Z = f * H_real / h_pixels)
                 time-to-collision from how fast the box grows
                 LEFT / CENTRE / RIGHT zone from where the box is in the frame
[4] risk.py      LightGBM: "will this vehicle pass dangerously close within 2 s?"
                 trained on your rides, compared against a simple TTC rule        <- AI #2
        |
        +--> [5] api.py       FastAPI: rides, tracks, events + WebSocket replay
        |          |
        |          v
        |    [6] web/         React + three.js (react-three-fiber): video next to a
        |                     3D top-down view of the vehicles behind you
        |
        +--> [7] alert_out.py pyserial -> ESP32 (MicroPython) -> 2 LED sticks on the handlebar
```

All Python except the web dashboard. React with `@react-three/fiber` was chosen over
Angular because it has far more three.js examples and answers online.

---

## Rules

1. **Fixed sessions.** Weekdays 21:00-23:00, weekends 2 x 4 h. Stop on time and sleep.
2. **Every day has a "Done when".** Don't start the next day's work until it is met.
3. **Commit and push at the end of every session**, even if it's broken (`wip:` prefix).
4. **Record every morning ride** Wed 28 Oct to Fri 30 Oct, plus about 1 h of daytime
   riding on Sat 31 Oct (busy road + main road). Riding is not counted in the hours.
5. **Feature freeze: end of Thu 5 Nov.** After that, only fixes, docs and the demo video.
6. **Safety:** never touch the laptop or phone while riding. The mount must not block the
   number plate, tail lamp or indicators.

### If you fall behind
Use an extra hour **only on a weekend day**, never by cutting weekday sleep. If you're
still behind, cut in this order:
1. Handlebar mounting: show the LEDs on the desk instead
2. Event list in the dashboard (keep the video + 3D view)
3. Detector fine-tune: 100 frames instead of 200, or keep the pretrained model and
   list it as next step
4. **Never cut** the learned alert model (step 4). It is the main AI result.

If you work Saturdays, move Sat 31's work to Sun 1 and drop items 1 and 3.

---

## Day by day

### Wed 28 Oct (2 h): first footage + detection
- [ ] Morning: mount the phone on the rear grab rail facing back, 1080p 30 fps. Horizon
      in the upper third, both lanes behind you visible. Mark the mount position with
      tape so it is the same every day. Record the ride to office.
- [ ] Lunch break (15 min): **order the hardware** (list below) so it arrives by Thu 5 Nov.
- [ ] (0.5 h) Python 3.11 venv: `ultralytics opencv-python pandas pyarrow scikit-learn
      lightgbm fastapi uvicorn pyserial`. Create the repo layout (below).
- [ ] (1.5 h) `pipeline/detect.py`: video in, annotated MP4 out, plus a Parquet table
      `frame, t, class, conf, x1, y1, x2, y2`. Sample at 10 fps.

**Done when:** you can watch an annotated video of this morning's ride.

### Thu 29 Oct (2 h): tracking
- [ ] Morning: record the ride.
- [ ] (0.5 h) `pipeline/ingest.py`: rides into `data/rides/<date>_<n>/` with `meta.json`.
- [ ] (1.5 h) `pipeline/track.py`: ByteTrack (`model.track(..., tracker="bytetrack.yaml")`)
      giving `track_id, t, class, bbox`. Drop tracks shorter than 0.5 s. Run overnight on
      both rides.

**Done when:** `tracks.parquet` exists for both rides.

### Fri 30 Oct (2 h): time-to-collision + rule baseline
- [ ] Morning: record the ride.
- [ ] (1 h) `pipeline/geometry.py`: TTC from box growth (slope of `log(box height)` over
      0.5 s, `TTC = 1 / slope`, smoothed). Zone = LEFT / CENTRE / RIGHT from the box centre
      x-position, with boundaries tuned once by eye.
- [ ] (1 h) `pipeline/rules.py`: light LEFT/RIGHT when `TTC < 2.5 s` in that zone. Overlay
      TTC and the two lights on the video.

**Done when:** an overlay video shows the lights switching on at sensible moments.

### Sat 31 Oct (8 h): distance + build the dataset + fine-tune (AI #1)
- [ ] Morning: about 1 h daytime ride (busy market road + a main road) for extra footage.
- [ ] (1 h) Focal length: photograph a car at 3 m, 5 m and 10 m (measured with tape),
      solve `f = h_pixels * Z / H_real`. Add distance to `geometry.py`.
- [ ] (0.5 h) Split by ride: Wed-Fri = **train**, Saturday's ride = **test**. Never split
      frames of the same ride between train and test.
- [ ] (0.5 h) Pull ~200 varied frames from train rides, ~50 from the test ride. Pre-label
      them with the pretrained model.
- [ ] (3.5 h) Fix labels in Label Studio (`pip install label-studio`). Classes:
      `car, bus, truck, two_wheeler, auto_rickshaw, bicycle, person`.
- [ ] (1.5 h) Upload to Kaggle (free GPU), train `yolo11n` about 50 epochs.
- [ ] (1 h) Evaluate on the 50 test frames: mAP50 and per-class recall, pretrained vs
      fine-tuned. Save to `results/detector.md`. Start re-running the pipeline on all
      rides with the new model overnight.

**Done when:** `results/detector.md` has the before/after table.

### Sun 1 Nov (8 h): learned alert model (AI #2) + backend
- [ ] (1.5 h) `pipeline/events.py` auto-labels from the future: a track at time *t* is
      positive if in the next 2 s it gets closer than 2 m or leaves the frame at the
      left/right edge (it passed you).
- [ ] (1 h) Features at *t*: TTC, distance, zone, box area, growth rate, class, track age.
- [ ] (1.5 h) `pipeline/risk.py`: LightGBM on train rides, pick the threshold on one
      train ride held back for validation.
- [ ] (1 h) Model vs rule on the **test ride**: precision, recall, false alarms per
      minute, median warning time. Save to `results/alerts.md` + a PR curve PNG.
- [ ] (3 h) `api/api.py` (FastAPI): `GET /rides`, `/rides/{id}`, `/rides/{id}/frames`,
      `/rides/{id}/events`, video streaming. One pytest per endpoint.

**Done when:** `results/alerts.md` has the comparison and the API serves a ride.

### Mon 2 Nov (2 h): replay stream
- [ ] (1.5 h) `WS /rides/{id}/replay`: per-frame state (vehicles, distance, zone, risk,
      LEFT/RIGHT light) at real-time speed.
- [ ] (0.5 h) Test it with a small Python WebSocket client.

**Done when:** the WebSocket streams a whole ride.

### Tue 3 Nov (2 h): dashboard, 3D view
- [ ] (0.5 h) `npm create vite@latest web -- --template react-ts`, add `three
      @react-three/fiber @react-three/drei`.
- [ ] (1.5 h) 3D top-down scene: road plane with lane lines, scooter built from simple
      shapes, each vehicle a box at its distance/zone, coloured green/amber/red by risk.

**Done when:** vehicles move in the 3D view while a replay streams.

### Wed 4 Nov (2 h): dashboard, video + events
- [ ] (1.5 h) Video panel next to the 3D view, synced to the replay; play/pause.
- [ ] (0.5 h) Event list: click an event to jump to it.

**Done when:** video and 3D view move together.

### Thu 5 Nov (2 h): handlebar lights (feature freeze tonight)
- [ ] (0.5 h) Flash MicroPython on the ESP32, light two WS2812 sticks with `neopixel`.
- [ ] (0.5 h) Serial protocol: one line per change, e.g. `L2 R0\n` (0 off, 1 amber,
      2 red flashing).
- [ ] (1 h) `alert_out.py`: replay a ride on the laptop and the real LEDs follow it.

**Done when:** a recorded ride plays and the LEDs on the desk blink in sync.

### Fri 6 Nov (2 h, Dhanteras): mount + end-to-end
- [ ] (1 h) Mount the sticks on the handlebar near the mirrors (cable ties, heat-shrink),
      USB power bank. Check they're visible in daylight.
- [ ] (1 h) Run the whole thing once from a clean start; fix the worst bug.

**Done when:** one command starts the backend, dashboard and LEDs.

### Sat 7 Nov (5 h): package it
- [ ] (1.5 h) README: problem, architecture diagram, results tables, how to run, limits
      (night, rain, phone camera) and next steps (Raspberry Pi 5 on the scooter, live mode).
- [ ] (2 h) 2-minute demo video: ride clip, dashboard, LEDs blinking, results.
      **Blur faces and number plates** in anything you publish.
- [ ] (1 h) Clean the repo, pin versions, check the README steps work from a fresh clone.
- [ ] (0.5 h) Pick the 3 numbers you'll lead with (e.g. mAP gain, false alarms per
      minute vs rule, seconds of warning).

**Done when:** a stranger can clone the repo and run the demo from the README.

### Sun 8 Nov: Diwali. No work.

---

## Repo layout

```
rear-alert/
  pipeline/   ingest.py detect.py track.py geometry.py rules.py events.py risk.py
  api/        api.py tests/
  web/        React + Vite + react-three-fiber
  firmware/   main.py (MicroPython, ESP32)
  alert_out.py
  data/       (gitignored) rides/, labels/
  models/     (gitignored) weights
  results/    detector.md alerts.md plots
```

## Hardware (order Wed 28 Oct)

| Item | Why |
|---|---|
| Phone holder for the rear grab rail | Camera mount (use your own phone) |
| ESP32 DevKit (USB) | Drives the lights, runs MicroPython |
| 2 x WS2812B 8-LED sticks | Bright enough for daylight, one data pin, amber/red |
| Jumper wires, heat-shrink, cable ties | Handlebar mounting |
| USB power bank + 2 m USB cable | Powers the ESP32 on the scooter |
| Measuring tape | Focal length measurement (Sat 31) |

Robu.in, Robocraze and Amazon.in usually deliver in 2-5 days. Check stock before ordering.

## Footage checklist

- 1 hour of 1080p is about 6 GB; copy rides to the laptop every evening.
- Target: 3 morning rides + 1 h on Saturday, about 2-2.5 h of daytime footage.
- Keep the phone mount in the same taped position every day.
- Running the pipeline on hours of video takes a while on a laptop CPU, so start it
  before you sleep.
