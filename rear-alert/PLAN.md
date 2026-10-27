# Rear Alert: 12-day plan (27 Oct to 7 Nov 2026, demo-ready by Diwali)

A rear-facing camera on a Honda Activa 3G records traffic behind the scooter. Python
detects and tracks vehicles, estimates how fast each one is closing in, and a trained
model decides when to light a **left** or **right** warning light on the handlebar.
Everything is built and tested on the laptop with recorded rides first; the handlebar
lights come last.

**Budget:** 12 days x 5 hours = 60 hours. The 6th hour is **only** for catching up, never
for new features. Diwali (Sun 8 Nov) is a day off.

---

## What gets built

```
rear camera (phone / dash cam)          rides recorded on SD card, 2-3 days of commutes
        |
        v
[1] detect.py    YOLO11 (fine-tuned on Indian traffic: autos, bikes, buses...)   <- AI #1
[2] track.py     ByteTrack: one ID per vehicle across frames
[3] geometry.py  ground-plane homography -> metres behind / to the side
                 bbox growth rate -> time-to-collision (TTC)
[4] risk.py      gradient-boosted model: "will this vehicle pass dangerously
                 close within 2 s?"  trained on your own rides                  <- AI #2
        |
        +--> [5] api.py        FastAPI: rides, detections, events, metrics, WebSocket replay
        |          |
        |          v
        |    [6] web/          React + three.js (react-three-fiber): video + 3D
        |                      bird's-eye view of vehicles around the scooter
        |
        +--> [7] alert_out.py  pyserial -> ESP32 (MicroPython) -> 2 LED sticks on the handlebar
```

All Python except the web dashboard (TypeScript/React). The ESP32 runs MicroPython,
so the hardware side is still Python too.

### Why React and not Angular
**React + `@react-three/fiber` + `@react-three/drei`.** It is the most used way to put
three.js in a web app. You write the 3D scene as components, drei gives you ready-made
camera controls, text and loaders, and Vite starts in seconds. Angular has `angular-three`,
but it has far fewer examples and answers online, and with 12 days that matters.

### Why there are two AI parts
1. **Fine-tuned detector.** Off-the-shelf YOLO (COCO) has no "auto-rickshaw" class and
   misses a lot of Indian two-wheelers. You label 300-400 frames from your own rides and
   fine-tune, then report mAP before and after.
2. **Learned alert model.** Instead of a hand-set "TTC < 2 s" rule, the labels come from
   the future: a track counts as a *close pass* if within the next 2 s it comes within 2 m
   or leaves the frame at the side. A model trained on features at time *t* (TTC, size,
   growth rate, lateral position, class, speed) predicts that. You compare it with the
   rule on **rides it has never seen**: precision, recall, false alarms per minute, and how
   many seconds of warning it gives.

Both give you numbers you can show, and that is what makes the AI part convincing.

---

## Rules (this is what makes the plan rigid)

1. **Fixed hours:** 5 blocks of 50 min + 10 min break. Same start time every day.
2. **Every day has a "Done when".** Don't start the next day's tasks until it is met.
3. **Commit and push at the end of every day**, even if it's broken (`wip:` prefix).
4. **Riding is outside the 5 hours.** Record **every** ride from 27 Oct to 31 Oct.
5. **Behind schedule?** Use the 6th hour. If you're still more than one day behind,
   cut in this order and don't feel bad about it:
   1. Live (non-replay) test, Day 11
   2. Metrics charts page, Day 9
   3. Virtual handlebar in the 3D scene
   4. Fine-tune shrinks to 150 labelled frames (the risk model is never cut)
6. **Feature freeze: end of Thu 5 Nov.** After that, only bug fixes, docs and the demo video.
7. **Safety:** never touch the laptop or phone while riding. The camera mount must not
   block the number plate, tail lamp or indicators.

---

## Day by day

### Day 1 - Tue 27 Oct: setup + first footage
- [ ] (1 h) Mount the rear camera. Today use your **phone** on a holder on the rear grab
      rail, facing backwards; don't wait for a dash cam. 1080p, 30 fps. Horizon in the upper
      third, both lanes behind you visible. Record the ride home.
- [ ] (1 h) Order hardware (list below) so it arrives by Day 10.
- [ ] (1 h) Python 3.11 venv: `ultralytics opencv-python pandas pyarrow scikit-learn
      lightgbm fastapi uvicorn pyserial`. Create the repo layout (below).
- [ ] (2 h) `detect.py`: video in, then annotated MP4 + one row per detection
      (`frame, t, class, conf, x1, y1, x2, y2`) as Parquet. Sample at 10 fps.

**Done when:** you can watch an annotated video of your own ride.

### Day 2 - Wed 28 Oct: tracking + calibration
- [ ] (1 h) `ingest.py`: copy rides into `data/rides/<date>_<n>/`, one `meta.json` each
      (date, duration, time of day, weather).
- [ ] (2 h) `track.py`: ByteTrack (`model.track(..., tracker="bytetrack.yaml")`).
      Output one table: `track_id, t, class, bbox`. Drop tracks shorter than 0.5 s.
- [ ] (1 h) Camera calibration with an OpenCV chessboard (print it, 15-20 photos), then
      save the intrinsics.
- [ ] (1 h) Ground-plane homography: park the scooter and lay out 4-6 tape marks on the
      ground behind it at measured distances. Click them in a frame and save `H`.

**Done when:** `tracks.parquet` exists for every ride so far, and `H` maps pixels to metres.

### Day 3 - Thu 29 Oct: geometry + rule-based alerts
- [ ] (2 h) `geometry.py`: bottom-centre of each box through `H` gives metres behind and
      metres sideways. Zones: LEFT / CENTRE / RIGHT.
- [ ] (1.5 h) TTC from scale change: slope of `log(box height)` over a 0.5 s window, so
      `TTC = 1 / slope`. Smooth it and clip silly values.
- [ ] (1 h) `rules.py` baseline: light LEFT/RIGHT if `TTC < 2.5 s` or distance `< 3 m`
      in that zone. Overlay distance, TTC and lights on the video.
- [ ] (0.5 h) Watch one ride with the overlay and write down 10 moments where it's wrong.

**Done when:** an overlay video shows the left/right lights switching on sensibly.

### Day 4 - Fri 30 Oct: build the dataset
- [ ] (0.5 h) Split rides: Days 1-3 = **train**, Days 4-5 = **test** (split by ride,
      never by frame, so nothing leaks).
- [ ] (0.5 h) Pull ~400 varied frames from the train rides (day/night, jams, highway).
- [ ] (0.5 h) Pre-label them with the pretrained model (export YOLO format).
- [ ] (3 h) Fix the labels in **Label Studio** (`pip install label-studio`). Classes:
      `car, bus, truck, two_wheeler, auto_rickshaw, bicycle, person`.
- [ ] (0.5 h) Label 80 frames from the **test** rides the same way.

**Done when:** 300+ train frames and 80 test frames are labelled and exported.

### Day 5 - Sat 31 Oct: fine-tune the detector (AI #1)
- [ ] (1 h) Upload the dataset to Kaggle or Colab (free GPU). Train `yolo11n` and `yolo11s`,
      about 50 epochs.
- [ ] (1 h) Evaluate on the 80 test frames: mAP50, mAP50-95, per-class recall,
      pretrained vs fine-tuned. Save the table to `results/detector.md`.
- [ ] (1 h) Export the best one to ONNX, check CPU FPS on the laptop.
- [ ] (2 h) Re-run detect, track and geometry on **all** rides with the new model
      (let the rest run overnight).

**Done when:** `results/detector.md` has the before/after table.

### Day 6 - Sun 1 Nov: the learned alert model (AI #2)
- [ ] (1.5 h) `events.py` auto-labels: a track at time *t* is positive if within the next
      2 s it gets closer than 2 m or leaves the frame at the left/right edge.
- [ ] (1 h) Features at time *t*: TTC, distance, sideways position, box area, growth
      rate, class, approach speed, track age.
- [ ] (1 h) Train LightGBM on train rides; tune the threshold on one validation ride.
- [ ] (1.5 h) Compare with `rules.py` on the **test rides**: precision, recall, false
      alarms per minute, median warning time. Save to `results/alerts.md` + a PR curve.

**Done when:** you have one table showing model vs rule on unseen rides.

### Day 7 - Mon 2 Nov: backend
- [ ] (3 h) `api.py` (FastAPI): `GET /rides`, `/rides/{id}` (meta + metrics),
      `/rides/{id}/frames?from=&to=` (tracks, metres, risk), `/rides/{id}/events`,
      video file streaming.
- [ ] (1.5 h) `WS /rides/{id}/replay`: pushes per-frame state at real-time speed,
      including the LEFT/RIGHT light state.
- [ ] (0.5 h) A pytest for each endpoint.

**Done when:** `uvicorn api:app` serves a ride and the WebSocket streams frames.

### Day 8 - Tue 3 Nov: dashboard part 1 (React + three.js)
- [ ] (0.5 h) `npm create vite@latest web -- --template react-ts`, add `three
      @react-three/fiber @react-three/drei`.
- [ ] (2.5 h) Bird's-eye 3D scene: road plane with lane lines, a simple scooter made of
      primitives, each tracked vehicle as a box with its class size at its metres
      position, coloured green/amber/red by risk.
- [ ] (2 h) Video panel next to it, synced to the WebSocket replay; play/pause/seek.

**Done when:** you press play and the video and the 3D scene move together.

### Day 9 - Wed 4 Nov: dashboard part 2
- [ ] (1.5 h) Two virtual handlebar lights in the scene that match the alert state.
- [ ] (1.5 h) Event timeline under the video; click an event to jump there.
- [ ] (1.5 h) Metrics page: detector table, model vs rule, PR curve.
- [ ] (0.5 h) Ride selector.

**Done when:** someone who has never seen the project can understand a ride in 30 seconds.

### Day 10 - Thu 5 Nov: handlebar lights (feature freeze tonight)
- [ ] (1 h) Flash MicroPython on the ESP32. Two WS2812 LED sticks via the `neopixel` module.
- [ ] (1 h) Serial protocol: one line per change, e.g. `L2 R0\n` (0 off, 1 amber,
      2 red flashing).
- [ ] (1.5 h) `alert_out.py`: replay a ride on the laptop and the real LEDs follow it.
      Measure delay (target under 100 ms).
- [ ] (1.5 h) Mount both sticks on the handlebar near the mirrors (cable ties, heat-shrink),
      powered by a USB power bank. Test visibility in daylight.

**Done when:** a recorded ride plays on the laptop and the physical lights blink in sync.

### Day 11 - Fri 6 Nov: end-to-end test + fixes
- [ ] (2 h) Live-ish test, **standing still**: phone streams with the IP Webcam app, laptop
      runs the pipeline, LEDs react while a friend rides past on a scooter in a quiet lane.
- [ ] (2 h) Fix the worst bugs from the whole week.
- [ ] (1 h) Write down FPS, delay, false alarms per minute and model size. These go
      in the README.

**Done when:** the full pipeline runs once without you touching code.

### Day 12 - Sat 7 Nov: package it
- [ ] (1.5 h) README: problem, architecture diagram, results tables, how to run, limits
      and next steps (Raspberry Pi 5 on the scooter, night riding, rain).
- [ ] (2 h) 2-minute demo video: ride clip, 3D dashboard, LEDs blinking, results table.
      **Blur faces and number plates** in anything you publish.
- [ ] (1 h) Clean the repo, pin versions, `make demo` / one-command start.
- [ ] (0.5 h) One-slide summary + the 3 numbers you'll lead with.

**Done when:** a stranger can clone the repo and run the demo from the README.

### Sun 8 Nov - Diwali. No work.

---

## Repo layout

```
rear-alert/
  pipeline/   detect.py track.py geometry.py rules.py events.py risk.py ingest.py
  api/        api.py tests/
  web/        React + Vite + react-three-fiber
  firmware/   main.py (MicroPython, ESP32)
  alert_out.py
  data/       (gitignored) rides/, labels/
  models/     (gitignored) weights
  results/    detector.md alerts.md plots
```

This project probably deserves its **own repository** rather than living next to the
folding-robot code. That way the GitHub link you share is only about this.

## Hardware to order on Day 1

| Item | Why |
|---|---|
| Phone holder for the rear grab rail (or a rear dash cam later) | Camera mount, start today |
| ESP32 DevKit (USB) | Drives the lights, runs MicroPython |
| 2 x WS2812B 8-LED sticks | Bright enough for daylight, one data pin, amber/red |
| Jumper wires, heat-shrink, cable ties, small project box | Handlebar mounting |
| USB power bank + 2 m USB cable | Powers the ESP32 on the scooter |
| A4 chessboard printout + measuring tape + masking tape | Calibration (Day 2) |

Robu.in, Robocraze and Amazon.in usually deliver in 2-5 days. Check stock before ordering.

## Footage checklist

- 1 hour of 1080p is about 6 GB, so carry a spare SD card or clear the phone every night.
- Aim for **4-6 hours** of riding over Days 1-5: commutes, a busy market road, at least
  one night ride and one main road.
- Keep the mount in the **same position** every day, or the homography from Day 2 breaks.
  Mark it with tape.
- Batch processing a few hours of video on a laptop CPU takes hours, so run it overnight.
