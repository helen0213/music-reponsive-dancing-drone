# Music-Responsive Dancing Drone

A Duckietown DD24-B Duckiedrone that dances to music. Audio is analyzed **offline**
with Librosa, turned into a timed choreography schedule, and executed through
**ROS 2** (distro `ente`, base image `dt-ros2-commons`), first in the Duckiematrix
simulator and then on the physical drone.

Built from Duckietown's [template-ros2](https://github.com/duckietown/template-ros2).

## Pipeline

Only predefined clips are used. Everything musical is computed ahead of time on
the laptop; the music itself is played on a phone.

```
laptop (offline, plain Python venv)
  clip.wav --analyze--> features.json --generate--> schedule.json   (schedule shipped to the drone)

laptop (show time)
  microphone hears the phone --listen_sync--> "song t=0 was N s ago"  ---> show/start_in

drone (ROS 2, runs on the DD24-B or the virtual drone)
  player_node --> safety_node --> primitives_node --> send_setpoint (TODO)
   schedule on     geofence,       min-jerk
   one monotonic   altitude,       setpoint
   clock + CSV log speed, kill,    streams
                   0.5 s watchdog
```

The player runs on the drone so Wi-Fi jitter only affects the single start
signal, and the safety node can still land the drone if the link drops.

## Layout

| Path | Contents |
| --- | --- |
| `packages/drone_primitives` | Movement primitives, patterns, min-jerk trajectories |
| `packages/choreographer` | `generate.py`: features.json -> schedule.json |
| `packages/show_runner` | `safety_node`, `player_node`, `show.launch.py` |
| `libraries/music_tools` | Laptop-only tools: `analyze.py`, `sync_eval.py`, `listen_sync.py` (plain Python, not in the drone image; see its `requirements.txt`) |
| `assets/config` | `limits.yaml`, `latency.yaml` |
| `assets/schemas` | Example `features.json` / `schedule.json` |
| `assets/music` | Place test clips here (no copyrighted audio) |
| `launchers/` | `default.sh` (show), `analyze.sh`, `sim.sh` |
| `tests/` | pytest unit tests |
| `dtproject/` | Duckietown project metadata |

## Status

Work in progress. Items that depend on the Duckietown tutorials (setpoint
topic/message, takeoff/land/arm services, simulator commands) are marked `TODO`
in the code.

## Build and run

See the Duckietown docs for `dts devel build` / `dts devel run`.
