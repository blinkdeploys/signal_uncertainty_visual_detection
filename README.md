# Effects of Signal Variation on Visual Pattern Detection Under Fixed and Variable Noise 


## Summary

This study investigates how signal uncertainty affects the detection of simple and complex visual patterns. Participants completed a two-interval forced-choice task, detecting either sinusoidal gratings or band-limited noise textures. Signal uncertainty was manipulated by presenting either a single signal type or one of five variations per trial. Additionally, noise was either fixed across trials or varied.

Contrast thresholds were measured to assess detection performance under these conditions, providing insight into how uncertainty influences the perception of structured and complex patterns.

## Getting Started


From the woring director, run:

`docker compose build --no-cache`

`docker compose up -d`

Nagigate to `https//localhost:5000/dashboard`

Errors:

If you get 

Run 

```sh
lsof -i :5000
```
You should see something similar to below:

```sh
COMMAND   PID           USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
python3   <PID_NUMBER>  user   5u  IPv4  12345      0t0  TCP *:5000 (LISTEN)
```

Note the PID number then run 

```sh
kill <PID_NUMBER>
```

Ensure that Docker is still running then rerun 

```sh
docker compose up -d
```