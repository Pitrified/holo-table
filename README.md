# Holo table

Ok just a pinch to zoom in the air for now.

## Setup and deploy

On both the server and the client:

```bash
git clone https://github.com/Pitrified/holo-table.git
poetry install
```

On the server:

```bash
poetry run receiver --ip <server_ip> --port <server_port>
```

On the client:

```bash
poetry run sender --ip <server_ip> --port <server_port>
```

## Ideas

In the demo show a fractal, and zoom it.

Video capture to generate a training dataset,
we want to be able to detect the pinch gesture.
We can identify the control position,
and the pinch gesture is a simple distance between the two fingers.
We track how the distance changes over time,
and set the zoom level based on that.

Or skip the control position, and just track the pinch gesture.
If the distance changes too quickly, ignore it.
If the distance changes too slowly, ignore it as well as it's probably noise.

Or we could detect a quick movement towards the screen,
and then track the pinch gesture.

Some sort of LSTM to learn the whole pinch gesture?

Is the message the current distance between the two fingers?
The proportion from the start to the current?
I think timestamp + distance is ok,
then the client can parse the zooming from that.

Or send the whole wireframe of the hand.

We could apply the smoothing on the landmarks directly.
