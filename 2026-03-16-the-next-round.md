I want the agent to be able to write (and run) mini-programs against an API that we provide.
The programs will parse the live camera feed and do things based on what they see.
These kinds of actions should have a much faster response than a normal agent round-trip.
I'd like to have a good set of object tracking APIs, so that we can build something like:
- Project: Place the toy on the box (e.g. the project creates a frame on the page for the child to place a toy)
- Track: the object on the box
- As the object moves around the page, it triggers different animations or sounds. Maybe there's pictures of instruments being projected, and each one makes a different sound when the toy is placed on it.
- We can also print out a set of tracking patterns if need be, for easier object detection, but really this shouldn't be too hard these days.
  
Or for storytelling the user can work with the agent to create a story, with different scenes, and then the agent writes code to actually play the story with narration, music, animation.
We can provide the agent with information about other google apis that it can call (ie. we give it an initliazed API client), for music and video generation. Instead of having the main voice agent write everything itself, it should be able to call out to another model  (e.g. the smarter gemini-3.0-flash). We'll need a mechanism to let the agent know when async tasks (such as code or image generation) are complete. I believe there's a bi-directional text channel we could use for these kinds of updates? We should test and prototype it first with the image generation flow, which is already async. We'll need to update the master prompt to demonstrate how to handle this.

So the main live agent can kick off background things, and it, and the code it writes, should have the ability to call other models or APIs, to play sounds, and to use our projector interface to write to coordinate space. We may also want to keep in memory a "map" of what our overlay looks like. We should be able to render an ASCII grid of the current overlay, and/or provide a JSON or XML description of it. Ideally our tools should take a "name" input, so a generation isn't just an image it's "image-of-hand-drawn-cat", so we can use that in our JSON description of the overlay.

Also we need to tune our image generation prompts. When the user provides an image, we want to enhance, not replace it. We should adhere closely to their original vision. We'll need to revise and improve our file/image handling system so that the agent can easily see and reference past created images. These images will need to be actually store in some kind of `session/` so we can reference them later (such as when creating a story, or in further iterative image generations).