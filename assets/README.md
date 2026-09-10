# Assets

## `powered-by-bgg.png`

Attribution for the BoardGameGeek data, which its terms of use require.

**This file is not in the repository** and needs to be added once. Download a
colour version from BoardGameGeek's
[Powered by BGG logo folder](https://drive.google.com/drive/folders/1k3VgEIpNEY59iTVnpTibt31JcO0rEaSw)
and save it here as `powered-by-bgg.png` (`.jpg` and `.svg` also work; the app
picks up whichever it finds).

Until it exists the app falls back to a plain text link reading "Powered by
BoardGameGeek", so attribution is still present and still points at
boardgamegeek.com. The logo is the better version of the same thing.

The image is inlined as a data URI at render time, so it works on static
hosting and needs no separate asset route.
