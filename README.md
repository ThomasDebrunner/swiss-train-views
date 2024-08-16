# Swiss Train Window

[LIVE SITE](https://thomasdebrunner.github.io/swiss-train-views/)

What can you see when looking out the window on a swiss train?

This project is a very dirty approach to figure out how far you can see when you look out the left and right of the train.

This is computed using terrain data from swisstopo and data from SBB.

Used data:

- [SBB kilometrage points](https://data.sbb.ch/explore/dataset/linienkilometrierung/export/?disjunctive.linienr&disjunctive.liniename&location=11,47.21281,8.58273&basemap=00c4d6)

- [Swisstopo DHM25](https://www.swisstopo.admin.ch/de/hoehenmodell-dhm25)


## How to run

1. Clone the repository
2. Download the datasets from above, and place them in the `data` folder
3. Run the `prepare_data.py` script to prepare the data
4. This creates a json file `train_data.json`
5. Copy that file to `page`, you can run `npm run dev` in the `page` folder to start a local server

## Known issues

Tunnels don't work. We take the altitude of the track from the altitude of the terrain, which is not correct for tunnels.

So whenever the train is in a tunnel, we estimate that we can see extremely far, as we're on a mountain. This is not correct.