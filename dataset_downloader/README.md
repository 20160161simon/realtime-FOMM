
### Dataset Download & Preprocessing

This part primarily references the [First Order Motion Model (FOMM)](https://github.com/AliaksandrSiarohin/first-order-model) and its [associated video preprocessing pipeline](https://github.com/AliaksandrSiarohin/video-preprocessing) (which cannot work currently).

Install dependencies

```bash
pip install -r requirements.txt
```

After installing all the dependencies.

```bash
cd ./preprocessing/VoxCeleb
python download_VoxCeleb.py --metadata metadata.csv --format .mp4 --out_folder data --workers 8  
```
If you do not have sufficient memory (around 300GB) for the full dataset, we also provide a smaller subset of VoxCeleb (approximately X GB). You can run the following code instead of the command above.

```bash
cd ./preprocessing/VoxCeleb
python metadata-filter.py
python download_VoxCeleb.py --metadata metadata_filtered.csv --format .mp4 --out_folder data --workers 8  
```

