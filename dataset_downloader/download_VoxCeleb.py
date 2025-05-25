import pandas as pd  
import imageio  
import os  
from multiprocessing import Pool  
from itertools import cycle  
import warnings  
from tqdm import tqdm  
from skimage import img_as_ubyte  
from skimage.transform import resize  
from argparse import ArgumentParser  
import yt_dlp  
  
warnings.filterwarnings("ignore")  
  
def download(video_id, args):  
    video_path = os.path.join(args.video_folder, video_id.split('#')[0] + ".mp4")  
      
    if os.path.exists(video_path):  
        return video_path
          
    ydl_opts = {  
        'format': 'best[ext=mp4]',  
        'outtmpl': video_path,  
        'quiet': True,  
        'no_warnings': True,  
        'writesubtitles': True,  
        'writeautomaticsub': True,  
        'subtitleslangs': ['en'],  
        'ignoreerrors': True,  
    }  
      
    try:  
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  
            ydl.download(['https://www.youtube.com/watch?v=' + video_id.split('#')[0]])  
        return video_path  
    except Exception as e:  
        print(f'Fail when download {video_id}: {str(e)}')  
        return None  
  
def save(path, frames, format):  
    if format == '.mp4':  
        imageio.mimsave(path, frames)  
    else:  
        os.makedirs(path, exist_ok=True)  
        for i, frame in enumerate(frames):  
            imageio.imsave(os.path.join(path, str(i).zfill(7) + '.png'), frame)  
  
def run(data):  
    video_id, args = data

    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):  
        download(video_id.split('#')[0], args)  
  
    if not os.path.exists(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4')):  
        print(f'Cannot load {video_id.split("#")[0]}, the link is invalid')  
        return  
          
    reader = imageio.get_reader(os.path.join(args.video_folder, video_id.split('#')[0] + '.mp4'))  
    fps = reader.get_meta_data()['fps']  
  
    df = pd.read_csv(args.metadata)  
    df = df[df['video_id'] == video_id]  
      
    all_chunks_dict = [{'start': df['start'].iloc[j], 'end': df['end'].iloc[j],  
                      'bbox': list(map(int, df['bbox'].iloc[j].split('-'))), 'frames':[]} for j in range(df.shape[0])]  
    ref_fps = df['fps'].iloc[0]  
    ref_height = df['height'].iloc[0]  
    ref_width = df['width'].iloc[0]  
    partition = df['partition'].iloc[0]  
      
    try:  
        for i, frame in enumerate(reader):  
            for entry in all_chunks_dict:  
                if (i * ref_fps >= entry['start'] * fps) and (i * ref_fps < entry['end'] * fps):  
                    left, top, right, bot = entry['bbox']  
                    left = int(left / (ref_width / frame.shape[1]))  
                    top = int(top / (ref_height / frame.shape[0]))  
                    right = int(right / (ref_width / frame.shape[1]))  
                    bot = int(bot / (ref_height / frame.shape[0]))  
                    crop = frame[top:bot, left:right]  
                    crop = img_as_ubyte(resize(crop, (256, 256), anti_aliasing=True)) 
                     
                    entry['frames'].append(crop)  
    except (imageio.core.format.CannotReadFrameError, IndexError, ValueError) as e:  
        print(f"Error when dealing with {video_id}: {str(e)}")  
  
    for entry in all_chunks_dict:  
        if 'person_id' in df:  
            first_part = df['person_id'].iloc[0] + "#"  
        else:  
            first_part = ""  
        first_part = first_part + '#'.join(video_id.split('#')[::-1])  
        path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6) + '.mp4'  
        if entry['frames']:  
            save(os.path.join(args.out_folder, partition, path), entry['frames'], '.png')  
  
if __name__ == "__main__":  
    parser = ArgumentParser()  
    parser.add_argument("--video_folder", default='youtube', help='Youtube video path')  
    parser.add_argument("--metadata", default='metadata.csv', help='metadata path')  
    parser.add_argument("--out_folder", default='VoxCeleb-png', help='output path')  
    parser.add_argument("--format", default='.png', help='store format (.png or .mp4)')  
    parser.add_argument("--workers", default=1, type=int, help='number of workers')
      
    args = parser.parse_args()  
      
    if not os.path.exists(args.video_folder):  
        os.makedirs(args.video_folder)  
    if not os.path.exists(args.out_folder):  
        os.makedirs(args.out_folder)  
    for partition in ['test', 'train']:  
        if not os.path.exists(os.path.join(args.out_folder, partition)):  
            os.makedirs(os.path.join(args.out_folder, partition))  
  
    df = pd.read_csv(args.metadata)  
    video_ids = set(df['video_id'])  
      
    pool = Pool(processes=args.workers)  
    args_list = cycle([args])  
    for chunks_data in tqdm(pool.imap_unordered(run, zip(video_ids, args_list))):  
        None