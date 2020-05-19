#imports
from moviepy.editor import *
from pydub import AudioSegment
import glob, os, argparse


## interface
parser = argparse.ArgumentParser(description='Cut mp4 regarding the annotations')
parser.add_argument('-a','--annotation_path', type=str, dest='annotation_path', action='store', default='./ELAN/Annotations', required=True,
                    help='annotation text file folder')
parser.add_argument('-m', '--mp4_path', type=str, dest='mp4_path', action='store', default='./ELAN/MP4', 
                    help='corresponding video file folder')
parser.add_argument('-o','--output_path', type=str, dest='output_path', action='store', default='./MP3_Snippets', 
                    help='output path')
parser.add_argument('-g','--gender', dest='gender', default='f', 
                    help='"f" for female, "m" for male')
args = parser.parse_args()

gender = args.gender 
path_txt = args.annotation_path 
path_vid = args.mp4_path 
save_path_audio = args.output_path 

annotations = {}
vids = []

# import and process text files
os.chdir(path_txt)
for file in glob.glob("*.txt"):
    annotations[file.__str__()] = []
    with open(file, 'r') as f:
        for line in f:
            input = []
            for word in line.replace('\n', '').split('\t'):
                input.append(word)
            annotations[file.__str__()].append(input)

# import video files
os.chdir(path_vid)
for file in glob.glob('*.mp4'):
    if (annotations.__contains__(file.replace('.mp4', '.txt'))):
        vids.append(file)

# extract audio track from video files
for v in vids:
    video = VideoFileClip(v)
    audio = video.audio
    audio.write_audiofile(os.path.join(path_vid, v.replace('mp4','wav')))
    audio = AudioSegment.from_file(path_vid + '/' + v.replace('mp4', 'wav'))
    print('Video {} audio extracted.'.format(v))

    ### ELAN: cut audio file into clips/segments ###
    # audio clip name convention: vid-name_gender_segNr_point/no-point.wav
    for key, value in annotations.items():
        if (key.replace('txt', 'mp4') == v):
            i = 0
            if not os.path.exists(save_path_audio + '/' + v.replace('.mp4', '')):
                os.makedirs(save_path_audio + '/' + v.replace('.mp4', ''))

            while (i < len(value)):
                clip_name = v.replace('.mp4', '') + '_' + gender + '_' + str(i + 1) + '_' + str(value[i][2]) + '.wav'
                clip = audio[int(value[i][0]):int(value[i][1])]
                clip.export(os.path.join(save_path_audio, v.replace('.mp4', ''), clip_name), format="wav")
                i += 1
            print('Video {}: annotated and cutted'.format(v))
