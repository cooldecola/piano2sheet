import mido
import csv
from mido import Message, MidiFile, MidiTrack

with open('notes_info.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

new_data = []

for i in range(len(data)):
    ls = data[i]
    #print(ls)
    sum = ''
    for itm in range(1,len(ls)):
        sum = sum + ls[itm]

    if (sum != ''): 
        new_data.append(ls)


mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

print(mid.ticks_per_beat)

track.append(Message('program_change', program=12, time=0))
track.append(Message('note_on', note=64, velocity=64, time=0))
track.append(Message('note_off', note=64, velocity=64, time=int(mido.second2tick(5,480,500000)) ))
track.append(Message('note_on', note=65, velocity=64, time=0))
track.append(Message('note_off', note=65, velocity=64, time=92))

mid.save('new_song.mid')