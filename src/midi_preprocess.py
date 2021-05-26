from PIL.Image import new
import mido
import csv
from mido import Message, MidiFile, MidiTrack



midi_note_numbers = {}

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
cnt = -1
for octave in range(-2, 8):
    for note in notes:
        cnt = cnt + 1 
        tmp = note + str(octave)
        midi_note_numbers[tmp] = cnt


#print(midi_note_numbers)



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



test = []
for i in range(len(new_data)):
    a = new_data[i]
    b = list(filter(None, a))
    test.append(b)


#print(test)


mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

tpb = mid.ticks_per_beat = 900


tick1 = int(mido.second2tick(5,tpb,500000)) 
tick = int(mido.second2tick(10,tpb,500000))
#print(tick)


track.append(Message('program_change', program=12, time=0))

stupid = []
for lst in test:
    note = lst[0]
    lst.pop(0)

    for beg, end in zip(lst[::2], lst[1::2]):
        #tick_beg = int(mido.second2tick(beg,tpb,500000))
        #tick_end = int(mido.second2tick( (end - beg), tpb ,500000))
        note_num = midi_note_numbers[note]
        #print(note + " " + str(note_num) + " " + "from " + str(beg) + " to " + str(end))
        tmp = [note, int(note_num), float(beg), float(end)]
        stupid.append(tmp)


        # track.append(Message('note_on', note=note_num, velocity=64, time=tick_beg))
        # track.append(Message('note_off', note=note_num, velocity=64, time=tick_end))
        



stupid.sort(key=lambda x: x[2])

for i in stupid: 
    print(i)



for i in range(len(stupid)-1):
    # if (stupid[i][2] == stupid[i+1][2]):
    #     pass

    # else:
    beg = stupid[i][2]
    end = stupid[i][3]
    note_num = stupid[i][1] 
    tick_end = int(mido.second2tick(end,tpb,500000))

    track.append(Message('note_on', note=note_num, velocity=64, time=0))
    track.append(Message('note_off', note=note_num, velocity=64, time=tick_end))



# track.append(Message('program_change', program=12, time=0))
# track.append(Message('note_on', note=64, velocity=64, time=0))
# track.append(Message('note_on', note=65, velocity=64, time=0))
# track.append(Message('note_off', note=65, velocity=64, time=tick))
# track.append(Message('note_off', note=64, velocity=64, time=0))

mid.save('new_song.mid')