from PIL.Image import new
import mido
import csv
from mido import Message, MidiFile, MidiTrack


#Ignore this! 
def getChord(index, lst):
    pass


#creating a dictionary for "lookup" when associating note name to MIDI note number
midi_note_numbers = {}

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
cnt = -1
for octave in range(-2, 8):
    for note in notes:
        cnt = cnt + 1 
        tmp = note + str(octave)
        midi_note_numbers[tmp] = cnt


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

print(len(stupid))

for i in stupid: 
    print(i)


i = 0
note_list = []

while i < (len(stupid)-1):
    
    def nextSame(i):
        if ( (stupid[i][2] == stupid[i+1][2]) ):
            return True
        else: 
            return False

    if (not nextSame(i)):
        tmp_ls = stupid[i]
        note_list.append(tmp_ls)
        i = i + 1
        print(i)

    if (nextSame(i)):
        tmp_ls = []
        while (nextSame(i)):
            tmp_ls.append(stupid[i])
            i = i + 1
            print(i)
        tmp_ls.append(stupid[i])
        note_list.append(tmp_ls)
        i = i+1

    if (i == len(stupid)-2):
        break


for i in note_list: 
    print(i)

#print(note_list)



# for i in range(len(stupid)-1):

#     #local definition 
#     def nextSame(i):
#         if ( (stupid[i][2] == stupid[i+1][2]) and (stupid[i][3] == stupid[i+1][3]) ):
#             return True
#         else: 
#             return False

#     def addChord(i):
#         tick_delta = int(mido.second2tick(delta,tpb,500000))
#         track.append(Message('note_on', note=stupid[i][1], velocity=64, time=tick_delta))
#         print("on" + str(stupid[i][1]) + " delta: " + str(delta))
#         if (nextSame(i)):
#             addChord(i+1)

#         track.append(Message('note_off', note=stupid[i][1], velocity=64, time=tick_end))
#         print("off" + str(stupid[i][1]) + " delta: " + str(delta))

#     beg = stupid[i][2]
#     end = stupid[i][3]
#     note_num = stupid[i][1] 
#     tick_end = int(mido.second2tick(end,tpb,500000))
#     delta = 0.0


#     #condition 1 : first not and single note
#     if (not nextSame(i) and i == 0):
#         track.append(Message('note_on', note=note_num, velocity=64, time=0))
#         track.append(Message('note_off', note=note_num, velocity=64, time=tick_end))
#         delta = stupid[i+1][2] - stupid[i][3]

#     #condition 2 : first notes are chords
#     if (nextSame(i) and i == 0):
#         pass

#     #condition 3 : chord but not first notes
#     if (nextSame(i) and i != 0):
#         addChord(i)
#         # tick_delta = int(mido.second2tick(delta,tpb,500000))
#         # tick1 = tick_end
#         # tick2 = int(mido.second2tick(stupid[i+1][3],tpb,500000))
#         # note1 = note_num
#         # note2 = stupid[i+1][1]
#         # track.append(Message('note_on', note=note1, velocity=64, time=tick_delta))
#         # track.append(Message('note_on', note=note2, velocity=64, time=tick_delta))
#         # track.append(Message('note_off', note=note1, velocity=64, time=tick1))
#         # track.append(Message('note_off', note=note2, velocity=64, time=tick2))
#         delta = stupid[i+1][2] - stupid[i][3]

#     if (not nextSame(i) and i != 0):
#         tick_delta = int(mido.second2tick(delta,tpb,500000))
#         track.append(Message('note_on', note=note_num, velocity=64, time=tick_delta))
#         track.append(Message('note_off', note=note_num, velocity=64, time=tick_end))
#         delt = stupid[i+1][2] - stupid[i][3]




five = int(mido.second2tick(5,tpb,500000))
three = int(mido.second2tick(3,tpb,500000))
seven = int(mido.second2tick(7,tpb,500000))
track.append(Message('program_change', program=12, time=0))
track.append(Message('note_on', note=65, velocity=64, time=0))
track.append(Message('note_on', note=70, velocity=64, time=0))
track.append(Message('note_on', note=75, velocity=64, time=three))
track.append(Message('note_on', note=76, velocity=64, time=0))
track.append(Message('note_off', note=76, velocity=64, time=five))
track.append(Message('note_off', note=75, velocity=64, time=0))
track.append(Message('note_off', note=70, velocity=64, time=three))
track.append(Message('note_off', note=65, velocity=64, time=three))



mid.save('new_song.mid')