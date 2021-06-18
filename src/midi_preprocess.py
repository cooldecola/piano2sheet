from PIL.Image import new
import mido
import csv
import pandas as pd 
from mido import Message, MidiFile, MidiTrack

# creating a dictionary for "lookup" when associating note name to MIDI note number
midi_note_numbers = {}

# linking each note on piano to the associated number in MIDI format
# ref: https://intuitive-theory.com/midi-from-scratch/
notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
cnt = -1
for octave in range(-2, 8):
    for note in notes:
        cnt = cnt + 1 
        tmp = note + str(octave)
        midi_note_numbers[tmp] = cnt


# getting rid of all notes that aren't used and 
# storing them in new_data
with open('notes_info.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

remove_dup_list = []

new_data = []
for i in range(len(data)):
    ls = data[i]
    sum = ''
    for itm in range(1,len(ls)):
        sum = sum + ls[itm]

    if (sum != ''): 
        new_data.append(ls)


# getting rid of all NaN values
cleaned_list = []
for i in range(len(new_data)):
    a = new_data[i] # first list 
    b = list(filter(None, a)) # same list but removed 
    print(b)
    b_len = len(b)
    new_b = [] # new list without extra times
    new_b.append(b[0]) # adding note info 
    rnge = len(b[1:b_len]) # length of list without note
    
    # case : only two times in list 
    if rnge == 2:
        tmp_ls = b[1:b_len]
        for i in tmp_ls:
            new_b.append(i)
        cleaned_list.append(new_b)

    
    if rnge > 2:
        list_remove = []
        new_b = new_b + b[1:b_len]
        #print(new_b)
        for i in range(1,b_len-1):
            abs_sub = float(b[i+1]) - float(b[i])
            abs_sum = (float(b[i+1]) + float(b[i]))/2
            percent_diff = 100 * (abs_sub/abs_sum)
            if percent_diff < 1:
                list_remove.append(b[i+1])
                #print(b[i+1])
        
        for i in list_remove:
            new_b.remove(i)

        cleaned_list.append(new_b)

print()
print()
for i in cleaned_list: 
    print(i)

# initializing MIDI file 
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)

# ticks per beat = 900 (conventional number for 120 bpm)
tpb = mid.ticks_per_beat = 900

# creating a list of notes played in order of sequence
seq_notes = []
for lst in cleaned_list:
    note = lst[0]
    lst.pop(0)

    for beg, end in zip(lst[::2], lst[1::2]):
        note_num = midi_note_numbers[note]
        tmp = [note, int(note_num), float(beg), float(end)]
        seq_notes.append(tmp)
        
# creating dataframe for manipulation
df =  pd.DataFrame(seq_notes)
df.columns = ['note', 'note_num', 'start', 'end']

#difference column
df['difference'] = df['end'] - df['start']

# df_full is duplicate of df - except first df contains start
# and second df contains end
df_full = df[['note', 'note_num', 'start', 'difference']]
df_full = df_full.append(df[['note', 'note_num', 'end', 'difference']])

# time column is the combination of start and end - deleting all NaN values
df_full['time'] = df_full['start'].fillna(df_full['end'])
df_full = df_full.drop('start', axis = 1)
df_full = df_full.drop('end', axis = 1)

#sorting df_full in accordance with MIDI file format 
sorted_df = df_full.sort_values(['time', 'difference'], ascending=[True, False])

# calculating the delta times of each note in del_list
del_list = []
del_list.append(0)
time_list = sorted_df['time'].tolist()
for i in range (len(time_list)-1):
    del_list.append(time_list[i+1] - time_list[i])

sorted_df['del'] = del_list
sorted_list = sorted_df.values.tolist()

# buffer list determines if a 'read' note is a starting one or ending one through
# use of pop() and append()
buffer = []

# starting the track
track.append(Message('program_change', program=12, time=0))

for i in range(len(sorted_list)):
    note_num = sorted_list[i][1]
    note = sorted_list[i][0]
    time = sorted_list[i][4]
    tick = int(mido.second2tick(time,tpb,500000))
    
    if not buffer: 
        track.append(Message('note_on', note=note_num, velocity=64, time=tick))
        buffer.append(sorted_list[i][0])

    elif len(buffer) > 0:
        if note in buffer:
            track.append(Message('note_off', note=note_num, velocity=64, time=tick))
            buffer.remove(note)
            
        else: 
            track.append(Message('note_on', note=note_num, velocity=64, time=tick))
            buffer.append(note)


mid.save('new_song.mid')