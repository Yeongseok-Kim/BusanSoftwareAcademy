def check_word(word):
    count_ab=word[0:3].count('a')
    count_ab+=word[0:3].count('b')
    return count_ab
count_word=0
f=open("song.txt","r")
while True:
    song_lyric=f.readline()
    if not song_lyric:
        break
    song_split=song_lyric.split(' ')
    for i in song_split:
        count_word+=check_word(i)
f.close()
print(count_word)