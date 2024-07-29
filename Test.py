# Use a pipeline as a high-level helper
from transformers import pipeline
import argparse
#print('в какую папку сохранить txt файл:')
#path = str(input())

query = "Make one small prompt for album. You must write it in one small sentence approximately 30 words."

pipe = pipeline("text-generation", model="ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16")

examples = """
1. Soulful pop ballad from the early 2000s with a delicate piano melody, soaring string arrangements, and heartfelt lyrics about lost love and self-discovery.
2. Upbeat indie rock anthem from the 2010s with jangly guitars, a driving rhythm section, and catchy chorus focusing on youthful rebellion and adventure.
3. High-energy electronic dance track from the late 2010s featuring pulsating synths, a thumping bassline, and minimal but impactful lyrics perfect for the club scene.
4. Classic country song from the 1990s with twangy guitars, a steady drumbeat, and heartfelt storytelling about heartache and redemption in a small town.
5. Smooth R&B groove from the mid-2000s with silky vocals, lush harmonies, and sensual lyrics about love and intimacy.
6. Moody alternative rock track from the early 2010s with distorted guitars, a brooding bassline, and introspective lyrics about existential angst and self-reflection.
7. Laid-back reggae tune from the 1980s with a syncopated rhythm, warm bassline, and positive lyrics promoting peace and unity.
8. Sophisticated jazz piece from the 1950s with intricate saxophone solos, a swinging rhythm section, and lush piano chords evoking the smoky ambiance of a late-night club.
9. Old-school hip-hop track from the late 1980s with a funky beat, turntable scratches, and socially conscious lyrics about urban life and resilience.
10. Retro synthpop song from the early 1980s featuring catchy synth melodies, a driving electronic beat, and nostalgic lyrics about love and dreams.
11. Classic blues track from the 1950s with soulful guitar riffs, a slow, steady rhythm, and raw, emotional lyrics about heartache and perseverance.
12. Acoustic folk song from the 1960s with gentle guitar strumming, harmonious vocals, and storytelling lyrics that reflect on nature and personal experiences.
"""
context = """ 
The review was taken from a site where most people write their feelings about the song, but I don't need this information.
"""

review_1 = """
1. I swear they say the word California in every song, it's actually insane. Also, Otherside is one of the greatest songs of all time
This album had the potential to be absolutely incredible but shot itself in its foot

REVIEW REPOST #090. I repost my old reviews, which got no attention, with updated thoughts. Posted this review 1.5 years ago, but I'm posting it again.

Look, I was never a big fan of Red Hot, despite the Can't Stop riff being one of the first that I learned on guitar. The vocalist always annoyed me with his funky "rapping", but if there were songs that I absolutely loved from them, they were all singlehandedly on this album. And it's all of the more melodic tracks, they can really do those well. Otherside, Scar Tissue, This Velvet Glove, Porcelain...

So I decided to finally re-listen and re-review this album.

And despite it being my favorite RHCP record, it's still extremely flawed. At times, it shows its awesomeness, and at other times, it kind of falls flat on its face. A lot of these songs just haven't aged well. Especially Anthony's fast-paced funky vocals, which are just so annoying and over-the-top at times. His singing isn't the best either, his style altogether just hasn't aged very well. I think this album's strongest part is easily the instrumentation and production, the guitar work is really great, and the basslines are always super memorable.

And there are some amazing highlights on here obviously. Usually with this band, the more popular the song is, the better it probably is honestly. And that's very much the case for this album. I mean Otherside is one of my favorite songs of all time, with that beautiful bassline and guitar, and probably Anthony's best vocal performance, Scar Tissue is also beautiful, and Californication is a timeless classic, with a gorgeous legendary riff, how beautiful that song is in general. This Velvet Glove also has an absolutely stunning riff and great vocals. Yet again one of their best songs ever and an absolutely amazing vocal performance too. Those riffs are just sooo good... I used to bump this song on my way to freshman year school, so much nostalgia... Oh and Porcelain?? Straight up sounds like a Duster song, that's actually CRAZY. It's quite beautiful. Make the whole album like that, I'd absolutely love it. But that isn't the case. And basically, every single other song is... Well the more "funky" style from RHCP, and I can't say I love it. Some of it isn't bad, but generally, Anthony's vocals are a huge turn-off on a lot of these songs, despite some of the instrumentals being decent. And the lyrics aren't anything special either.

But I still feel like Red Hot Chili Peppers make Rock music for people who don't like Rock music, honestly. With that said though, this is their best album. I WISH I liked it more, honestly. I was hoping for it to grow on me, I really hoped I could like the rest of the tracks, except for the classic hits, but that didn't really happen. Easily grew on me and a few more but that's it. Still, though it's not a bad album at all. The reason I'm upset is because it could've been incredible, and it ended up being just good. Particularly the 2nd half really suffered the most. I Like Dirt, Get On Top, Purple Stain and Emit Remmus all should've been removed from the tracklist... All of those songs drag the album down a lot.

Still though, this is their best album by a long stretch with some OUTSTANDING highlights. Could've condensed it to 10 tracks, and it would've been much better, but hey, I still like it a lot for a RHCP album. They ain't topping this.

FAV TRACKS: Otherside, Californication, Scar Tissue, This Velvet Glove, Porcelain, Easily
LEAST FAV TRACKS: Get On Top, Purple Stain, I Like Dirt

2. Albums that taught me when I was a kid >>> Red Hot Chilli Peppers - Californication - 1999
"Ding, dang, dong, dong, deng, deng, dong, dong, ding, dang" - Around The World

Although it's not the Red Hot's best album in terms of impact and musical innovation (although that's debatable in the end), Californication remains my favourite album and the one I enjoy listening to again. In a context where Red Hot are skating a few years after a more disappointing One Hot Minute, John Fusciante reintegrates the band and gives a new saving breath. We can then speak of a miracle

For my part, I think it's one of the best Pop Rock/Rap album where the fusion is perfectly balanced and mastered, which makes the album still sounds remarkably "current", I'd rather say that it has aged very well, and even the rapped parts that may seem a bit old-fashioned, don't impact the rest. The melodies are sensational, the rhythmics are monumental and the riffs are so catchy that it creates a perfect and unique alchemy. What's also clever is that the Red Hot have stayed in their comfort zones regarding the themes. They all have the culture. Moreover, the theme is broad and rich enough to make something real and coherent.

Ironically, although Californication makes 15 tracks, only the first part up to Porcelain is really worth it, the rest is globally uneven, but it contains at least 5 wonderful songs, including 1 cult (Californication, Scar Tissue, Otherside, Around The World, Parallel Universe) that I could devour every day without being disgusted and forget the flaws of the album. Finally, this may be the first album that contains rap I've ever listened to in my life before I even discovered this genre and fell in love with it afterwards.

3.This album's predecessor 'One Hot Minute' made one thing abundantly clear - the Chilis would never sound convincingly cool again, and to their great credit the band seemed to acknowledge this.
'Californication' saw an immediate move towards more mature writing and stadium ready sounds befitting a group of ageing rockers and better yet it proved to be one of the more successful attempts at this type of transitional release.

The title track, 'Scar Tissue' and 'Around the World' were big radio hits, the sort missing from 'One Hot Minute' - while album tracks like 'Easily', 'Parallel Universe' and 'Road Trippin' were altogether cleaner sounding, more slick.

Maybe it says a lot about this band that this middle ground/mostly safe recording ranks as their second most consistently enjoyable release - it takes a certain craftsmanship to make this type of material sound natural so don't knock it I guess, NR

4.  Red Hot Chilli Pepper's seventh studio album topped certainliy not their classics Blood Sugar Sex Magik and Mother's Milk, but it contents some popular songs with "Scar Tissue", "Otherside" and "Californication" and some typical RHCP sounding, punky-funky ones, in particular "Parallel Universe" or the opener "Around the World". Therefore Californication is not an excellent, but a really good RHCP album and quite collectible.

5. As I continue my old CD reviews, I didn’t know which artist to pick next so I decided to randomly select from all those where I own more than one album (minus The Beatles I might do them separately). For the bands where I only own one I will cover these as a series at the end.
So I landed on the Red Hot Chilli Peppers a band I really liked in the late 90s and early 00s but I lost interest with a bit in the last 10-15 years. On the band Chad Smith is a good drummer, Anthony Kiedis can be an inconsistent lyricist and vocalist but for me the reason I was drawn to the band and continued to come back for was the talents of Flea and John Frusciante (I know there is band members through the years and I will cover that when needed during that album review).

Rather than start with the earliest album I own I will start with the album I bought first and that is Californication. I firstly grew in love with the song Otherside having seen the video and bought the album after that. The album has a great start and in fact a really good first half. Around the World has a frantic guitar intro before the drums and then Fleas funky bassline hits. This sets the tone for a band in sync throughout. Frusciante is consistently great and versatile throughout the record with infectious hooks and rifts and Flea supports perfectly with the two harmoniously in sync. What lets the album down and probably any RHCP album is there is some daft lyrics but as a teen a completely saw past them and even now those guitars rule. Song wise the highlights include the aforementioned Otherside, good harmony in the vocals and amazing guitar performance make this possibly my favourite RHCP song. Scar Tissue is a slower song with a nice guitar solo and. Californication has a nice groove throughout and good hook in the chorus. These of course are the singles but I also do like Easily and This Velvet Glove.

Overall, l I still really like the album despite some flaws and the overuse of California. The guitar riffs and basslines are enough to make up for that.

Fav Songs - Scar Tissue, Otherwise, Californication, This Velvet Glove

6. CaliforNIAAAAA. CaTIOOONNN. I always accent the end of my sentence or my WORRRRRRRDS
Classic 90s album that got really popular. These guys are such a unique group in terms of there style because they are so funky and all over the place. Sometimes they make really weird funk rock songs and then other times they just make slow ballads. This album is both of those things easily. With this being their most popular album, there is probably at least 3 songs you’ve heard on the radio from this album. Those songs being Californication, Otherside, and Scar Tissue, which are all great songs. They are popular for a reason I guess. Parallel universe is also pretty good besides some of the singing from time to time. This Velvet Glove is another standout that grabbed my ear on this listen, and Road Trippin is a nice closer. This album is great, but also some of the songs in the middle drag it down a bit. If those were cut, it would be better. Songs like Porcelain and I Like Dirt are kinda just boring and don’t really do much for me. Still, overall this a solid album and it’s kind of the album to start with if you’re getting into these guys.

7. This is without a doubt my most nostalgic album; the RHCP are what got me into Alt Rock music when I was younger, I can remember discovering the most iconic songs here, like 'Californication' or 'Otherside', and playing those intensely for a couple of months, and then becoming obsessed by the more deeper cuts, like 'Parallel Universe' with its eccentric guitar and striking chorus, or 'Savior' for its uncommon composition and varied sounds.
And obviously the best RHCP track stands right here in this iconic record: 'Scar Tissue', such a perfect song. I can listen to this lovely rhythmic ballad again and again without ever getting bored by it; it's delicate, but dynamic enough for me to listen to it in any circumstance, the guitar melody is charming and comforting, as are Anthony Kiedis' southern vocals. I love it so much. So yeah, it's my entry into this vast music genre and I support this with all my heart.

8. A classic and a half. This was really when RHCP truly developed their renowned sound that everyone assosiates them with. This record even if i loved it originally it has grown on me over time, to the point that i dont dislike any of the songs here. I mean this shit is a classic for a reason every hit of the Chilli Peppers are here i mean Californication, Scar Tissue, Around the world, Otherside its fucking stacked. This was when John Frusciante joined back the band and you already know the guitars are amazing. The bass is always on the forfront by Flea because its the Chilli Peppers and the drums work so well in the albums favour. Anthony's vocals sound amazing here, it is clear that time has done him justice. Every song here sound nothing like the other which is great as when you are listening this funky style, if everything mashes together it can be kinda tedious to listen to a whole album but this is anything but that.
Favorite songs: "Scar Tissue" amazing slow song one of the best the Chillis have ever done, "Otherside" this is a melancholic type of song about addiction that fucking slaps so hard, this is one of the top 20 RHCP songs at the very least. "Easily" is easily the best deep cut here in my opinion and a very underrated song of theirs, as is "This Velvet Glove" and "Purple Stain" which are all top 20 meterial again. Then you got the funky jams like "I like Dirt" and "Get on top" that are absolute bangers, also "Around the world" is one their best openers ever and "Parallel Universe" is easy a top 10 RHCP song. Lastly "Californication" is probably my 3rd favorite RHCP song and one of my all time favorites. Thing is if i love all these songs so much and this is one of my favorite bands albums why isnt it a 10. Well even if ive grown to like some of the lesser moments they still drug the enjoyment of the album out in the form of "Savior" which is kinda confusing to this day, "Emit Rammus" is kinda mid and "Porcellain" even if beutifull it isnt a song id ever really seek out of the context of this album.

All in all an amazing album with some hardly noticable week points that has claimed classic status for a reason... because its so godamn good.
"""

lyrics_1 = """
All around the world, we could make time
Rompin' and a stompin' 'cause I'm in my prime
Born in the north and sworn to entertain ya
'Cause I'm down for the state of Pennsylvania
I try not to whine, but I must warn ya
'Bout the motherf-cking girls from California
Alabama baby said hallelujah
Good God girl, I wish I knew ya
I know, I know for sure
That life is beautiful around the world
I know, I know it's you
You say hello and then I say adieu
Come back baby 'cause I'd like to say
I've been around the world, back from Bombay
Fox hole love Pie in your face
Living in and out of a big fat suitcase
Bonafide ride step aside my Johnson
Yes, I could in the woods of Wisconsin
Wake up the cake, it's a lake she's kissin' me
As they do when they do in Sicily
I know, I know for sure
That life is beautiful around the world
I know, I know it's you
You say hello and then I say adieu
Where you wanna go, who you wanna be
What you wanna do, just come with me
I saw God and I saw the fountains
You and me girl sittin' in the Swiss mountains
Me oh my oh me and guy ho
Freer than a bird 'cause were rockin' Ohio
Around the world I feel dutiful
Take a wife 'cause life is beautiful
I know, I know for sure
Ding ding, dong dong, ding ding, dong dong, ding ding
I know, I know it's you
Ding ding, dong dong, ding ding, dong dong, ding ding
Mother Russia do not suffer
I know you're bold enough
I've been around the world
And I have seen your love
I know, I know it's you
You say hello and then I say adieu

"""

lyrics_2 = """
To finger paint is not a sin
I put my middle finger in
Your monthly blood is what I win
I'm in your house now let me spin
Python power straight from Monty
Celluloid loves got a John Frusciante
Spread your head and spread the blanket
She's too free and I'm the patient
Black and white a red and blue
Things that look good on you
And if I scream don't let me go
A purple stain, I know
Knock on wood we all stay good
'Cause we all live in Hollywood
With Dracula and Darla Hood
Unspoken words were understood
Up to my ass in alligators
Let's get it on with the alligator haters
Did what you did, did what you said
What's the point yo what's the spread
Black and white a red and blue
Things that look good on you
And if I scream don't let me go
A purple stain, I know
And if I call for you to stay
Come hit the funk on your way
It's way out there but I don't care
'Cause this is where, I go
Knock on wood we all stay good
'Cause we all live in Hollywood
With Dracula and Darla Hood
Unspoken words were understood
It's way out there but I don't care
'Cause this is what I want to wear
Knock on wood we all stay good
'Cause we all live in Hollywood
To finger paint is not a sin
I put my middle finger in
Your monthly blood is what I win
I'm in your house now let me spin
Feather light but you can't move this
Farley is an angel and I can prove this
Purple is a stain upon my pillow
Let's sleep weeping willow
Black and white a red and blue
Things that look good on you
And if I scream don't let me go
A purple stain, I know
And if I call for you to stay
Come hit the funk on your way
It's way out there but I don't care
'Cause this is where, I go
Knock on wood we all stay good
'Cause we all live in Hollywood
With Dracula and Darla Hood
Unspoken words were understood
It's way out there
But I don't care
'Cause this is what
I want to wear
Knock on wood we all stay good
'Cause we all live in Hollywood
"""

lyrics_3 = """
Psychic spies from China try to steal your mind's elation
And little girls from Sweden dream of silver screen quotation
And if you want these kind of dreams it's Californication
It's the edge of the world and all of Western civilization
The sun may rise in the East, at least it settled in a final location
It's understood that Hollywood sells Californication
Pay your surgeon very well to break the spell of aging
Celebrity skin, is this your chin, or is that war you're waging?
First born unicorn
Hardcore soft porn
Dream of Californication
Dream of Californication
Dream of Californication
Dream of Californication
Marry me, girl, be my fairy to the world, be my very own constellation
A teenage bride with a baby inside getting high on information
And buy me a star on the boulevard, it's Californication
Space may be the final frontier, but it's made in a Hollywood basement
And Cobain, can you hear the spheres singing songs off Station To Station?
And Alderaan's not far away, it's Californication
Born and raised by those who praise control of population
Well, everybody's been there and I don't mean on vacation
First born unicorn
Hardcore soft porn
Dream of Californication
Dream of Californication
Dream of Californication
Dream of Californication
Destruction leads to a very rough road, but it also breeds creation
And earthquakes are to a girl's guitar, they're just another good vibration
And tidal waves couldn't save the world from Californication
Pay your surgeon very well to break the spell of aging
Sicker than the rest, there is no test, but this is what you're craving
First born unicorn
Hardcore soft porn
Dream of Californication
Dream of Californication
Dream of Californication
Dream of Californication
"""


prompt = f"""
{query}
following context, {context}
You must to follow the examples: {examples}

album reviews: {review_1}
song lyrics: {lyrics_1} 

"""

a = [lyrics_1, lyrics_2, lyrics_3]
order = ["Around the World", 'purple stain', 'californication']


def LammaTest(path):
    for i in range(0, len(a)):
        for num in range(1, 4):


            prompt = f"""
            {query}
            following context, {context}
            You must to follow the examples: {examples}

            album reviews: {review_1}
            song lyrics: {a[i]} 

            """


            res = pipe(prompt)

            with open((path + 'Lamma70B.txt'), "a", encoding="utf-8") as file:
                file.write('песня: ' + order[i] + 'запуск '+ num + res + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLAMA test')
    parser.add_argument('-s', '--save_dir', default=None, type=str,
                      help='path to save directory')
    args = parser.parse_args()
    LammaTest(args.save_dir)
