from vocabulary.vocabulary import Vocabulary as vb
import random, time
noword = ['zero', 'one', 'two', 'get', 'watch', 'eleven', 'good', 'job', 'ChaeJiseok']
yesword = ['tear', 'bow', 'produce', 'wind', 'ellipses', 'bass', 'does', "dove"]
candidate_list = (noword + yesword)
random.shuffle(candidate_list)
for word in candidate_list:
    print(word)
    # if ((i + 1) % 100) == 0 :
    #     print("hetero number %d" %(i + 1))
    # i += 1
    pron_dict = vb.pronunciation(word, format = "dict")
    time.sleep(0.5)
    americanHeritageProns = []
    if pron_dict is False:
        print(pron_dict)
    else :
        for i in range(len(pron_dict)):
            if ("American Heritage" in pron_dict[i]['attributionText']):
                americanHeritageProns.append(pron_dict[i]['raw'])
        if len(set(americanHeritageProns)) != 1:
            print(True,  set(americanHeritageProns))
        else :
            print(False,  set(americanHeritageProns))

############WIKTIONARY_RELATED_TEST############
if input("TEST? Y/n : ") == "Y":
    noword = ['zero', 'one', 'two', 'get', 'watch', 'eleven', 'good', 'job']
    yesword = ['tear', 'bow', 'produce', 'wind', 'ellipses', 'bass', 'does', "dove"]
    for i in range(1):
            stt = time.time()
            for word in noword:
                    wd = wikparser.fetch(word)
                    wdict = makeDictFromWikiWord(wd)
                    print(heteronym_check_from_wiktionary(wdict))
            print(time.time() - stt)
    for i in range(1):
            stt = time.time()
            for word in yesword:
                    wd = wikparser.fetch(word)
                    wdict = makeDictFromWikiWord(wd)
                    print(heteronym_check_from_wiktionary(wdict))
            print(time.time() - stt)