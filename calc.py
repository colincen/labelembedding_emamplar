import os
import json
intent = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']
att_ways = ['dot-attention','multiplicative','additive']
fw = open('totals.txt','w')
for i in att_ways:
    fw.write(i+'\n\n')
    for j in intent:
        path ='./data/'+i+'/'+j
        f = open(path +'/result.txt', 'r')
        t = 0
        d = {}
        for k, line in enumerate(f):
            t += 1
            temp = json.loads(line)
            if len(d) == 0:
                d = temp
            else:
                for k1,v1 in temp.items():
                    d[k1]['prec'] += v1['prec']
                    d[k1]['rec'] += v1['rec']
                    d[k1]['f1'] += v1['f1']
        print(t)
        for k1, v1 in d.items():
            d[k1]['prec'] /= t
            d[k1]['rec'] /= t
            d[k1]['f1'] /= t
            K = k1
            if k1 == 'total':
                K = j
            fw.write(K +' %.2f\n' % (d[k1]['f1']))
        fw.write('\n')
#        tmp = json.dumps(d)
#        fw.write(j+'\n\n')
        

            
            
