
en=open('joint_train.en','w')

with open('./it/train.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./koran/train.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./law/train.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./medical/train.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./subtitles/train.full.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

en.close()


de=open('joint_train.de','w')

with open('./it/train.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./koran/train.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./law/train.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./medical/train.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./subtitles/train.full.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

de.close()


en=open('joint_dev.en','w')

with open('./it/dev.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./koran/dev.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./law/dev.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./medical/dev.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./subtitles/dev.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

en.close()


de=open('joint_dev.de','w')

with open('./it/dev.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./koran/dev.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./law/dev.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./medical/dev.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./subtitles/dev.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

de.close()


en=open('joint_test.en','w')

with open('./it/test.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./koran/test.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./law/test.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./medical/test.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

with open('./subtitles/test.en','r') as f:
    lines=f.readlines()
    for line in lines:
        en.write(line)

en.close()


de=open('joint_test.de','w')

with open('./it/test.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./koran/test.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./law/test.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./medical/test.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

with open('./subtitles/test.de','r') as f:
    lines=f.readlines()
    for line in lines:
        de.write(line)

de.close()