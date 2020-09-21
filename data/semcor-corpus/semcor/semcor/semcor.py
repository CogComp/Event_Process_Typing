# add quotes around XML attributes, rename files with ".xml" extension

import sys
import re

for f in sys.argv[1:]:
    c = open(f).read()
    c = re.sub(r'&', r'&amp;', c)
    for i in range(10):
        c = re.sub(r'(<[^>]+=)([^">]+)([ >])', r'\1"\2"\3', c)
    f2 = open(f + ".xml", "w")
    f2.write(c)
    f2.close()






