import errno
import os
from datetime import datetime
import random

def filecreation():
    x=str(random.randint(1,100000))
    name=str(datetime.now().strftime('%Y%m%d%M'))
    mydir = os.path.join(os.getcwd()+"/root_files",name+"uid:"+x)
    result="{}uid:{}".format(name,x)
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("error")
    return result,name,x
