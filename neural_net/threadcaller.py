import threading,sys
from edge1500generate import *


s = int(sys.argv[1])-1

s = s*20000



# defining threads
t1 = threading.Thread(target=threadfun, args=(s+3000,s+5000,1))
t2 = threading.Thread(target=threadfun, args=(s+5000+3000,s+10000,2)) 
t3 = threading.Thread(target=threadfun, args=(s+10000+3000,s+15000,3))
t4 = threading.Thread(target=threadfun, args=(s+15000+3000,min(119503,s+20000),4)) 
# t5 = threading.Thread(target=threadfun, args=(s+4000,s+5000,5))
# t6 = threading.Thread(target=threadfun, args=(s+5000,s+6000,6)) 
# t7 = threading.Thread(target=threadfun, args=(s+6000,s+7000,7))
# t8 = threading.Thread(target=threadfun, args=(s+7000,s+8000,8)) 
# t9 = threading.Thread(target=threadfun, args=(s+8000,s+9000,9))
# t10 = threading.Thread(target=threadfun, args=(s+9000,min(119503,s+10000),10)) 


# starting threads
t1.start()
t2.start()
t3.start()
t4.start()
# t5.start()
# t6.start()
# t7.start()
# t8.start()
# t9.start()
# t10.start()

# wait until all threads finish
t1.join()
t2.join()
t3.join()
t4.join()
# t5.join()
# t6.join()
# t7.join()
# t8.join()
# t9.join()
# t10.join()

print("All threads done on files : {} - {}".format(s,min(119503,s+10000)))
