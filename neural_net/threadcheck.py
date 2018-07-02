import multiprocessing as mp
import time
def fun(a,queue):
    print("here")
    queue.put(a+1)

procs = [None]*4
queue = mp.Queue()
l = []
for i in range(4):
    procs[i] = mp.Process(target= fun,  args= (i,queue))

for i in range(4):
    procs[i].start()

# while True:
#     c=0
#     for i in range(4):
#         if(procs[i].is_alive()):
#             c+=1
#     if(c==0):
#         break

for i in range(4):
    procs[i].join()


print(queue)
while not queue.empty():
    print(queue.get())


for oi in range(0,len(train_files),num_threads):
    if(oi+7>=len(train_files)):
        break
    mananger = mp.Manager()
    queue = mananger.Queue()
    procs = [None]*num_threads
    for ti in range(oi,oi+num_threads):
        try:
            fc+=1
            t_file = train_files[ti]
            t_file = str(t_file).split('.ds.bz2')[0]
            print(t_file)
            try:
                procs[ti-oi] = mp.Process(target = trainer.Train, args = (queue,t_file,bz2_input_folder,'least_edge_first'))
            except Exception as e:
                pass

        except Exception as e:
            continue
    for i in range(num_threads):
        procs[i].start()
    for i in range(num_threads):
        procs[i].join()

    while not queue.empty():
        dLdOut, lnd, featVMat, _debug = queue.get()
        trainer.neuralnet.Back_Prop(dLdOut, lnd, featVMat, _debug)
