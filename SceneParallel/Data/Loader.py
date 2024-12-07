class scneTensor():
    def __init__(self,lst,batchsz=-1,maxObj=-1,maxWall=-1,format={"obj":[("t",3),("s",3),("o",2),("c",1)],"wal":["tx","tz","nz","nx"]}):
        from SceneClasses.Basic.Scne import scneDs
        import torch
        self.dataset = scneDs(lst=["a"])

        self.batchsz = len(self.dataset)
        self.maxObj = maxObj  if maxObj > 0 else max([len(s.OBJES) for s in self.dataset])
        self.maxWall= maxWall if maxWall> 0 else max([len(s.WALLS) for s in self.dataset])
        self.objTensor = torch.cat([s.OBJES.toTensor() for s in self.dataset],axis=0)
        self.walTensor = torch.cat([s.WALLS.toTensor() for s in self.dataset],axis=0)
        self.widTensor = torch.cat([s.WALLS.windoors.toTensor() for s in self.dataset],axis=0)


        #然后的话就是说，我的各大tensor都是基于self.dataset的数据给出的
        #经过每个时间步之后，还需要对self.dataset中的信息进行调整
        #调整之后，还是使用scne固有的可视化方法去可视化此场景



        #还有一个问题就是，对他进行“实验”或“使用”的时候，是怎么做？
        #我觉得在“非可视化”模式之下，需要支持我们将tensor都提取出来之后，销毁self.dataset对象，
        #然后在所有时间片全部结束之后，再将self.dataset重构出来，然后对这个最终结果进行可视化


        pass


    def upd(self):
        for i,s in enumerate(self.dataset):
            s.OBJES.fromTensor(self.objTensor[i]) #s.WALLS.fromTensor(self.walTensor[i]) #s.WALLS.windoors.fromTensor(self.widTensor[i])

    def tensor(self):
        pass

    #我觉得关键就是怎么配置，怎么利用一个过程把他们的都穿起来，

"""

def givename(dir, id=0):
    name = os.listdir(dir)[id]

    #print(name)
    contour = np.load(dir + name + "/contours.npz", allow_pickle=True)["contour"]
    # if visual:
    #     print("contour")
    #     print(contour.shape)
    cont = np.load(dir + name + "/conts.npz", allow_pickle=True)["cont"]
    # if visual:
    #     print("cont")
    #     print(cont.shape)
    boxes =  np.load(dir + name + "/boxes.npz", allow_pickle=True)
    # if visual:
    #     print("boxes[translations]")
    #     print(boxes["translations"].shape)
    # if visual:
    #     print("boxes[sizes]")
    #     print(boxes["sizes"].shape)
    # if visual:
    #     print("boxes[angles]")
    #     print(boxes["angles"].shape)
    boxess = {}
    for k in ["translations", "sizes", "angles", "floor_plan_centroid", "class_labels"]:
        boxess[k] = torch.tensor(boxes[k])
    return torch.tensor(contour), torch.tensor(cont), boxess, name, id

def dataLoader(dir="../../lab/"):
    global ts
    betas = [3 for _ in range(1000)]
    RELAYL = motion_guidance(A(), betas=betas, widoAsObj=True)
    PLOTTER = plotSimple(len(ts),RELAYL.batchsz)
    absoluteTensor = torch.zeros(size = (0,RELAYL.maxObj,RELAYL.bbox_dim+RELAYL.class_dim+1))
    wallTensor = torch.zeros(size = (0,RELAYL.maxWall,RELAYL.wall_dim))
    widoTensor = torch.zeros(size = (0,RELAYL.maxWidos,RELAYL.wido_dim))
    realObjs = [RELAYL.maxObj for _ in range(RELAYL.batchsz)]
    realWalls = [RELAYL.maxWall for _ in range(RELAYL.batchsz)]
    
    ID =-1 
    for _ in range(RELAYL.batchsz):
        ID += 1
        contour, cont, boxes, name, ID = givename(dir, ID)
        contour = torch.cat([contour[:,:-2], contour[:,-1:], contour[:,-2:-1]],axis=-1)
        realWalls[_] = len(contour)
        cen = boxes["floor_plan_centroid"].reshape((1,-1))
        #singleWa = torch.cat( [ contour[:,:1]-cen[:,:1], contour[:,1:2]-cen[:,2:3], contour[:,2:] ], axis = -1)
        singleWa = torch.cat( [ contour[:,:1]-cen[:,:1], contour[:,1:2]-cen[:,2:3], contour[:,3:4] , contour[:,2:3] ], axis = -1)
        swallTensor = torch.cat([singleWa, singleWa[-1:].repeat( (RELAYL.maxWall - realWalls[_], 1) ) ], axis = 0)
        # print("swallTensor")
        # print(swallTensor.shape)
        wallTensor = torch.cat([wallTensor, swallTensor.reshape(1,RELAYL.maxWall,-1)], axis=0)
        #wallTensor, from where, reading -> padding -> concatenating

        realObjs[_] = len(boxes["translations"]) + len(cont)#

        boxes["class_labels"] = torch.cat([boxes["class_labels"], torch.zeros((len(boxes["translations"]),2))],axis=-1)
        currentTensor = torch.cat([boxes["translations"], boxes["sizes"], torch.cos(boxes["angles"]), torch.sin(boxes["angles"]), boxes["class_labels"]],axis=-1)
        #print(boxes["class_labels"].shape)
        #tr = boxes["translations"]
        #bb = boxes["sizes"]
        #an = boxes["angle"]
        #absoluteTensor, from where???
        widosTensor = torch.cat([cont[:,:3]-cen[:,:], torch.tensor([max(cont[i][3],cont[i][5]) for i in range(cont.shape[0])]).reshape((cont.shape[0],1)), cont[:,4:5], torch.tensor([min(cont[i][3],cont[i][5]) for i in range(cont.shape[0])]).reshape((cont.shape[0],1)), cont[:,6:8], torch.zeros((cont.shape[0],RELAYL.class_dim))], axis=-1)
        for k in range(cont.shape[0]):
            if cont[k][1]-cont[k][4] < 0.1:
                widosTensor[k][-1] = 1
            else:
                widosTensor[k][-2] = 1
        swidoTensor = torch.cat([cont[:,:3]-cen[:,:], cont[:,3:]],axis=-1)
        swidoTensor = torch.cat([swidoTensor,torch.zeros((RELAYL.maxWidos-swidoTensor.shape[0],RELAYL.wido_dim))],axis=0)
        widoTensor = torch.cat([widoTensor, swidoTensor.reshape(1,RELAYL.maxWidos,-1)], axis=0)

        currentTensor = torch.cat([currentTensor, widosTensor],axis=0)

        #print("currentTensor")
        #print(currentTensor.shape)
        #print(swallTensor)
        
        sabsoluteTensor = np.vstack([currentTensor, np.tile(np.zeros(currentTensor.shape[-1])[None, :], [RELAYL.maxObj - realObjs[_], 1])])
        sabsoluteTensor = torch.tensor(sabsoluteTensor)
        #print("sabsoluteTensor")
        #print(sabsoluteTensor.shape)
        objectness = torch.cat([torch.zeros((realObjs[_])), torch.ones((RELAYL.maxObj - realObjs[_]))], axis=-1).reshape(RELAYL.maxObj,1)
        #print("objectness")
        #print(objectness.shape)
        sabsoluteTensor = torch.cat([sabsoluteTensor, objectness], axis=-1)
        absoluteTensor = torch.cat([absoluteTensor, sabsoluteTensor.reshape(1,RELAYL.maxObj,-1)], axis=0)

    PLOTTER.storeRealValues(realWalls, realObjs)
    # print("realObjs")
    # print(realObjs)
    # print("realWalls")
    # print(realWalls)
    #choose some scenes:
    #form the absoluteTensor, wallTensor, windoorTensor
    #throw into the rela

    # print("absoluteTensor")
    # print(absoluteTensor.shape)
    # print("wallTensor")
    # print(wallTensor.shape)
    return RELAYL, absoluteTensor, wallTensor, PLOTTER, widoTensor

    #RELAYL.rela(absoluteTensor,wallTensor,windoorTensor)
    #forRelaRecon(RELAYL,absoluteTensor,wallTensor,windoorTensor)
    #RELAYL.visualizer.plotOver()

def dataLoaderPT(name,wallName):
    pt = torch.tensor(torch.load(name)[0])
    pt = torch.cat([pt[:,-7:-1],torch.cos(pt[:,-1:]),torch.sin(pt[:,-1:]),pt[:,:-7],torch.zeros_like(pt[:,-1:])],axis=-1)
    other = torch.zeros_like(pt[0])
    other[-1] = 1
    other[6] = 1
    other = other.reshape((1,-1)).repeat((RELAYL.maxObj-pt.shape[0],1))
    absolute = torch.cat([pt,other],axis=0).reshape((1,RELAYL.maxObj,-1))

"""