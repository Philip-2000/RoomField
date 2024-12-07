def parse():  #Optimizing!!!
    import argparse,sys
    parser = argparse.ArgumentParser(prog='Optimizing!!!')
    parser.add_argument("-l","--list") #list is prior than cnt
    return parser.parse_args(sys.argv[1:])
    
if __name__ == "__main__":
    args = parse()

"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import torch
import random
import os
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from motion_guidance import motion_guidance
visual = True

class A():
    def get(self,nm,va):
        return va

class plotSimple():
    def __init__(self, states, batchsz):
        self.batchsz = batchsz
        self.states = states
        self.realObjs = []
        self.realWalls = []

    def storeRealValues(self, realWalls, realObjs):
        self.realWalls = realWalls
        self.realObjs = realObjs

    def flatten(self,stateid,batchid):
        return stateid*self.batchsz + batchid+1

    def plotWall(self,contour,stateid,batchid):
        plt.figure(self.flatten(stateid,batchid))
        bound = self.realWalls[batchid] #if stateid < 3 else 1
        #raise NotImplementedError
        #if stateid==1 and batchid==1 and objid==0:
            #print(contour[:self.realWalls[batchid]+1])
        #print("contour")
        #print(contour.shape)
        plt.plot(contour[:bound+1,1], contour[:bound+1,0],marker="o", color="blue")
        plt.plot([contour[bound,1],contour[0,1]], [contour[bound,0], contour[0,0]], marker="o", color="blue")
        P = abs(contour).max()
        # print('contour shape:  ', contour.shape)
        plt.xlim(-P,P)
        plt.ylim(-P,P)
        plt.axis('square') #plt.show()

    def plotWalls(self,contour,stateid):
        for batchid in range(self.batchsz):
            self.plotWall(contour[batchid],stateid,batchid)

    def plotObj(self,objs,stateid,batchid):
        plt.figure(self.flatten(stateid,batchid))
        # print('obj shape: ', objs.shape)
        tr = objs[:,:3]
        bb = objs[:,3:6]
        an = objs[:,6:8]

        for i in range(self.realObjs[batchid]):
            t = tr[i]
            b = bb[i]
            a = an[i]
            # print(bb[i][0])
            # print(bb[i][2])
            # print(a[1])
            # print(a[0])
            # raise NotImplementedError

            CenterZ = torch.tensor([t[2]]).repeat(5)#np.repeat([t[2]],5)#,t[2]+ce[2],t[2]+ce[2],t[2]+ce[2],t[2]+ce[2]]
            CenterX = torch.tensor([t[0]]).repeat(5)#np.repeat([t[0]],5)#[t[0]+ce[0],t[0]+ce[0],t[0]+ce[0],t[0]+ce[0],t[0]+ce[0]]
            CornerOriginalZ = torch.tensor([b[2], b[2],-b[2],-b[2], b[2]])#np.array([b[2], b[2],-b[2],-b[2], b[2]])
            CornerOriginalX = torch.tensor([b[0],-b[0],-b[0], b[0], b[0]])#np.array([b[0],-b[0],-b[0], b[0], b[0]])

            normZ = a[0]#np.cos(a[0])
            normX = a[1]#np.sin(a[0])

            z2z = torch.tensor([ normZ]).repeat(5)# np.repeat(normZ,5)
            z2x = torch.tensor([ normX]).repeat(5)# np.repeat(normX,5)
            x2z = torch.tensor([-normX]).repeat(5)#-np.repeat(normX,5)
            x2x = torch.tensor([ normZ]).repeat(5)# np.repeat(normZ,5)

            CornerRealZ = CornerOriginalZ*z2z + CornerOriginalX*x2z
            CornerRealX = CornerOriginalZ*z2x + CornerOriginalX*x2x

            realZ = CenterZ + CornerRealZ
            realX = CenterX + CornerRealX

            plt.plot( realZ, realX, marker="*")
            plt.plot([CenterZ[0], CenterZ[0]+0.5*normZ], [CenterX[0], CenterX[0]+0.5*normX], marker="x")

    def plotObjs(self,objs,stateid):
        for batchid in range(self.batchsz):
            self.plotObj(objs[batchid],stateid,batchid)

    def plotFields(self, RELAYL, walltensor, stateid):
        for batchid in range(self.batchsz):
            fig = plt.figure(self.flatten(stateid, batchid))
            ax = fig.axes[0]
            ax.autoscale(True)
            fieldPlotter(RELAYL, walltensor, ax, batchid)
            ax.margins(0, tight=True)
    
    def plotOver(self,stateid):
        for batchid in range(self.batchsz):
            plt.figure(self.flatten(stateid,batchid))
        
            plt.savefig("./imgs/bid={} sid={}.png".format(batchid,stateid), bbox_inches = 'tight')
            # plt.clf()


def plotObj(objs):
    # print('obj shape: ', objs.shape)
    tr = objs[:,:3]
    bb = objs[:,3:6]
    an = objs[:,6:8]

    for i in range(objs.shape[0]):
        a = an[i]
        t = (tr[i] + torch.tensor([a[1]*2*bb[i][2],0,a[0]*2*bb[i][2]])) / 0.025 #tr[i]/0.025#
        b = (bb[i]*torch.tensor([1,1,3])) / 0.025 #bb[i]/0.025#

        CenterZ = torch.tensor([t[2]]).repeat(5)#np.repeat([t[2]],5)#,t[2]+ce[2],t[2]+ce[2],t[2]+ce[2],t[2]+ce[2]]
        CenterX = torch.tensor([t[0]]).repeat(5)#np.repeat([t[0]],5)#[t[0]+ce[0],t[0]+ce[0],t[0]+ce[0],t[0]+ce[0],t[0]+ce[0]]
        CornerOriginalZ = torch.tensor([b[2], b[2],-b[2],-b[2], b[2]])#np.array([b[2], b[2],-b[2],-b[2], b[2]])
        CornerOriginalX = torch.tensor([b[0],-b[0],-b[0], b[0], b[0]])#np.array([b[0],-b[0],-b[0], b[0], b[0]])

        normZ = a[0]#np.cos(a[0])
        normX = a[1]#np.sin(a[0])

        z2z = torch.tensor([ normZ]).repeat(5)# np.repeat(normZ,5)
        z2x = torch.tensor([ normX]).repeat(5)# np.repeat(normX,5)
        x2z = torch.tensor([-normX]).repeat(5)#-np.repeat(normX,5)
        x2x = torch.tensor([ normZ]).repeat(5)# np.repeat(normZ,5)

        CornerRealZ = CornerOriginalZ*z2z + CornerOriginalX*x2z
        CornerRealX = CornerOriginalZ*z2x + CornerOriginalX*x2x

        realZ = CenterZ + CornerRealZ
        realX = CenterX + CornerRealX

        plt.plot( realZ, realX, marker="*")
        plt.plot([CenterZ[0], CenterZ[0]+20*normZ], [CenterX[0], CenterX[0]+20*normX], marker="x")



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

ts = [0.1 * steps for steps in range(11)] + [1.0, 1.0, 1.0]
#print('ts:  ', ts)
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
    betas = [3 for _ in range(1000)]
    RELAYL = motion_guidance(A(), betas=betas)
    PLOTTER = plotSimple(len(ts),RELAYL.batchsz)
    realObjs = [RELAYL.maxObj for _ in range(RELAYL.batchsz)]
    realWalls = [RELAYL.maxWall for _ in range(RELAYL.batchsz)]
    pt = torch.tensor(torch.load(name)[0])
    pt = torch.cat([pt[:,-7:-1],torch.cos(pt[:,-1:]),torch.sin(pt[:,-1:]),pt[:,:-7],torch.zeros_like(pt[:,-1:])],axis=-1)
    other = torch.zeros_like(pt[0])
    other[-1] = 1
    other[6] = 1
        
    realObjs[0] = pt.shape[0]
    other = other.reshape((1,-1)).repeat((RELAYL.maxObj-pt.shape[0],1))
    absolute = torch.cat([pt,other],axis=0).reshape((1,RELAYL.maxObj,-1))
    contour = torch.tensor(np.load(wallName, allow_pickle=True)["contour"])
    mid = torch.tensor([contour[:,0].max() + contour[:,0].min(),contour[:,1].max() + contour[:,1].min()])/2.0
    wallTensor = torch.cat([contour[:,:2]-mid.reshape((1,-1)),contour[:,2:]],axis=-1)
    realWalls[0] = wallTensor.shape[0]
    wallTensor = torch.cat([wallTensor,wallTensor[-1:,:].repeat((RELAYL.maxWall-wallTensor.shape[0],1))],axis=0)
    wallTensor = torch.tensor(wallTensor).reshape((1,RELAYL.maxWall,-1))
    PLOTTER.storeRealValues(realWalls, realObjs)
    return RELAYL, absolute, wallTensor, PLOTTER

def fieldPlotter(RELAYL, wallTensor, absolute=None, widos=None, ax = None, id=None):
    lst = []
    #print(wallTensor)
    halfSize = 150#3
    scale = 0.025#1
    rate=5#2
    maxLen = 5
    for _ in range(RELAYL.batchsz):
        lst.append([])
        for i in range(-halfSize,halfSize,rate):
            for j in range(-halfSize,halfSize,rate):
                lst[_].append([i*scale,j*scale])
    #fields = RELAYL.field(torch.Tensor(lst).reshape((RELAYL.batchsz, -1, RELAYL.sample_dim, 2)), wallTensor).reshape((RELAYL.batchsz, int(2*halfSize/rate), int(2*halfSize/rate), -1))
    fields = RELAYL.printField(torch.Tensor(lst).reshape((RELAYL.batchsz, -1, 2)), wallTensor, absolute, widos).reshape((RELAYL.batchsz, int(2*halfSize/rate), int(2*halfSize/rate), -1))

    width = 2*halfSize
    height = 2*halfSize
    background = (0, 0, 0, 255)
    # figure = plt.figure(figsize=(width,height))
    X = np.arange(-halfSize, halfSize, rate)
    Y = np.arange(-halfSize, halfSize, rate)
    if ax is None:
        figure, ax = plt.subplots()
    else:
        X = X * scale
        Y = Y * scale
    # print(X.shape, Y.shape)
    U = fields[:, :, :, 0] #* 1.2
    V = fields[:, :, :, 1] #* 1.2
    hypot = np.hypot(U,V)
    # print(hypot.shape)
    # print(U.shape, V.shape)
    # ax.quiver(1,2,10,20)
    # plt.show()
    # print(fields[0, 0, 0 ,0], fields[0, 0, 0, 1])
    # print(fields[0, int(2*halfSize/rate)-1, 0 ,0], fields[0, int(2*halfSize/rate)-1, 0, 1])
    # print(fields[0, 0, int(2*halfSize/rate)-1 ,0], fields[0, 0, int(2*halfSize/rate)-1, 1])
    # print(fields[0, int(2*halfSize/rate)-1, int(2*halfSize/rate)-1 ,0], fields[0, int(2*halfSize/rate)-1, int(2*halfSize/rate)-1, 1])
    cmap = plt.get_cmap('gist_heat_r')
    # test_ten = torch.full_like(hypot, fill_value = 255.0)
    # for i in range(hypot.shape[0]):
    #     for _ in range(hypot.shape[1]):
    #         test_ten[i][_] -= (5*(_+1))
    # torch.set_printoptions(threshold = 1601)
    # print(test_ten[0])
    if id is None:
        for _ in range(RELAYL.batchsz):
            # print(U[_].shape)
            Q = ax.quiver(X, Y, V[_], U[_], hypot[_], scale = 70, cmap=cmap)
            plt.plot(wallTensor[_][:,1]/scale, wallTensor[_][:,0]/scale,marker="o", color="blue")
            plt.plot([wallTensor[_][-1,1]/scale,wallTensor[_][0,1]/scale], [wallTensor[_][-1,0]/scale, wallTensor[_][0,0]/scale], marker="o", color="blue")
            #plotObj(absolute[_][0:4])
            plt.savefig("./imgs/"+str(_)+'.png')
            plt.cla()
    else:
        ax.quiver(X, Y, V[id], U[id], hypot[id], scale = 70, cmap=cmap)

def dataNoiser(RELAYL, absolute, PLOTTER):
    for i in range(RELAYL.batchsz):
        for j in range(PLOTTER.realObjs[i]):
            if j%2 == 1:
                continue
            absolute[i][j][0] -= 1.0 * absolute[i][j][7]
            absolute[i][j][2] -= 1.0 * absolute[i][j][6]
            deg = random.random() * math.pi / 4
            ori_deg = math.acos(absolute[i][j][6])
            absolute[i][j][6] = abs(math.cos(ori_deg + deg)) * torch.sign(absolute[i][j][6])
            absolute[i][j][7] = abs(math.sin(ori_deg + deg)) * torch.sign(absolute[i][j][7])
            absolute[i][j][3] *= 1.1
            absolute[i][j][5] *= 1.5
    return absolute

def guidancePlotter(RELAYL, absolute, wallTensor, PLOTTER, widos):
    global ts
    absolute = dataNoiser(RELAYL, absolute, PLOTTER)
    # print('absolute shape: ', absolute.shape)
    result = absolute
    for ti in range(len(ts)):
        #result= RELAYL.full(result, wallTensor, torch.Tensor([ts[ti]]).reshape(1,1).repeat(RELAYL.batchsz,1))
        if ti > 0:
            result= RELAYL.fulls(result, wallTensor, torch.Tensor([ts[ti]]).reshape(1,1).repeat(RELAYL.batchsz,1), widos)
        PLOTTER.plotWalls(wallTensor,ti)
        PLOTTER.plotObjs(result,ti)
        #PLOTTER.plotFields(RELAYL, wallTensor, ti)
        PLOTTER.plotOver(ti)
        plt.close('all')

def potentialPlotter(RELAYL, absolute, wallTensor):
    lst = []
    halfSize = 400#3
    scale = 0.025#1
    rate=5#2
    for _ in range(RELAYL.batchsz):
        lst.append([])
        for i in range(-halfSize,halfSize+1,rate):
            for j in range(-halfSize,halfSize+1,rate):
                lst[_].append([i*scale,j*scale])
    #fields = RELAYL.field(torch.Tensor(lst).reshape((RELAYL.batchsz, -1, RELAYL.sample_dim, 2)), wallTensor).reshape((RELAYL.batchsz, int(2*halfSize/rate), int(2*halfSize/rate), -1))
    field = RELAYL.printPotential(torch.Tensor(lst).reshape((RELAYL.batchsz, -1, 2)), wallTensor, int(halfSize/rate), absolute)

    
    # # 创建一个三维网格
    x = np.linspace(-halfSize, halfSize+1, int(2*halfSize/rate)+1)*scale
    y = np.linspace(-halfSize, halfSize+1, int(2*halfSize/rate)+1)*scale
    x, y = np.meshgrid(x, y)
    
    # # 定义一个高度函数
    # z = (1 - x/2 + x**5 + y**3) * np.exp(-x**2 - y**2)
    z = np.array(-field[3])
    
    # 创建一个新的图像
    fig = plt.figure()
    
    # 创建一个3D绘图区域
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制三维曲面
    surf = ax.plot_surface(x, y, z, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    
    # 在曲面上绘制等高线
    # 设置等高线的高度
    levels = np.linspace(0, 10, 5)**2
    
    # 绘制等高线
    ax.contour(x, y, z, levels, zdir='z', offset=0, cmap=plt.cm.coolwarm)
    
    # 设置图形的标签和标题
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    #ax.set_zlabel('Z axis')
    ax.set_title('00c36b04-369f-4df1-9db1-b29913d2c51f_LivingDiningRoom-7050')
    
    # 显示图形
    
    # for angle in range(0, 360):
    #     ax.view_init(30, angle)
    #     plt.draw()
    #     plt.pause(0.1)
    plt.show()

def update(frame):
    # 更新数据
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x + frame/10.0)  # 根据当前帧更新y值
    plt.cla()                   # 清除上一帧的图像
    plt.plot(x, y)              # 绘制新的图像
    plt.axis([0, 2*np.pi, -1, 1])
    plt.title(f'Frame: {frame}')

if __name__ == "__main__":
#     fig = plt.figure()
#     ani = animation.FuncAnimation(fig, update, frames=50, interval=50)
#     Writer = animation.writers['ffmpeg']
#     writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
#     ani.save('sine_wave_video.mp4', writer=writer)# More at https://blog.csdn.net/weixin_44274609/article/details/136577650
# else:
    a, b, c, d, e = dataLoader()
    #guidancePlotter(a, b, c, d, e)
    fieldPlotter(a, c, b, e)
    #potentialPlotter(a, b, c)
    # a, b, c, d = dataLoaderPT("./SecondBedroom-11669_191_000_time0395.pt","./SecondBedroom-11669 contours.npz")
    # guidancePlotter(a, b, c, d)
"""