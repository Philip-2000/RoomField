class Images():
    def __init__():
        pass
        #并行编程结果的可视化也很特殊，
        #因为一整个批的信息是一起处理的，但是在可视化的过程中需要抽丝剥茧，把他们都打开
        #另外尤其需要注意的是，我们需要有效地管理这些图片的文件结构以及合成视频

        #搞笑，不想写了，傻逼。

    def __call__(self, scneTensor):

        pass

"""
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

        

"""