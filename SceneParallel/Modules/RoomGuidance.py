import torch
import torch.nn.functional as F

def crs(A,B):
    return A[:,:,:,0]*B[:,:,:,1]-A[:,:,:,1]*B[:,:,:,0]

object_area = torch.Tensor([
    [[.0,.0],[.01,.01]],#pendant_lamp
    [[.0,.0],[.01,.01]],#ceiling_lamp
    [[.2,.0],[1., .9]],#bookshelf
    [[.0,.0],[.5, .5]],#round_end_table
    [[.0,.0],[.5, .5]],#dining_table
    [[.0,.0],[.9, .9]],#console_table
    [[.0,.0],[.9, .9]],#corner_side_table
    [[.0,.0],[.9, .9]],#desk
    [[.0,.0],[1.,1.1]],#coffee_table
    [[.0,.0],[.9, .9]],#dressing_table
    [[.2,.0],[1., .9]],#children_cabinet
    [[.2,.0],[1., .9]],#cabinet
    [[.2,.0],[1., .9]],#shelf
    [[.2,.0],[1., .9]],#wine_cabinet
    [[.0,.0],[.5, .5]],#lounge_chair
    [[.0,.0],[.5, .5]],#chinese_chair
    [[.0,.0],[.5, .5]],#dressing_chair
    [[.0,.0],[.5, .5]],#dining_chair
    [[.0,.0],[.5, .5]],#armchair
    [[.0,.0],[.5, .5]],#barstool
    [[.0,.0],[.5, .5]],#stool
    [[.2,.0],[1., .9]],#multi_seat_sofa
    [[.2,.0],[1., .9]],#loveseat_sofa
    [[.0,.0],[.6, .9]],#l_shaped_sofa
    [[.2,.0],[1., .9]],#lazy_sofa
    [[.2,.0],[1., .9]],#chaise_longue_sofa
    [[.2,.0],[1., .9]],#wardrobe
    [[.2,.0],[1., .9]],#tv_stand
    [[.0,.0],[.5, .5]],#nightstand
    [[.2,.0],[1.,1.2]],#double_bed
    [[.2,.0],[1.,1.2]],#kids_bed
    [[.2,.0],[1.,1.2]],#bunk_bed
    [[.2,.0],[1.,1.2]],#single_bed
    [[.2,.0],[1.,1.2]],#bed_frame
    [[.0,.0],[.01,.01]],#window
    [[.0,.0],[.01,.01]],#door
])
#[[.0,.0],[.01,.01]]:floating objects, not interactive
#[[.0,.0],[.9, .9]]: ordinary objects
#[[.2,.0],[1., .9]]: wardrobe
#[[.2,.0],[1.,1.2]]: bed
#[[.0,.0],[1.,1.2]]: coffee table

class RoomGuidance():
    def __init__(self, config, betas, scaled=False, minimum=torch.tensor([-6.752, -0.185, -6.573, 0.275, 0.020, 0.0005]), maximum=torch.tensor([6.634, 3.204, 6.8265, 2.6445, 2.789, 2.259]),widoAsObj=False):
        #all kinds of dimensions here
        self.batchsz = config.get("batchsz", 8)

        #................basic length of the data structure...................
            #.........absoluteTensor.....................
        self.translation_dim = config.get("translation_dim", 3)
        self.size_dim = config.get("size_dim", 3)
        self.angle_dim = config.get("angle_dim", 2)
        self.bbox_dim = self.translation_dim + self.size_dim + self.angle_dim
        self.class_dim = 34+2
        self.maxObj = config.get("sample_num_points", 21)

            #.........wallTensor.....................
        self.process_wall = config.get("process_wall", False)
        self.wall_translation_dim = config.get("wall_translation_dim", 2)
        self.wall_norm_dim = config.get("wall_norm_dim",2)
        self.wall_dim = self.wall_translation_dim + self.wall_norm_dim
        self.maxWall = config.get("maxWall", 16)
        self.maxWidos = config.get("maxWidos", 8)
        self.wido_dim = config.get("wido_dim", 8)

        #^^^^^^^^^^^^^^^^^^^^^basic length of the data structure^^^^^^^^^^^^^^^^^^^

        if self.angle_dim != 2 or self.wall_norm_dim != 2:
            raise NotImplementedError

        self.betas = betas
        dv = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
        self.temp = torch.Tensor([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]).to(dv)
        self.tempp= torch.Tensor([[-1,1],[-1,0],[-1,-1]]).to(dv)  
        self.sample_dim = self.temp.shape[0]
        self.border = 0.2
        self.maxT = 0.8
        self.mats = None
        if scaled:
            self.minimum = minimum.to(dv).reshape((1,1,6)).repeat((self.batchsz, self.maxObj, 1))
            self.maximum = maximum.to(dv).reshape((1,1,6)).repeat((self.batchsz, self.maxObj, 1))
        self.scaled = scaled
        self.fieldTest = False
        self.widoAsObj = widoAsObj#raise NotImplementedError
    
    #region: -hyper-parameter--------#
    def tArrangement(self,t):
        return (torch.ones_like(t) * self.maxT).reshape((-1,1)) if t[0][0] > 0 else torch.zeros_like(t).reshape((-1,1))
        return (self.maxT*(1. - t / float(len(self.betas)))).reshape((-1,1))
    #endregion: -hyper-parameter-----#

    #region: ----room-field----------#

        #region: -----door-field-----#
    def doorIs(self, widos):
        h = widos[:,:,1] - widos[:,:,4]#widos: batchsz = 128 : maxWidos = 8 : location_dim = 8
        V = torch.logical_and(torch.abs(widos[:,:,3])>torch.ones_like(h)*0.01,torch.abs(widos[:,:,5])>torch.ones_like(h)*0.01)
        return torch.logical_and((h < torch.ones_like(h)*0.1),V)

    def doorFields(self, locations, widos):
        B,O,W,L=self.batchsz,self.maxObj*self.temp.shape[0],self.maxWidos,2
        if self.fieldTest:
            O = locations.shape[1]
        #locations: batchsz = 128 : maxObj*sample_dim = 96 : widos_dim = 1 : location_dim = 2
        #widos: batchsz = 128 : maxWidos = 8 : location_dim = 8

        RT,OT = 3,0.1
        widosTr = torch.cat([widos[:,:,0:1],widos[:,:,2:3]],axis=-1).reshape((B,1,W,L))
        widosSz = torch.cat([widos[:,:,3:4],widos[:,:,5:6]],axis=-1).reshape((B,1,W,L))
        widosDr = torch.cat([widos[:,:,-1:],widos[:,:,-2:-1]],axis=-1).reshape((B,1,W,L))
        widosSz+= torch.abs(widosSz * widosDr * (RT-1))
        widosTr+= widosSz * widosDr #* (RT+1)/RT

        LF = widosDr*widosSz + widosTr + torch.cat([widosDr[:,:,:,1:2],widosDr[:,:,:,0:1]],axis=-1)*widosSz*(OT*2+1)
        RG = widosDr*widosSz + widosTr - torch.cat([widosDr[:,:,:,1:2],widosDr[:,:,:,0:1]],axis=-1)*widosSz*(OT*2+1)
        
        lf = LF-locations.reshape((B,O,1,L))
        rg = RG-locations.reshape((B,O,1,L))
        #lf/rg: batchsz = 128 : maxObj*sample_dim = 96 : widos_dim = 8 : location_dim = 2
        cond = (lf**2).sum(axis=-1) > (rg**2).sum(axis=-1)
        #cond: batchsz = 128 : maxObj*sample_dim = 96 : widos_dim = 8
        lf[cond] = rg[cond]
        
        normed = (widosTr-locations.reshape((B,O,1,L)))/widosSz
        #normed: batchsz = 128 : maxObj*sample_dim = 96 : widos_dim = 8 : location_dim = 2
        x = torch.ones_like(normed[:,:,:,1]) + normed[:,:,:,1]*OT*widosTr[:,:,:,1]>torch.abs(normed[:,:,:,0])
        z = torch.ones_like(normed[:,:,:,0]) + normed[:,:,:,0]*OT*widosTr[:,:,:,0]>torch.abs(normed[:,:,:,1])
        inDoor = torch.logical_and(x,z)
        f = torch.zeros_like(lf)
        f[inDoor] = lf[inDoor]
        #return f
        
        g = torch.zeros_like(f)
        #f: batchsz = 128 : maxObj*sample_dim = 96 : maxWidos = 8 : field_dim = 2
        cond = self.doorIs(widos).reshape((B,1,W,1)).repeat((1,O,1,L))
        #doorIs: batchsz = 128 : maxWidos = 8
        g[cond] = f[cond]
        return g.sum(axis=-2)*1.5
        #endregion: -----door-field--#

        #region: ---object-field-----#
    def invalidObject(self,absolute):
        B,O,L=self.batchsz,self.maxObj,1
        zeo = torch.zeros_like(absolute[:,:,-1])
        #cond = torch.logical_or(torch.logical_or((absolute[:,:,-1] > zeo),(absolute[:,:,self.bbox_dim+3] > zeo)),(absolute[:,:,-9] > zeo))
        cond = torch.logical_or(torch.logical_or((absolute[:,:,-1] > zeo),(absolute[:,:,self.bbox_dim] > zeo)),(absolute[:,:,self.bbox_dim+1] > zeo))
        #cond: batchsz = 128 : maxObj = 16 : sig = 1
        if self.fieldTest:
            return cond.reshape((B,1,O,L))
        condx = cond.reshape((B,1,O,L))#print(condx[0,0,:,0])
        condy = cond.reshape((B,O,1,1,L)).repeat((1,1,self.temp.shape[0],1,1)).reshape((B,-1,1,L))
        return torch.logical_or(condx.repeat((1,O*self.temp.shape[0],1,1)),condy.repeat((1,1,O,1)))

    def invalidObj(self,absolute):
        B,O,L=self.batchsz,self.maxObj,1
        zeo = torch.zeros_like(absolute[:,:,-1])
        #cond = torch.logical_or(torch.logical_or((absolute[:,:,-1] > zeo),(absolute[:,:,self.bbox_dim+3] > zeo)),(absolute[:,:,-9] > zeo))
        cond = torch.logical_or(torch.logical_or((absolute[:,:,-1] > zeo),(absolute[:,:,self.bbox_dim] > zeo)),(absolute[:,:,self.bbox_dim+1] > zeo))
        #cond: batchsz = 128 : maxObj = 16 : sig = 1
        if self.fieldTest:
            return cond.reshape((B,1,O,L))
        condx = cond.reshape((B,1,O,L))#print(condx[0,0,:,0])
        condy = cond.reshape((B,O,1,1,L)).repeat((1,1,self.temp.shape[0],1,1)).reshape((B,-1,1,L))
        return torch.logical_or(condx.repeat((1,O*self.temp.shape[0],1,1)),condy.repeat((1,1,O,1)))

    def objectField(self, locations, absolute, mats):
        if mats is None:
            self.flattenn(absolute)
            mats = self.mats


        
        #absolute:  batchsz = 128 : maxObj = 12 : bbox_dim = 8
        center = torch.cat([absolute[:,:,0],absolute[:,:,2]],dim=-1)
        #center:  batchsz = 128 : maxObj = 12 : L = 2
        
        #self.maxObj*self.temp.shape[0]  #s = self.temp.shape[0]
        B,S,O,L=self.batchsz,locations.shape[1],self.maxObj,2


        #select from the object_area
        classId = torch.argmax(absolute[:,:,self.bbox_dim:],dim=-1)
        #classId: batchsz = 128 : maxObj = 12
        oa = object_area.reshape((1,1,-1,L*L)).repeat((B,O,1,1))
        id0= torch.arange(B).reshape((-1,1,1)).repeat(1,O,L*L)
        id1= torch.arange(O).reshape((1,-1,1)).repeat(B,1,L*L)
        id2= classId.reshape((B,O,1)).repeat(1,1,L*L)
        id3= torch.arange(L*L).reshape((1,1,-1)).repeat(B,O,1)
        res= oa[id0,id1,id2,id3]
        scl= res.reshape((B,O,L,L))[:,:,1,:]
        ofs= res.reshape((B,O,L,L))[:,:,0,:]
        #scl/ofs: batchsz = 128 : maxObj = 12 : location_dim = 2

        C = torch.cat([absolute[:,:,0:1],absolute[:,:,self.translation_dim-1:self.translation_dim]], axis=-1).reshape((B,1,O,L))
        #C: batchsz = 128 : ??=1 : maxObj = 12 : location_dim = 2
        C+= (ofs.reshape((B,-1,O,L,1)) * mats.reshape((B,1,O,L,L))).sum(axis=-2)

        sizs = torch.cat([absolute[:,:,self.translation_dim:self.translation_dim+1],absolute[:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim]], axis=-1).reshape((B,1,O,L))
        sizs*= scl.reshape((B,1,O,L))
        #sizs: batchsz = 128 : ??=1 : maxObj = 12 : location_dim = 2
        
        #mats: batchsz = 128 : maxObj = 12 : location_dim = 2 : location_dim = 2
        
        #然后是从location 到各个absolute 的中心的方向，
        #叫啥呢？叫radial吧。
        point = locations.reshape((B,-1,1,L))
        #point:  batchsz = 128 : maxObj*sample_dim = 12*8 = 96 : maxObj = 12 : L = 2
        radial = center.reshape((B,1,-1,L)) - point
        #radial:  batchsz = 128 : maxObj*sample_dim = 12*8 = 96 : maxObj = 12 : L = 2
        Point = point - C
        #Point:  batchsz = 128 : maxObj*sample_dim = 12*8 = 96 : maxObj = 12 : L = 2
        
        #下一步是将point和radial转移到C的坐标体系之下，
        #转一下然后缩一下就行了，因为C已经减过了

        #
        Point = ((point - C).reshape((B,-1,O,1,L)) * mats.reshape((B,1,O,L,L))).sum(axis=-1) / sizs
        Radial = (radial.reshape((B,-1,O,1,L)) * mats.reshape((B,1,O,L,L))).sum(axis=-1) / sizs

        #sizs: batchsz = 128 : maxObj*sample_dim = 12*8 = 96 : maxObj = 12 : location_dim = 2
        #咋做？应该有已有的代码吧？

        n2 = (Radial*Radial).sum(axis=-1)
        crs= (Radial[:,:,:,0]*Point[:,:,:,1] + Radial[:,:,:,1]*Point[:,:,:,0])**2
        n2 = torch.max(n2,crs)
        
        #把小于
        rate = torch.sqrt(n2 - (Radial[:,:,:,0]*Point[:,:,:,1] + Radial[:,:,:,1]*Point[:,:,:,0])**2 - (Radial*Point).sum(axis=-1))/n2
        field = Radial * rate.reshape((B,-1,O,1))

        

        #transform the locations into absolute's world
        # as [X,Z]

        #transform the -sp.radial into absolute's world
        #what about this?
        #how do we know which sample is from which object?



        #we have to give what?
        #do we have to give what?

        # as [A,C]

        #calculate the field with formulation in self's world

        #the result will be what?

        # √(A²+C² - (AZ+CX)²) -AX -CZ
        #----------------------------------- [A,C]
        #            A²+C²

        #as [F,H]

        #transform this field back into the world
        #as [f,h]
        #field(sp.transl,sp.radial) = [f,h]

        pass

    def objField(self, locations, absolute, mats):
        if mats is None:
            self.flattenn(absolute)
            mats = self.mats
        B,O,W,L=self.batchsz,self.maxObj*self.temp.shape[0],self.maxObj,2
        if self.fieldTest:
            O = locations.shape[1]
        #locations: batchsz = 128 : maxObj*sample_dim = 96 : obj_dim = 1 : location_dim = 2
        
        #select from the object_area
        classId = torch.argmax(absolute[:,:,self.bbox_dim:-3],dim=-1)
        #classId: batchsz = 128 : maxObj = 12
        oa = object_area.reshape((1,1,-1,L*L)).repeat((B,W,1,1))
        id0= torch.arange(B).reshape((-1,1,1)).repeat(1,W,L*L)
        id1= torch.arange(W).reshape((1,-1,1)).repeat(B,1,L*L)
        id2= classId.reshape((B,W,1)).repeat(1,1,L*L)
        id3= torch.arange(L*L).reshape((1,1,-1)).repeat(B,W,1)
        res= oa[id0,id1,id2,id3]
        scl= res.reshape((B,W,L,L))[:,:,1,:]
        ofs= res.reshape((B,W,L,L))[:,:,0,:]
        #scl/ofs: batchsz = 128 : maxObj = 12 : location_dim = 2

        C = torch.cat([absolute[:,:,0:1],absolute[:,:,self.translation_dim-1:self.translation_dim]], axis=-1).reshape((B,1,W,L))
        #C: batchsz = 128 : ??=1 : maxObj = 12 : location_dim = 2
        C+= (ofs.reshape((B,-1,W,L,1)) * mats.reshape((B,1,W,L,L))).sum(axis=-2)

        A = locations.reshape((B,O,1,L)) - C
        #A: batchsz = 128 : maxObj*sample_dim = 96 : maxObj = 12 : location_dim = 2

        #transform A into objects' co-ordinate
        sizs = torch.cat([absolute[:,:,self.translation_dim:self.translation_dim+1],absolute[:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim]], axis=-1).reshape((B,1,W,L))
        sizs*= scl.reshape((B,1,W,L))
        
        normed = (A.reshape((B,-1,W,1,L)) * mats.reshape((B,1,W,L,L))).sum(axis=-1) / sizs
        #normed: batchsz = 128 : maxObj*sample_dim = 96 : maxObj = 12 : location_dim:2
         
        l = torch.max(torch.sqrt((normed*normed).sum(axis=-1)),0.001*torch.ones_like(normed[:,:,:,0]))  # l = |normed|  
        A*= torch.max(torch.zeros_like(l),(torch.ones_like(l) - l) / l).reshape((B,-1,W,1)) * 2.0       # lim_{l→0} H(1 - l)/l*l = H  (2.0)
        cond = self.invalidObject(absolute).repeat((1,O if self.fieldTest else 1,1,L))
        A[cond] = torch.zeros_like(A)[cond]
        #A: batchsz = 128 : maxObj*sample_dim = 96 : maxObj = 12 : location_dim = 2

        #return: batchsz = 128 : maxObj*sample_dim = 96 : location_dim = 2
        return A.sum(axis=-2) 
        #endregion: ---object-field--#

        #region: -----wall-field-----#
    def modulateResN(self,resN): #1-x²
        K2,H=(0.4)**2,5
        L = (resN*resN).sum(axis=-1)
        R = torch.max(torch.zeros_like(L),(torch.ones_like(L)*K2 - L)/K2)
        return resN * R.reshape((self.batchsz,-1,1))*H

    def dualField(self, locations, wallTensor):#self.maxObj*self.temp.shape[0]
        B,O,W,L=self.batchsz,locations.shape[1],self.maxWall,2
        if self.fieldTest:
            O = locations.shape[1]
        
        #wallTensor: batchsz = 128 : maxWall = 16 : location_dim+norm_dim = 2 + 2 = 4
        wallS = wallTensor[:,:,:2].reshape((B,1,W,L))
        wallN = torch.cat([wallTensor[:,:,3:4],wallTensor[:,:,2:3]], axis=-1).reshape((B,1,W,L))   #X-Z
        #wallS/wallN: batchsz = 128 : sample_dim = 1 : maxWall = 16 : location_dim = 2
        
        #locations: batchsz = 128 : maxObj*sample_dim = 96 : wall_dim = 1: location_dim = 2
        S = wallS - locations.reshape((B,O,1,L))
        #print(S[0])
        E = torch.cat([S[:,:,1:],S[:,:,:1]], axis=-2)
        #print(E[0])
        X = wallN*(S*wallN).sum(axis=-1).reshape((B,O,W,1))
        #print(X[0])
        #S/E/X: batchsz = 128 : maxObj*sample_dim = 96 : maxWall = 16 : location_dim = 2

        R = crs(S,X)*crs(X,E)
        #R: batchsz = 128 : maxObj*sample_dim = 96 : maxWall = 16 : location_dim = 2
        iR = (R < torch.zeros_like(R))#.reshape((B,O,W,1)).repeat((1,1,1,2))
        #print(iR[0])
        X[iR] = S[iR]

        iI = ((X*wallN).sum(axis=-1) < torch.zeros_like(R))
        DP = torch.ones_like(R)*10000
        DN = torch.ones_like(R)*(-10000)
        D = (X*X).sum(axis=-1)
        DN[iI] = (-D)[iI]
        DP[torch.logical_not(iI)] = D[torch.logical_not(iI)]
        #DN/DP: batchsz = 128 : maxObj*sample_dim = 96 : maxWall = 16

        _ ,argmaxs = DN.max(axis=-1)
        __,argmins = DP.min(axis=-1)
        #from X, extract the certain vector
        id0 = torch.arange(B).reshape((-1,1,1)).repeat(1,O,L)
        id1 = torch.arange(O).reshape((1,-1,1)).repeat(B,1,L)
        id2 = argmins.reshape((B,O,1)).repeat(1,1,L)
        id22= argmaxs.reshape((B,O,1)).repeat(1,1,L)
        id3 = torch.arange(L).reshape((1,1,-1)).repeat(B,O,1)

        resN = X[id0, id1, id22,id3]
        #resN: batchsz = 128 : maxObj*sample_dim = 96 : location_dim = 2
        resN = self.modulateResN(resN)
        resP = X[id0, id1, id2, id3]
       
        circle_ori = (torch.sign(crs(S,E)) * torch.arccos((S*E).sum(axis=-1)/((S**2).sum(axis=-1)*(E**2).sum(axis=-1))**0.5)).sum(axis=-1)
        inRoom = (torch.abs(circle_ori) > torch.ones_like(circle_ori) * 0.001)#.reshape((B,O,1)).repeat((1,1,2))
        #print(inRoom[0])
        resP[inRoom] = (torch.zeros_like(resP))[inRoom]
        resN[torch.logical_not(inRoom)] = (torch.zeros_like(resN))[torch.logical_not(inRoom)]
        return resP, resN
        #endregion: -----wall-field--#

        #region: ------all-fields----#
    def mixField(self, locations, absolute, wallTensor, mats, widos):
        B,O,W,L=self.batchsz,self.maxObj,self.maxWall,2
        #each sample point is effected by this field. But there is three types of field.
        #this function mixed the effect of these three types of field.
        #
        ofield, ifield = self.dualField(locations, wallTensor)
        ifield = ifield.reshape((B,O,-1,L))

        #how to use?
        dirs = absolute[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim].reshape((B,O,-1,self.angle_dim))
        iForce = -(ifield * dirs).sum(axis=-1).reshape((B,O,-1,1)) * ifield

        F = ofield.reshape((B,O,-1,L)) + self.objectField(locations, absolute, mats).reshape((B,O,-1,L))
        if not (widos is None):
            F += self.doorFields(locations, widos).reshape((B,O,-1,L))
        F[:,:,3:6,:] += iForce[:,:,3:6,:]#F += iForce#
        return F
        #endregion: ------all-field--#

    #endregion: ---------fields---------#

    #region: -----use-field----------#
    def flattenn(self, absolute):
        B,O,L=self.batchsz,self.maxObj,2
        #batchsz = 128 : maxObj = 12 : bbox_dim = 8
        #

        tras = torch.cat([absolute[:,:,0:1],absolute[:,:,self.translation_dim-1:self.translation_dim]], axis=-1).reshape((B,O,1,L))
        sizs = torch.cat([absolute[:,:,self.translation_dim:self.translation_dim+1],absolute[:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim]], axis=-1).reshape((B,O,1,1,L))
        coss = absolute[:,:,self.bbox_dim-2:self.bbox_dim-1]
        sins = absolute[:,:,self.bbox_dim-1:self.bbox_dim]
        mat1 = torch.cat([coss,sins], axis=-1).reshape((B,O,1,1,L))
        mat2 = torch.cat([-sins,coss], axis=-1).reshape((B,O,1,1,L))
        mats = torch.cat([mat1,mat2], axis=-2).reshape((B,O,1,L,L))
        templa = self.temp.reshape((1,1,-1,1,L))

        directions = (sizs * templa * mats).sum(axis=-1).reshape((B,O,-1,L))
        locations = directions + tras
        #batchsz = 128 : maxObj = 12 : sample_dim = 8 : location_dim = 2
        self.mats = mats.reshape((B,O,L,L))
        return directions, locations.reshape((B,-1,L)), mats.reshape((B,O,L,L))

    def unmoveables(self,absolute,hints):
        #absolute: batchsz = 128 : maxObj = 12 : obj_dim = 41?
        flag = torch.zeros_like(absolute[:,:,-1])
        #flag: batchsz = 128 : maxObj = 12
        if self.widoAsObj:
            return torch.logical_or(hints,torch.logical_or((absolute[:,:,-3]>flag),torch.logical_or(absolute[:,:,-2]>flag,absolute[:,:,-1]>flag)))
        else:
            return absolute[:,:,-1]>flag if hints is None else torch.logical_or(hints,absolute[:,:,-1]>flag)

    def synthesis(self, directions, fields, absolute, t, hints=None):
        cp = torch.clone(absolute)
        #absolute: batchsz = 128 : maxObj = 12 : bbox_dim = 8
        #directions / locations / fields: batchsz = 128 : maxObj = 12 : sample_dim = 8 : location_dim = 2
        translate = fields.mean(axis=-2)
        #directions / locations / fields: batchsz = 128 : maxObj = 12 : location_dim = 2
        normals = F.normalize(torch.cat([-directions[:,:,:,1:], directions[:,:,:,:1]],axis=-1),dim=-1)
        
        rotate = ((normals * fields).sum(axis=-1) / (directions**2).sum(axis=-1)**0.5).mean(axis=-1)
        #directions / locations / fields: batchsz = 128 : maxObj = 12
        absoluteVector = absolute[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim].reshape((self.batchsz, self.maxObj, 1, self.angle_dim))
        coss = torch.cos(rotate * t * 0.5).reshape((self.batchsz,-1,1,1))
        sins = torch.sin(rotate * t * 0.5).reshape((self.batchsz,-1,1,1))
        mat1 = torch.cat([coss,sins], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mat2 = torch.cat([-sins,coss], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mats = torch.cat([mat1,mat2], axis=-2).reshape((self.batchsz,self.maxObj,2,2))
        absoluteVector = (mats * absoluteVector).sum(axis=-1).reshape((self.batchsz, self.maxObj, self.angle_dim))
        
        mat1 = torch.cat([absoluteVector[:,:,:1],-absoluteVector[:,:,1:]], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mat2 = torch.cat([absoluteVector[:,:,1:],absoluteVector[:,:,:1]], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mats = torch.cat([mat1,mat2], axis=-2).reshape((self.batchsz,self.maxObj,1,2,2))#.repeat(1,1,self.sample_dim,1,1)
        rotatedFields = (mats * fields.reshape((self.batchsz,self.maxObj,-1,1,2))).sum(axis=-1).reshape((self.batchsz, self.maxObj, -1, 2))
        originalDirection = F.normalize(self.temp,dim=-1).reshape((1,1,-1,2))
        
        scale = (originalDirection * rotatedFields).mean(axis=-2) * (4/3) * 0.1
        
        absolute[:,:,0:1] += translate[:,:,0:1] * t.reshape((-1,1,1))
        absolute[:,:,self.translation_dim-1:self.translation_dim] += translate[:,:,1:2] * t.reshape((-1,1,1))
        #absolute[:,:,self.translation_dim:self.translation_dim+1] += scale[:,:,0:1] * t.reshape((-1,1,1))
        #absolute[:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim] += scale[:,:,1:2] * t.reshape((-1,1,1))

        #absoluteVector: batchsz = 128 : maxObj = 12 : angle_dim = 2
        absoluteVector = absolute[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim].reshape((self.batchsz, self.maxObj, 1, self.angle_dim))
        coss = torch.cos(rotate * t).reshape((self.batchsz,-1,1,1))
        sins = torch.sin(rotate * t).reshape((self.batchsz,-1,1,1))
        mat1 = torch.cat([coss,sins], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mat2 = torch.cat([-sins,coss], axis=-1).reshape((self.batchsz,self.maxObj,1,2))
        mats = torch.cat([mat1,mat2], axis=-2).reshape((self.batchsz,self.maxObj,2,2))
        absoluteVector = (mats * absoluteVector).sum(axis=-1).reshape((self.batchsz, self.maxObj, self.angle_dim))
        absolute[:,:,self.bbox_dim-self.angle_dim:self.bbox_dim] = absoluteVector

        if self.widoAsObj:
            fl = self.unmoveables(absolute,hints).reshape((self.batchsz,self.maxObj,1))
            #fl: batchsz = 128 : maxObj = 12 : obj_dim:1
            absolute[fl.repeat((1,1,absolute.shape[-1]))] = cp[fl.repeat((1,1,absolute.shape[-1]))]
            translate[fl.repeat((1,1,2))] = (torch.zeros_like(translate))[fl.repeat((1,1,2))]
            mats[fl.reshape((self.batchsz,self.maxObj,1,1)).repeat((1,1,2,2))] = (torch.zeros_like(mats))[fl.reshape((self.batchsz,self.maxObj,1,1)).repeat((1,1,2,2))]
            scale[fl.repeat((1,1,2))] = (torch.zeros_like(scale))[fl.repeat((1,1,2))]
        #denoise_out is the difference in the objects' own co-ordinates of ABSOLUTETENSOR
        return absolute, translate, mats, scale
    #endregion: --use-field----------#

    #region: -------workflow---------#
    def fulls(self, absolute, wallTensor, t, widos=None, hints=None):
        if self.scaled:
            absolute[:,:,:6] = ((absolute[:,:,:6]+1)/2)*(self.maximum-self.minimum)+self.minimum

        directions, locations, mats = self.flattenn(absolute)
        # print("locations.shape")
        # print(locations.shape)
        #print("wallTensor.shape")
        #print(wallTensor.shape)
        #print(wallTensor)
        fields = self.mixField(locations, absolute, wallTensor, mats, widos)
        # if t[0] == 0:
        #     #print(translate)
        #     #print(scale)
        #     print(wallTensor[:,:6,:])
        #     for j in range(1):
        #         for i in range(25):
        #             if absolute[j][i][-1] < 0.0:
        #                 print(absolute[j,i,:3])
        #                 print(directions[j,i,:,:].reshape((16)))
        #                 print(locations[j,i,:,:].reshape((16)))
        #                 print(fields[j,i,:,:].reshape((16)))
        #         print("separate")
            #raise NotImplementedError
        #print("fields.shape")
        #print(fields.shape)
        #print(fields)
        #raise NotImplementedError

        result, tr, ro, sc = self.synthesis(directions, fields, absolute, self.tArrangement(t), hints)
        #result = self.propogation(result, tr)
        if self.scaled:
            absolute[:,:,:6] =-1+2*((torch.clip(absolute[:,:,:6],self.minimum,self.maximum)-self.minimum)/(self.maximum-self.minimum))
        return result
    #endregion: ----workflow---------#
    
    #region: ---------debug----------#
    def printField(self, locations, wallTensor, absolute=None, widos=None):
        self.fieldTest = True
        B,O,W,L=self.batchsz,self.maxObj,self.maxWall,2
        #each sample point is effected by this field. But there is three types of field.
        #this function mixed the effect of these three types of field.
        ofield, ifield = self.dualField(locations, wallTensor)
        if not (widos is None):    
            dfield = self.doorFields(locations,widos)
            
        if not (absolute is None):
            bfield = self.objectField(locations,absolute,None)
        return dfield + ofield + ifield + bfield
    
    def printPotential(self, locations, wallTensor, W2, absolute=None, mats=None):
        self.fieldTest = True
        B=self.batchsz
        W = W2*2+1
        assert locations.shape[1] % W == 0 and (locations.shape[1] / W) % 2 == 1
        L = int(locations.shape[1] / W)
        L2 = L>>1
        #each sample point is effected by this field. But there is three types of field.
        #this function mixed the effect of these three types of field.
        #
        ofield, ifield = self.dualField(locations, wallTensor)
        F = ofield+ifield
        if not (absolute is None):
            F+= self.objectField(locations,absolute,None)
        F = F.reshape((B,W,L,2))
        P = torch.zeros((B,W,L))
        dirs = torch.tensor([[1,0],[-1,0],[0,1],[0,-1]]).reshape(4,1,1,1,2)
        for a in range(max(W2,L2)):
            up,dn,lf,rh = min(W2+a,W-1),max(W2-a,0),max(L2-a,0),min(L2+a,L-1)
            up1,dn1,lf1,rh1 = up+1,dn-1,lf-1,rh+1
            if a<W2:
                P[:,up1,lf:rh1] = P[:,up,lf:rh1]+(F[:,up,lf:rh1]*dirs[0]).sum(axis=-1)
                P[:,dn1,lf:rh1] = P[:,dn,lf:rh1]+(F[:,dn,lf:rh1]*dirs[1]).sum(axis=-1)
            
            if a<L2:
                P[:,dn:up1,rh1] = P[:,dn:up1,rh]+(F[:,dn:up1,rh]*dirs[2]).sum(axis=-1)
                P[:,dn:up1,lf1] = P[:,dn:up1,lf]+(F[:,dn:up1,lf]*dirs[3]).sum(axis=-1)
            
            if a<L2 and a<W2:
                P[:,up1,rh1] = (P[:,up1,rh]+(F[:,up1,rh]*dirs[2]).sum(axis=-1)+P[:,up,rh1]+(F[:,up,rh1]*dirs[0]).sum(axis=-1))/2.0
                P[:,up1,lf1] = (P[:,up1,lf]+(F[:,up1,lf]*dirs[3]).sum(axis=-1)+P[:,up,lf1]+(F[:,up,lf1]*dirs[0]).sum(axis=-1))/2.0
                P[:,dn1,rh1] = (P[:,dn1,rh]+(F[:,dn1,rh]*dirs[2]).sum(axis=-1)+P[:,dn,rh1]+(F[:,dn,rh1]*dirs[1]).sum(axis=-1))/2.0
                P[:,dn1,lf1] = (P[:,dn1,lf]+(F[:,dn1,lf]*dirs[3]).sum(axis=-1)+P[:,dn,lf1]+(F[:,dn,lf1]*dirs[1]).sum(axis=-1))/2.0

        return P
    #endregion: ------debug----------#