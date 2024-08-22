import torch
import torch.nn.functional as F

def crs(A,B):
    return A[:,:,:,0]*B[:,:,:,1]-A[:,:,:,1]*B[:,:,:,0]

#["pendant_lamp", "ceiling_lamp", "bookshelf", "round_end_table", "dining_table", "console_table", "corner_side_table", "desk", "coffee_table", "dressing_table", "children_cabinet", "cabinet", "shelf", "wine_cabinet", "lounge_chair", "chinese_chair", "dressing_chair", "dining_chair", "armchair", "barstool", "stool", "multi_seat_sofa", "loveseat_sofa", "l_shaped_sofa", "lazy_sofa", "chaise_longue_sofa", "wardrobe", "tv_stand", "nightstand", "double_bed", "kids_bed", "bunk_bed", "single_bed", "bed_frame"]

class motion_guidance():
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
        self.widoAsObj = widoAsObj
        #raise NotImplementedError
        pass

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

    def scale(self, x):
        return -1+2*((torch.clip(x,self.minimum,self.maximum)-self.minimum)/(self.maximum-self.minimum))

    def descale(self, x):
        return ((x+1)/2)*(self.maximum-self.minimum)+self.minimum

    def unmoveables(self,absolute):
        #absolute: batchsz = 128 : maxObj = 12 : obj_dim = 41?
        flag = torch.zeros_like(absolute[:,:,-1])
        #flag: batchsz = 128 : maxObj = 12
        return torch.logical_or((absolute[:,:,-3]>flag),torch.logical_or(absolute[:,:,-2]>flag,absolute[:,:,-1]>flag))

    def synthesis(self, directions, fields, absolute, t):
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
            fl = self.unmoveables(absolute).reshape((self.batchsz,self.maxObj,1))
            #fl: batchsz = 128 : maxObj = 12 : obj_dim:1
            absolute[fl.repeat((1,1,absolute.shape[-1]))] = cp[fl.repeat((1,1,absolute.shape[-1]))]
            translate[fl.repeat((1,1,2))] = (torch.zeros_like(translate))[fl.repeat((1,1,2))]
            mats[fl.reshape((self.batchsz,self.maxObj,1,1)).repeat((1,1,2,2))] = (torch.zeros_like(mats))[fl.reshape((self.batchsz,self.maxObj,1,1)).repeat((1,1,2,2))]
            scale[fl.repeat((1,1,2))] = (torch.zeros_like(scale))[fl.repeat((1,1,2))]
        #denoise_out is the difference in the objects' own co-ordinates of ABSOLUTETENSOR
        return absolute, translate, mats, scale

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

    def printField(self, locations, wallTensor, absolute=None, widos=None):
        self.fieldTest = True
        B,O,W,L=self.batchsz,self.maxObj,self.maxWall,2
        #each sample point is effected by this field. But there is three types of field.
        #this function mixed the effect of these three types of field.
        ofield, ifield = self.dualField(locations, wallTensor)
        if not (widos is None):    
            dfield = self.doorFields(locations,widos)
        return dfield + ofield + ifield
    
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

    def invalidObject(self,absolute):
        B,O,L=self.batchsz,self.maxObj,1
        zeo = torch.zeros_like(absolute[:,:,-1])
        #cond = torch.logical_or(torch.logical_or((absolute[:,:,-1] > zeo),(absolute[:,:,self.bbox_dim+3] > zeo)),(absolute[:,:,-9] > zeo))
        cond = torch.logical_or(torch.logical_or((absolute[:,:,-1] > zeo),(absolute[:,:,self.bbox_dim] > zeo)),(absolute[:,:,self.bbox_dim+1] > zeo))
        #cond: batchsz = 128 : maxObj = 16 : sig = 1
        condx = cond.reshape((B,1,O,L))#print(condx[0,0,:,0])
        condy = cond.reshape((B,O,1,1,L)).repeat((1,1,self.temp.shape[0],1,1)).reshape((B,-1,1,L))
        return torch.logical_or(condx.repeat((1,O*self.temp.shape[0],1,1)),condy.repeat((1,1,O,1)))

    def modulateA(self,A,normed): #(1-(x/K)²)/(x/K)²    K=0.9
        K2 = (0.9)**2
        L = (normed*normed).sum(axis=-1)
        L = torch.max(L,0.001*torch.ones_like(L)) / K2
        R = torch.max(torch.zeros_like(L),(torch.ones_like(L) - L) / L)
        return A * R.reshape((self.batchsz,-1,self.maxObj,1))

    def objectField(self, locations, absolute, mats):
        B,O,W,L=self.batchsz,self.maxObj*self.temp.shape[0],self.maxObj,2
        #locations: batchsz = 128 : maxObj*sample_dim = 96 : obj_dim = 1 : location_dim = 2
        
        C = torch.cat([absolute[:,:,0:1],absolute[:,:,self.translation_dim-1:self.translation_dim]], axis=-1).reshape((B,1,W,L))
        
        #C: batchsz = 128 : ??=1 : maxObj = 12 : location_dim = 2
        A = locations.reshape((B,O,1,L)) - C
        #A: batchsz = 128 : maxObj*sample_dim = 96 : maxObj = 12 : location_dim = 2

        #transform A into objects' co-ordinate
        sizs = torch.cat([absolute[:,:,self.translation_dim:self.translation_dim+1],absolute[:,:,self.translation_dim+self.size_dim-1:self.translation_dim+self.size_dim]], axis=-1).reshape((B,1,W,L))
        normed = (A.reshape((B,-1,W,1,L)) * mats.reshape((B,1,W,L,L))).sum(axis=-1) / sizs
        #normed: batchsz = 128 : maxObj*sample_dim = 96 : maxObj = 12 : location_dim:2
        A = self.modulateA(A,normed)
        cond = self.invalidObject(absolute).repeat((1,1,1,L))
        A[cond] = torch.zeros_like(A)[cond]
        #A: batchsz = 128 : maxObj*sample_dim = 96 : maxObj = 12 : location_dim = 2

        #A: batchsz = 128 : maxObj*sample_dim = 96 : location_dim = 2
        return A.sum(axis=-2) 
     
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

    def fulls(self, absolute, wallTensor, t, widos=None):
        if self.scaled:
            absolute[:,:,:6] = self.descale(absolute[:,:,:6])

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

        result, tr, ro, sc = self.synthesis(directions, fields, absolute, self.tArrangement(t))
        #result = self.propogation(result, tr)
        if self.scaled:
            absolute[:,:,:6] = self.scale(absolute[:,:,:6])
        return result

    def tArrangement(self,t):
        return (torch.ones_like(t) * self.maxT).reshape((-1,1)) if t[0][0] > 0 else torch.zeros_like(t).reshape((-1,1))
        return (self.maxT*(1. - t / float(len(self.betas)))).reshape((-1,1))
