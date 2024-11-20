class Fields():
    def __init__():
        pass

    def __call__():
        pass

"""


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


"""