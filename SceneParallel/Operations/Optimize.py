class Optimize():
    def __init__(self,debug=True):
        #所谓的operation，指的就是我们管理这个“过程”！
        #过程需要的内容，对象，甚至超参数、全局变量，都是我的成员变量
        #只有我这个过程对象本身才具备调用他们的权限
        self.debug=True
        pass
    
    def __call__(self):
        pass
        #不是哥们，你能干啥呀，我没想明白呀


        #就你吧，我觉得你还不错！！！！
        #对，就你了
        
        #你，配置好scneTensor，配置好Images，配置好room guidance
        
        #然后实现过程的操作
        
        
"""
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
"""