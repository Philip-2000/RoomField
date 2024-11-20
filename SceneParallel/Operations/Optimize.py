class Optimize():
    def __init__(self):
        pass
    
    def __call__(self):
        pass
        #不是哥们，你能干啥呀，我没想明白呀
        
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