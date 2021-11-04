import torch
from torch import nn
import torch.nn.functional as F
from vgg import vgg16
from resnet import resnet50
from attention import DANetHead

config_vgg = {'cf':[[192, 384, 768, 1536, 1536], [64, 128, 256, 512, 512]],
              'fusion': [[64, 128, 64, 3, 1], [128, 256, 128, 3, 1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2],
                         [512, 0, 512, 5, 2], [512, 0, 512, 7, 3]]}

config_resnet = {'convert': [[64, 256, 512, 1024, 2048], [128, 256, 512, 512, 512]],
                 'cf':[[384, 768, 1536, 1536], [128, 256, 512, 512]],
                 'fusion': [[128, 256, 128, 3, 1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2],
                            [512, 0, 512, 7, 3]]}

class ConvertLayer(nn.Module):
    def __init__(self, list_k):
        super(ConvertLayer, self).__init__()
        up0, up1, up2 = [], [], []
        for i in range(len(list_k[0])):
            up0.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.convert0 = nn.ModuleList(up0)

    def forward(self, list_x):
        resl = []
        for i in range(len(list_x)):
            resl.append(self.convert0[i](list_x[i]))
        return resl


class FeatureEnhance(nn.Module):
    def __init__(self, list_k, base_model):
        super(FeatureEnhance, self).__init__()
        self.list_k = list_k
        finalscoretrans, nltrans, trans, up, score, nlcatrans, edgetrans=[], [], [], [], [], [], []
        for i, ik in enumerate(list_k):
            if ik[1] > 0:
                trans.append(nn.Sequential(nn.Conv2d(ik[1], ik[0], 1, 1, bias=False), nn.ReLU(inplace=True)))

            # feature enhance
            up.append(nn.Sequential(nn.Conv2d(ik[0], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True),
                                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True),
                                    nn.Conv2d(ik[2], ik[2], ik[3], 1, ik[4]), nn.ReLU(inplace=True),
                                    ))

            score.append(nn.Conv2d(ik[2], 1, 3, 1, 1))

            nltrans.append(nn.Sequential(nn.Conv2d(512, ik[0], 1, 1, bias=False), nn.ReLU(inplace=True)))
            if base_model=='vgg':
                edgetrans.append(nn.Sequential(nn.Conv2d(128, ik[0], 1, 1, bias=False), nn.ReLU(inplace=True)))
            elif base_model=='resnet':
                edgetrans.append(nn.Sequential(nn.Conv2d(256, ik[0], 1, 1, bias=False), nn.ReLU(inplace=True)))
            if i>0:
                finalscoretrans.append(nn.Sequential(nn.Conv2d(ik[0], list_k[0][0], 1,1, bias=False)))
        self.relu = nn.ReLU()
        self.trans, self.up, self.score, self.nltrans, self.finalscoretrans, self.edgetrans = nn.ModuleList(trans), nn.ModuleList(up), nn.ModuleList(score), nn.ModuleList(nltrans), nn.ModuleList(finalscoretrans), nn.ModuleList(edgetrans)
        self.final_score = nn.Sequential(nn.Conv2d(list_k[0][0], list_k[0][0], 5, 1, 2), nn.ReLU(inplace=True), nn.Conv2d(list_k[0][0], 1, 3, 1, 1))

        if base_model=='vgg':
            self.FeatureFusionGate = FeatureFusionGate(config_vgg['cf'])
            self.EdgeExtraction = EdgeExtraction(config_vgg['fusion'], base_model)
            self.edgeh2l = nn.Sequential(nn.Conv2d(512, 128, 1, 1, bias=False), nn.ReLU(inplace=True))
        elif base_model=='resnet':
            self.FeatureFusionGate = FeatureFusionGate(config_resnet['cf'])
            self.EdgeExtraction = EdgeExtraction(config_resnet['fusion'], base_model)
            self.edgeh2l = nn.Sequential(nn.Conv2d(512, 256, 1, 1, bias=False), nn.ReLU(inplace=True))



    def forward(self, output_base, output_attention, img_size):

        final_sal, up_sal, sal_feature = [], [], []
        num_f = len(output_base)
        tmp = self.up[num_f - 1](output_base[num_f - 1])
        #Skip Connection
        #tmp = self.relu(self.up[num_f - 1](output_base[num_f - 1])+output_base[num_f - 1])

        sal_feature.append(tmp)
        up_sal.append(F.interpolate(self.score[num_f - 1](tmp), img_size, mode='bilinear', align_corners=True))
        U_tmp = tmp
        edge_feature, edge_lossreturn = self.EdgeExtraction(output_base[1] + F.interpolate((self.edgeh2l(output_base[-2])), output_base[1].size()[2:], mode='bilinear', align_corners=True), img_size)
        #edge_feature, edge_lossreturn = self.EdgeExtraction(output_base[1] + F.interpolate((self.edgeh2l(output_base[-2])), output_base[1].size()[2:], mode='bilinear', align_corners=True), img_size)
        for j in range(2, num_f+1):
            i = num_f-j
            if output_base[i].size()[1] < U_tmp.size()[1]:
                down = output_base[i]
                edge = F.interpolate((self.edgetrans[i](edge_feature[i])), output_base[i].size()[2:], mode='bilinear', align_corners=True)
                right = F.interpolate((self.trans[i](U_tmp)), output_base[i].size()[2:], mode='bilinear', align_corners=True)
                top = F.interpolate((self.nltrans[i](output_attention[i])), output_base[i].size()[2:], mode='bilinear', align_corners=True)
                cat =torch.cat([down, right*edge, top*edge], 1)
                U_tmp = self.FeatureFusionGate(i, cat)
            else:
                down = output_base[i]
                edge = F.interpolate((self.edgetrans[i](edge_feature[i])), output_base[i].size()[2:],mode='bilinear', align_corners=True)
                top = F.interpolate((self.nltrans[i](output_attention[i])), output_base[i].size()[2:], mode='bilinear',align_corners=True)
                right = F.interpolate((U_tmp), output_base[i].size()[2:], mode='bilinear', align_corners=True)
                cat = torch.cat([down, right*edge, top*edge], 1)
                U_tmp = self.FeatureFusionGate(i, cat)
            tmp = self.up[i](U_tmp)
            sal_feature.append(tmp)
            up_sal.append((F.interpolate(self.score[i](tmp), img_size, mode='bilinear', align_corners=True)))
            U_tmp = tmp

        tmp_sal_feature = sal_feature[-1]
        for i in range(len(tmp_sal_feature)-1):
            k=len(tmp_sal_feature)-2-i
            tmp_sal_feature = self.relu(torch.add(tmp_sal_feature, F.interpolate(self.finalscoretrans[i](sal_feature[k]), sal_feature[-1].size()[2:], mode='bilinear', align_corners=True)))

        final_sal.append(F.interpolate(self.final_score(tmp_sal_feature), img_size, mode='bilinear', align_corners=True))

        return final_sal, up_sal, edge_lossreturn


class FeatureFusionGate(nn.Module):
    def __init__(self, list_k):
        super(FeatureFusionGate, self).__init__()
        before, linear1, linear2, end , mid=[], [], [], [], []
        for i in range(len(list_k[0])):
            before.append(nn.Sequential(nn.Conv2d(list_k[0][i],list_k[0][i], 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
            mid.append(nn.Sequential(nn.Conv2d(list_k[0][i],list_k[0][i], 3, 1, 1, bias=False), nn.ReLU(inplace=True)))
            linear1.append(nn.Linear(list_k[0][i], list_k[0][i]//4))
            linear2.append(nn.Linear(list_k[0][i]//4, list_k[0][i]))
            end.append(nn.Sequential(nn.Conv2d(list_k[0][i], list_k[1][i], 1, 1, bias=False), nn.ReLU(inplace=True)))
        self.before, self.linear1, self.linear2, self.end, self.mid = nn.ModuleList(before), nn.ModuleList(linear1), nn.ModuleList(linear2), nn.ModuleList(end), nn.ModuleList(mid)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, i, cat):
        h, w = cat.size()[2:]
        num_c = cat.size()[1]
        cat = self.before[i](cat)
        ca = nn.AvgPool2d((h, w))(cat).view(-1, num_c)
        ca = self.linear1[i](ca)
        ca = self.relu(ca)
        ca = self.linear2[i](ca)
        ca = self.sigmoid(ca).view(-1, num_c, 1, 1).repeat(1, 1, h, w)
        result = ca * cat
        # Skip Connection
        # result = result + cat

        result = self.end[i](result)
        result = self.relu(result)
        return result

class EdgeExtraction(nn.Module):
    def __init__(self, list, base_model): #lisk: [[64, 128, 64, 3, 1], [128, 256, 128, 3, 1], [256, 512, 256, 3, 1], [512, 0, 512, 5, 2], [512, 0, 512, 5, 2], [512, 0, 512, 7, 3]]
        edge_Enhance, edge_loss=[],[]
        self.list=list
        super(EdgeExtraction,self).__init__()
        for i in range(0,(len(self.list)-1)):
            if base_model=='vgg':
                edge_Enhance.append(nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True)))
                edge_loss.append(nn.Sequential(nn.Conv2d(128, 1, 3, 1, 1)))
            elif base_model=='resnet':
                edge_Enhance.append(nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True)))
                edge_loss.append(nn.Sequential(nn.Conv2d(256, 1, 3, 1, 1)))

        self.edge_Enhance, self.edge_loss= nn.ModuleList(edge_Enhance), nn.ModuleList(edge_loss)
    def forward(self, base2, img_size):
        edge_feature, edge_lossreturn=[], []
        tmp = self.edge_Enhance[0](base2)
        edge_feature.append(tmp)
        edge_lossreturn.append(F.interpolate(self.edge_loss[0](tmp), img_size, mode='bilinear', align_corners=True))
        for i in range(1, (len(self.list)-1)):
            tmp = self.edge_Enhance[i](tmp)
            edge_feature.append(tmp)
            edge_lossreturn.append(F.interpolate(self.edge_loss[i](tmp), img_size, mode='bilinear', align_corners=True))
        return edge_feature, edge_lossreturn

class build_model(nn.Module):
    def __init__(self, base_model_cfg):
        super(build_model, self).__init__()
        self.base_model_cfg=base_model_cfg
        if base_model_cfg=='vgg':
            self.base = vgg16()
            self.nl = DANetHead(512, 512, nn.BatchNorm2d)
            self.FeatureEnhance = FeatureEnhance(config_vgg['fusion'], base_model_cfg)
        elif base_model_cfg=='resnet':
            self.base=resnet50()
            self.convert=ConvertLayer(config_resnet['convert'])
            self.nl = DANetHead(512, 512, nn.BatchNorm2d)
            self.FeatureEnhance=FeatureEnhance(config_resnet['fusion'], base_model_cfg)

    def forward(self, x):
        x_size = x.size()[2:]

        if self.base_model_cfg == 'vgg':
            self.baseout = self.base(x)
            attention1 = self.nl(self.baseout[-2])
            attention2 = self.nl(attention1[0])
            attention3 = self.nl(attention2[0])
            attention4 = self.nl(attention3[0])
            attention5 = self.nl(attention4[0])
            self.attention = [attention5[0], attention4[0], attention3[0], attention2[0], attention1[0]]


        elif self.base_model_cfg == 'resnet':
            x = self.base(x)
            self.baseout = self.convert(x)
            attention1 = self.nl(self.baseout[-2])
            attention2 = self.nl(attention1[0])
            attention3 = self.nl(attention2[0])
            attention4 = self.nl(attention3[0])
            self.attention = [attention4[0], attention3[0], attention2[0], attention1[0]]

        output = self.FeatureEnhance(self.baseout, self.attention, x_size)

        return output

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()

if __name__ == '__main__':

    im = torch.randn(1, 3, 256, 256)
    net = build_model('resnet')
    #net.load_state_dict(torch.load('/home/psxbd1/project/NLB/logs/16by16_5NLboxNogammaonetooneReOrderHighandLow1ConvEDGEXBOTH_RD30_CAFFG/models/BetterMaxFb_0.813880_epoch_40_bone.pth',map_location=torch.device('cpu')), strict=False)
    net.load_state_dict(torch.load('/db/psxbd1/SDLDNet-ResNet.pth', map_location=torch.device('cpu')), strict=False)
    final_sal_val, up_sal_val, edge_lossreturn = net(im)












